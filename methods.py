import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv
from utils import load_data
from datasets import load_dataset, Dataset
from utils import WildfireDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from models import Net, ImageEncoder, resnet_classifier
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50

# ViT
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from utils import compute_metrics
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from abc import abstractmethod


# ViT unsupervised: get embeddings
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


class Method():

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def run(debug: bool=False):
        pass

    @abstractmethod
    def save():
        pass

class BasicCNN():

    def __init__(
            self,
            network: nn.Module = None,
            device: str = "cpu",
            nb_epochs: int = 7,
            batch_size: int = 32,
            learning_rate: float = 1e-3
            ):

        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if network is None:
            self.network = Net(num_classes=1)
        else:
            self.network = network

        self.network = self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.criterion = nn.BCEWithLogitsLoss()

    def process_data(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            test_df: pd.DataFrame,
            use_train_data: bool=False
        ) -> None:
        
        transform = v2.Compose([
            v2.ToImage(), # Convert into Image tensor
            v2.ToDtype(torch.float32, scale=True)
        ])

        if not use_train_data:
            train_df, valid_df = train_test_split(valid_df, test_size=0.2, shuffle=True, random_state=42) #split valid in new train/valid

        self.train_dataset = WildfireDataset(train_df, transform)
        self.val_dataset = WildfireDataset(valid_df, transform)
        self.test_dataset = WildfireDataset(test_df, transform)

        print(f">> train dataset : {len(self.train_dataset)} rows.")
        print(f">> validation dataset : {len(self.val_dataset)} rows.")
        print(f">> test dataset : {len(self.test_dataset)} rows.\n")

    def train(self, debug: bool=False) -> None:
        
        trainloader: DataLoader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)
        valloader: DataLoader = DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=4)

        print(">>> TRAIN")

        nb_epochs = self.nb_epochs if not debug else 2
        for epoch in range(nb_epochs):

            ## TRAIN ##
            self.network.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"train")
            for i, data in pbar:
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.network(inputs)

                # ## TODELETE ##
                # sigm_outputs = nn.functional.sigmoid(outputs)
                # print(min(sigm_outputs))
                # print(max(sigm_outputs))
                # ##############


                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss/(i+1)}, refresh=True)

            ## VALIDATION ##
            with torch.no_grad():
                    
                self.network.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                pbar = tqdm(enumerate(valloader, 0), total=len(valloader), desc=f"val")
                for i, data in pbar:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.network(images)
                    loss = self.criterion(outputs, labels.view(-1, 1))
                    val_loss += loss.item()
                    
                    # Compute accuracy
                    predicted = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                    labels = labels.view_as(predicted)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Print statistics
            train_loss = running_loss / len(trainloader)
            val_loss /= len(valloader)
            val_accuracy = 100 * correct / total

            self.scheduler.step(val_loss)

            print(f"Epoch [{epoch+1}/{self.nb_epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    def test(self) -> None:

        correct = 0
        total = 0

        print(">>> TEST")
        print("TODO: Voir autres scores")
        testloader: DataLoader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.network(images)

                predicted = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                labels = labels.view_as(predicted)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network : {100 * correct / total} %')

    def run(
            self,
            debug: bool=False
        ) -> None:

        print(f">> Classifier : {type(self.network)}")

        self.train(debug)
        self.test()

    def save(self):
        """save the model, the loss plot..."""
        pass

class SemiSupervisedCNN():

    def __init__(
            self,
            network: nn.Module = None,
            device: str = "cpu",
            nb_epochs: int = 5,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            confidence_rate: float = 0.85,
            max_pseudo_label: int = 500, # Maximum number of samples to pseudo label?
            steps: int = 5
            ):
        
        self.network = network
        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.confidence_rate = confidence_rate
        self.max_pseudo_label = max_pseudo_label
        self.steps = steps

    def process_data(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            test_df: pd.DataFrame,
        ) -> None:
        
        self.transform = v2.Compose([
            v2.ToImage(), # Convert into Image tensor
            v2.ToDtype(torch.float32, scale=True)
        ])

        valid_train_df, valid_valid_df = train_test_split(valid_df, test_size=0.2, shuffle=True, random_state=42) #split valid in new train/valid

        self.unlabeled_dataset = WildfireDataset(train_df, self.transform, method_name="semisupervised_cnn")
        self.train_dataset = WildfireDataset(valid_train_df, self.transform)
        self.val_dataset = WildfireDataset(valid_valid_df, self.transform)
        self.test_dataset = WildfireDataset(test_df, self.transform)

        print(f">> unlabeled_dataset : {len(self.unlabeled_dataset)} rows.")
        print(f">> train dataset : {len(self.train_dataset)} rows.")
        print(f">> validation dataset : {len(self.val_dataset)} rows.")
        print(f">> test dataset : {len(self.test_dataset)} rows.\n")

    def update_labels(self, trained_network: nn.Module):

        unlabeledloader = DataLoader(self.unlabeled_dataset, batch_size=self.batch_size, num_workers=4)

        with torch.no_grad():
                
            trained_network.eval()
            results = []

            pbar = tqdm(enumerate(unlabeledloader, 0), total=len(unlabeledloader), desc=f"unlabeled")
            for _, data in pbar:
                images, labels, indexs = data
                images = images.to(torch.float32).to(self.device)
                labels = labels.to(torch.float32).to(self.device)

                outputs = trained_network(images)
                sigm_outputs = nn.functional.sigmoid(outputs)

                for idx, out in zip(indexs, sigm_outputs):
                    results.append({
                        "index": idx.item(),
                        "pseudo_label": int((out > 0.5).cpu().item()),
                        "confidence": abs((out - 0.5)*2).cpu().item()
                    })

        results_df = pd.DataFrame(results)
        
        high_confident: pd.DataFrame = results_df[results_df["confidence"]> self.confidence_rate]
        
        # We want to add 50% label 0, 50% label 1
        high_confident_label0 = high_confident[high_confident["pseudo_label"]==0]
        high_confident_label0 = high_confident_label0.sort_values("confidence", ascending=False)
        high_confident_label0 = high_confident_label0[:self.max_pseudo_label//2]
        
        high_confident_label1 = high_confident[high_confident["pseudo_label"]==1]
        high_confident_label1 = high_confident_label1.sort_values("confidence", ascending=False)
        high_confident_label1 = high_confident_label1[:self.max_pseudo_label//2]

        high_confident = pd.concat([high_confident_label0, high_confident_label1])

        for _, row in high_confident.iterrows():
            self.unlabeled_dataset.update_label(row["index"], row["pseudo_label"])

        train_df = self.train_dataset.dataframe
        unlabeled_df = self.unlabeled_dataset.dataframe
        pseudolabeled_df = unlabeled_df[unlabeled_df["label"]!=-1]
        unlabeled_df = unlabeled_df[unlabeled_df["label"]==-1]

        # print stats on newly generated pseudo labels
        new_labels = pseudolabeled_df["label"].value_counts()
        new_labels_0 = new_labels[0] if 0 in new_labels.index else 0
        new_labels_1 = new_labels[1] if 1 in new_labels.index else 0
        print(f">> Newly generated 0 pseudo labels : {new_labels_0} ({new_labels_0/new_labels.sum()*100:0.2f} %)")
        print(f">> Newly generated 1 pseudo labels : {new_labels_1} ({new_labels_1/new_labels.sum()*100:0.2f} %)")

        # Update train_dataset and unlabeled_dataset
        self.unlabeled_dataset = WildfireDataset(unlabeled_df, self.transform, method_name="semisupervised_cnn")
        self.train_dataset = WildfireDataset(pd.concat([train_df, pseudolabeled_df]), self.transform)

    def train(self, debug: bool=False) -> None:

        for step in range(self.steps):

            nb_unlabeled_data = len(self.unlabeled_dataset)
            print(f"\n>>>>> \033[33mSTEP {step+1} - {nb_unlabeled_data} unlabeled data\033[0m")

            sub_method = BasicCNN(
                network=self.network,
                device=self.device,
                nb_epochs=self.nb_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )

            sub_method.process_data(
                self.train_dataset.dataframe,
                self.val_dataset.dataframe, 
                self.test_dataset.dataframe,
                use_train_data=True
            )
            sub_method.train(debug)

            if step < self.steps-1:
                # Useless at the last step because we don't retrain the model
                self.update_labels(trained_network=sub_method.network)

        self.network = sub_method.network

    def test(self) -> None:

        correct = 0
        total = 0

        print(">>> TEST")
        testloader: DataLoader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.network(images)

                predicted = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                labels = labels.view_as(predicted)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network : {100 * correct / total} %')

    def run(self, debug: bool=False):
        self.train(debug)
        self.test()

    def save(self):
        pass

class ViT():

    def __init__(
            self,
            model_name: str = "google/vit-base-patch16-224-in21k",
            device: str = "cpu",
            nb_epochs: int = 50,
            batch_size: int = 50,
            learning_rate: float = 1e-2
            ):

        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)

        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels = 2, 
            id2label = {0:'Normal', 1:'Fire'}, 
            label2id = {'Normal':0, 'Fire':1}
            )
        self.model.to(device)

    def process_data(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            test_df: pd.DataFrame,
    ):
        
        feature_extractor_ = self.feature_extractor
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # Scale pixel values to [0, 1]
            v2.Resize((224, 224)),  # Resize images to 224x224
            v2.Normalize(
                mean=torch.tensor(feature_extractor_.image_mean, dtype=torch.float32).tolist(),
                std=torch.tensor(feature_extractor_.image_std, dtype=torch.float32).tolist()
            ),  # Normalize with float32
        ])

        target_transform = v2.Compose([
            v2.Lambda(lambda target: torch.tensor([1., 0.]) if target == 0 else torch.tensor([0., 1.])),
            v2.ToDtype(torch.float32, scale=True)
        ])

        train, val = train_test_split(valid_df, train_size=0.8, random_state=42, shuffle=True)

        self.train_dataset = WildfireDataset(train, transform, target_transform, "vit")
        self.val_dataset = WildfireDataset(val, transform, target_transform, "vit")
        self.test_dataset = WildfireDataset(test_df, transform, target_transform, "vit")

        print(f">> train dataset : {len(self.train_dataset)} rows.")
        print(f">> validation dataset : {len(self.val_dataset)} rows.")
        print(f">> test dataset : {len(self.test_dataset)} rows.")

    def run(self, debug: bool=False) -> None:
        
        print(">>> TRAIN & TEST VIT")

        # default: adamW optimizer
        training_args = TrainingArguments(
            # use_mps_device=True,
            output_dir="./vit-fire-detection",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            num_train_epochs=self.nb_epochs if not debug else 2,

            learning_rate=self.learning_rate,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            warmup_steps=100,

            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            load_best_model_at_end=True
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            processing_class=self.feature_extractor
        )
        trainer.train()
        try:
        # Train the model
            trainer.train()
        except Exception as e:
            print(f"Exception. {e}")

    def save(self, save_path: str = "./vit-fire-detection"):
        """save the model, the loss plot..."""
        print(">>> SAVING MODEL")

        os.makedirs(save_path, exist_ok=True)
        try:
            self.model.save_pretrained(save_path)
            self.feature_extractor.save_pretrained(save_path)
        except Exception as e:
            print(f"Error while saving the model: {e}")

    def load_model(self, save_path: str = None):
        """
        Load the vit model for inference
        """

        self.model = AutoModelForImageClassification.from_pretrained(save_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(save_path)

    def infer(self, image_path: str, model_path: str = "./vit-fire-detection"):
        """
        Perform inference using a saved ViT model.
        :param image_path: Path to the image to infer.
        :param model_path: Path where the model and feature extractor are saved.
        :return: Predicted label and probabilities.
        """
        
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Perform inference
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class and probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = probs.argmax(dim=-1).item()
        
        return {
            "predicted_class": predicted_class,
            "probabilities": probs.tolist()[0]
        }

class BasicClustering():

    def __init__(
            self,
            encoder: ImageEncoder,
            device: str = "cpu",
            algo: str = "kmeans",
            **kwargs
        ):
        
        self.device = device
        self.encoder = encoder
        self.clustering_model = None
        self.algo = algo

        self.kwargs = kwargs

    def process_data(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        
        transform = None

        self.train_dataset = WildfireDataset(valid_df.sample(frac=1, random_state=42), transform)
        self.test_dataset = WildfireDataset(test_df, transform)
        
    def run(self, debug: bool=False):
        """
        Apply a clustering algorithm to the extracted embeddings.
        :param embeddings: Matrix of embeddings (numpy array).
        :param method: Clustering method ('kmeans' or 'dbscan').
        :param kwargs: Additional parameters for the clustering algorithm (e.g., n_clusters for K-Means).
        :return: Cluster labels for the embeddings.
        """

        encoded_images, labels_true = self.encoder.encode_images(self.train_dataset)

        if self.algo == "kmeans":
            n_clusters = self.kwargs.get("n_clusters", 2)
            print(f"Clustering with K-Means (n_clusters={n_clusters})")
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        
        elif self.algo == "dbscan":
            eps = self.kwargs.get("eps", 0.5)
            min_samples = self.kwargs.get("min_samples", 5)
            print(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})")
            self.clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
        
        else:
            raise ValueError("Invalid clustering method. Choose 'kmeans' or 'dbscan'.")
        
        labels_kmeans = self.clustering_model.fit_predict(encoded_images)

        print(f"labels kmean: {labels_kmeans}") 

        accuracy = sum(labels_true==labels_kmeans)/len(labels_true)

        print(f"Accuracy: {accuracy * 100:.2f}%")

    def save(self):
        pass

class AdvancedClustering():

    def __init__(
            self,
            encoder: ImageEncoder,
            device: str = "cpu",
            algo: str = "kmeans",
            nb_epochs: int = 7,
            batch_size: int = 32,
            learning_rate: float = 0.01,
            classifier: nn.Module = None,
            **kwargs
    ):
        self.device = device
        self.encoder = encoder
        self.clustering_model = None
        self.algo = algo

        self.sub_method = BasicCNN(
            network=classifier,
            device=self.device,
            nb_epochs=nb_epochs,
            batch_size = batch_size,
            learning_rate=learning_rate
        )

        self.kwargs = kwargs

    def process_data(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        
        transform = None

        valid_train_df, valid_valid_df = train_test_split(valid_df, test_size=0.2, shuffle=True, random_state=42) #split valid in new train/valid

        self.train_dataset = WildfireDataset(pd.concat([train_df, valid_train_df]), transform)
        self.val_dataset = WildfireDataset(valid_valid_df, transform)
        self.test_dataset = WildfireDataset(test_df, transform)

        print(f">> train dataset : {len(self.train_dataset)} rows.")
        print(f">> validation dataset : {len(self.val_dataset)} rows.")
        print(f">> test dataset : {len(self.test_dataset)} rows.\n")

    def load_data_from_hf(self):
        """Load encoded images from the huggingface hub."""
        try:
            print(f">> Loading encoded images from the hub. (WARNING : ensure random state is 42 in the train_test_split() of process_data())")
            dataset = load_dataset("florian-morel22/cvproject-vit-encoding", token=HF_TOKEN)
            train_dataset: Dataset = dataset['train']
            encoded_images = np.array([train_dataset[i]['image'] for i in range(train_dataset.shape[0])])
            true_labels = np.array([train_dataset[i]['label'] for i in range(train_dataset.shape[0])])

        except Exception as e:
            print(f">> Impossible to load the encoded images from the hub. ({e})")
            encoded_images, true_labels = self.encoder.encode_images(self.train_dataset)
            dataset = Dataset.from_dict({
                "image": encoded_images,
                "label": true_labels
            })
            print(">> Push the encoded images dataset to the hub.")
            dataset.push_to_hub("florian-morel22/cvproject-vit-encoding", token=HF_TOKEN, private=False)
        
        print("")
        return encoded_images, true_labels

    def generate_synthetic_annotation(self, load_from_hf: bool=False):

        if load_from_hf:
            encoded_images, true_labels = self.load_data_from_hf()
        else:
            print(f">> Encoder : {type(self.encoder)}")
            encoded_images, true_labels = self.encoder.encode_images(self.train_dataset)

        # Clustering_algo

        if self.algo == "kmeans":
            n_clusters = self.kwargs.get("n_clusters", 2)
            print(f">> Clustering with K-Means (n_clusters={n_clusters})")
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        
        elif self.algo == "dbscan":
            eps = self.kwargs.get("eps", 0.5)
            min_samples = self.kwargs.get("min_samples", 5)
            print(f">> Clustering with DBSCAN (eps={eps}, min_samples={min_samples})")
            self.clustering_model = DBSCAN(eps=eps, min_samples=min_samples)

        else:
            raise ValueError("Invalid clustering method. Choose 'kmeans' or 'dbscan'.")
        
        predicted_clusters = self.clustering_model.fit_predict(encoded_images)

        cluster2label = self.assign_label_to_cluster(n_clusters, true_labels, predicted_clusters)

        for i, (cluster, true_label) in enumerate(zip(predicted_clusters, true_labels)):
            if true_label == -1:
                new_label = cluster2label[cluster]
                self.train_dataset.update_label(i, new_label)

    def run(self, debug: bool=False):
        self.generate_synthetic_annotation(load_from_hf=True if not debug else False)

        self.sub_method.process_data(
            self.train_dataset.dataframe,
            self.val_dataset.dataframe,
            self.test_dataset.dataframe,
            use_train_data=True,
        )
        self.sub_method.run()

    def save(self):
        pass

    def assign_label_to_cluster(
            self,
            n_clusters: str,
            true_labels: np.ndarray,
            predicted_clusters: np.ndarray,            
        ) -> list[dict]:

        cluster2label: dict = {}

        print("Cluster | label | % of labeled data | Label homogeneity | Nb of data")
        print("=========================================================================")

        for cluster in range(n_clusters):
            cluster_labels: np.ndarray = true_labels[(predicted_clusters==cluster) & (true_labels!=-1)]
            cluster_labels = cluster_labels.astype(np.int32)

            if cluster_labels.size > 0: # cluster_labels not empty

                count_labels = np.bincount(cluster_labels)
                label_homogeneity = max(count_labels)/sum(count_labels) *100
                nb_data = sum(predicted_clusters==cluster)
                nb_labeled_data = sum((predicted_clusters==cluster) & (true_labels!=-1))
                print(f"{cluster : 03d}     | {np.argmax(count_labels): 02d}    | {nb_labeled_data/nb_data*100 :.2f} %           | {label_homogeneity :.2f} %           | {nb_data}")


                cluster2label[cluster] = np.argmax(count_labels)
            else:
                print(f"WARNING : cluster {cluster} has no labled data.")
                cluster2label[cluster] = 1 # randomly assigned
        
        print("")
        return cluster2label
