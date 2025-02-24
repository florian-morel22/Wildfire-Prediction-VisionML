import os
import copy
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from abc import abstractmethod
from dotenv import load_dotenv
from models import ImageEncoder
from utils import compute_metrics
from utils import WildfireDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, DBSCAN
from torchvision.tv_tensors import Image
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, davies_bouldin_score, silhouette_score
from transformers import (
    Trainer,
    ViTImageProcessor,
    TrainingArguments,
    AutoFeatureExtractor,
    ViTForImageClassification,
    AutoModelForImageClassification,
)


load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


class Method():

    @abstractmethod
    def process_data(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """
        Define the train_dataset, valid_dataset and test_dataset.
        They can be different from the original ones.
        
        Datasets are utils.WildfireDataset() instances.
        """
        pass

    @abstractmethod
    def run(debug: bool=False):
        """Train and Test the model."""
        pass

    @abstractmethod
    def save(method_name: str, path: str):
        """Save metadata and plot usefull graphs, like loss..."""
        pass

## Supervised methods

class SupervisedClassifier(Method):

    def __init__(
            self,
            classifier: nn.Module = None,
            device: str = "cpu",
            nb_epochs: int = 7,
            batch_size: int = 32,
            learning_rate: float = 1e-4
            ):

        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if classifier is None:
            self.classifier = Net(num_classes=1)
        else:
            self.classifier = classifier

        self.classifier = self.classifier.to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.criterion = nn.BCEWithLogitsLoss()

        self.metadata = {"train":[], "test": {}, "params":{}}

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
            self.classifier.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"train")
            for i, data in pbar:
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.classifier(inputs)

                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss/(i+1)}, refresh=True)

            ## VALIDATION ##
            with torch.no_grad():
                    
                self.classifier.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                pbar = tqdm(enumerate(valloader, 0), total=len(valloader), desc=f"val")
                for i, data in pbar:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.classifier(images)
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

            self.metadata['train'].append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

    def test(self) -> None:

        print(">>> TEST")
        testloader: DataLoader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        predicted = torch.empty((0, 1))
        ground_truth = torch.empty((0, 1))
        
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data
                images, labels = images.to(self.device), labels

                outputs = self.classifier(images)

                predicted = torch.cat((
                    predicted,
                    (torch.sigmoid(outputs) > 0.5).to(torch.float32).cpu()
                ), dim=0)

                ground_truth = torch.cat((
                    ground_truth, 
                    labels.view(-1, 1)
                ), dim=0)

        acc = 100 * (predicted == ground_truth).sum() / predicted.shape[0]
        print(f'Accuracy of the classifier : {acc.item():0.2f} %')

        self.metadata['test'] = {
            "accuracy": acc.item(),
            "confusion_matrix": confusion_matrix(predicted, ground_truth).tolist()
        }

    def run(
            self,
            debug: bool=False
        ) -> None:

        print(f">> Classifier : {type(self.classifier)}")

        self.train(debug)
        self.test()

    def save(self, method_name: str, path: str="./assets/"):
        """save the model, the loss plot..."""
        folder_path = path + method_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        self.metadata['params'] = {
            "learning_rate": self.learning_rate,
            "batchsize": self.batch_size,
            "classifier": str(type(self.classifier)),
            "method_name": method_name
        }

        with open(os.path.join(folder_path, 'results.json'), 'w') as f:
            json.dump(self.metadata, f)

        epochs = [
            elem["epoch"]
            for elem in self.metadata['train']
        ]
        train_loss = [
            elem["train_loss"]
            for elem in self.metadata['train']
        ]
        val_loss = [
            elem["val_loss"]
            for elem in self.metadata['train']
        ]

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, train_loss, label="train loss")
        plt.plot(epochs, val_loss, label="validation loss")
        plt.legend(fontsize=12)
        plt.title("Loss functions accross epochs")
        plt.xlabel("epochs")
        plt.savefig(os.path.join(folder_path, "loss.png"))
        
class SupervisedViT(Method):

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
            v2.ToDtype(torch.float32, scale=True),
        ])

        train, val = train_test_split(valid_df, train_size=0.8, random_state=42, shuffle=True)

        self.train_dataset = WildfireDataset(train, transform, target_transform, "s_vit")
        self.val_dataset = WildfireDataset(val, transform, target_transform, "s_vit")
        self.test_dataset = WildfireDataset(test_df, transform, target_transform, "s_vit")

        print(f">> train dataset : {len(self.train_dataset)} rows.")
        print(f">> validation dataset : {len(self.val_dataset)} rows.")
        print(f">> test dataset : {len(self.test_dataset)} rows.")

    def run(self, debug: bool=False) -> None:
        self.train(debug)
        self.test()        

    def train(self, debug: bool=False):
        # default: adamW optimizer
        training_args = TrainingArguments(
            # use_mps_device=True,
            output_dir="./vit-fire-detection",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
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
            report_to="none",
            load_best_model_at_end=True
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.feature_extractor
        )

        trainer.train()
        
        self.model = trainer.model

    def test(self):

        print(">> TEST")

        testloader: DataLoader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        predicted = torch.empty((0, 1))
        ground_truth = torch.empty((0, 1))

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images: Image = data["pixel_values"]
                labels: torch.Tensor = data["labels"]
                inputs = self.feature_extractor(images=images.clamp(0, 1), return_tensors="pt")
                inputs = inputs.to(self.device)
                outputs = self.model(**inputs)

                logits = outputs.logits
                probs = torch.sigmoid(logits)
                new_predicted = probs.argmax(dim=-1).view(-1, 1)
                new_labels = labels.argmax(dim=-1).view(-1, 1)

                predicted = torch.cat((
                    predicted,
                    new_predicted.to(torch.int64).cpu()
                ), dim=0)

                ground_truth = torch.cat((
                    ground_truth, 
                    new_labels.to(torch.int64)
                ), dim=0)

        acc = 100 * (predicted == ground_truth).sum() / predicted.shape[0]

        print(f'Accuracy of the classifier : {acc.item():0.2f} %')

    def save(self, method_name: str, path: str="./assets/"):
        """save the model, the loss plot..."""

        print(">>> SAVING MODEL")
        method_path = os.path.join(path, method_name)
        os.makedirs(method_path, exist_ok=True)
        try:
            self.model.save_pretrained(method_path)
            self.feature_extractor.save_pretrained(method_path)
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

## Semi-Supervised methods

class SemiSupervisedSelfTraining(Method):

    def __init__(
            self,
            classifier: nn.Module = None,
            device: str = "cpu",
            nb_epochs: int = 5,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            confidence_rate: float = 0.85,
            max_pseudo_label: int = 500, # Maximum number of samples to pseudo label
            steps: int = 5
            ):
        
        self.classifier = classifier
        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.confidence_rate = confidence_rate
        self.max_pseudo_label = max_pseudo_label
        self.steps = steps

        self.metadata = {"train": {}, "test": {}, "params":{}}

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

        self.unlabeled_dataset = WildfireDataset(train_df, self.transform, method_name="ss_selftraining")
        self.train_dataset = WildfireDataset(valid_train_df, self.transform)
        self.val_dataset = WildfireDataset(valid_valid_df, self.transform)
        self.test_dataset = WildfireDataset(test_df, self.transform)

        print(f">> unlabeled_dataset : {len(self.unlabeled_dataset)} rows.")
        print(f">> train dataset : {len(self.train_dataset)} rows.")
        print(f">> validation dataset : {len(self.val_dataset)} rows.")
        print(f">> test dataset : {len(self.test_dataset)} rows.\n")

    def update_labels(self, trained_classifier: nn.Module, step: int, debug:bool):

        unlabeledloader = DataLoader(self.unlabeled_dataset, batch_size=self.batch_size, num_workers=4)

        with torch.no_grad():
                
            trained_classifier.eval()
            results = []

            pbar = tqdm(enumerate(unlabeledloader, 0), total=len(unlabeledloader), desc=f"unlabeled")
            for _, data in pbar:
                images, labels, indexs = data
                images = images.to(torch.float32).to(self.device)
                labels = labels.to(torch.float32).to(self.device)

                outputs = trained_classifier(images)
                sigm_outputs = nn.functional.sigmoid(outputs)

                for idx, out in zip(indexs, sigm_outputs):
                    results.append({
                        "index": idx.item(),
                        "pseudo_label": int((out > 0.5).cpu().item()),
                        "confidence": abs((out - 0.5)*2).cpu().item()
                    })

        results_df = pd.DataFrame(results, columns=["index", "pseudo_label", "confidence"])

        high_confident: pd.DataFrame = results_df[results_df["confidence"] > self.confidence_rate]
        
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
        print(f">> Newly generated 0 pseudo labels : {new_labels_0} ({new_labels_0/new_labels.sum()*100 if new_labels.sum()>0 else 0:0.2f} %)")
        print(f">> Newly generated 1 pseudo labels : {new_labels_1} ({new_labels_1/new_labels.sum()*100 if new_labels.sum()>0 else 0:0.2f} %)")

        # Update train_dataset and unlabeled_dataset
        self.unlabeled_dataset = WildfireDataset(unlabeled_df, self.transform, method_name="ss_selftraining")
        self.train_dataset = WildfireDataset(pd.concat([train_df, pseudolabeled_df]), self.transform)

        self.metadata['train'][step]["update_label"] = {
            "new_pseudo_labels_0": int(new_labels_0),
            "new_pseudo_labels_1": int(new_labels_1)
        }

    def train(self, debug: bool=False) -> None:

        for step in range(self.steps if not debug else 2):

            nb_unlabeled_data = len(self.unlabeled_dataset)
            print(f"\n>>>>> \033[33mSTEP {step+1} - {nb_unlabeled_data} unlabeled data\033[0m")
            print(f">> learning rate : {self.learning_rate}")

            sub_method = SupervisedClassifier(
                classifier=copy.deepcopy(self.classifier),
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
            self.metadata['train'][step] = sub_method.metadata

            # self.learning_rate *= 0.95

            if step < self.steps-1:
                # Useless at the last step because we don't retrain the model
                self.update_labels(trained_classifier=sub_method.classifier, step=step, debug=debug)
                del sub_method



        self.classifier = sub_method.classifier

    def test(self) -> None:

        print(">>> TEST")
        testloader: DataLoader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        predicted = torch.empty((0, 1))
        ground_truth = torch.empty((0, 1))
        
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data
                images, labels = images.to(self.device), labels

                outputs = self.classifier(images)

                predicted = torch.cat((
                    predicted,
                    (torch.sigmoid(outputs) > 0.5).to(torch.float32).cpu()
                ), dim=0)

                ground_truth = torch.cat((
                    ground_truth, 
                    labels.view(-1, 1)
                ), dim=0)

        acc = 100 * (predicted == ground_truth).sum() / predicted.shape[0]
        print(f'Accuracy of the classifier : {acc.item():0.2f} %')

        self.metadata['test'] = {
            "accuracy": acc.item(),
            "confusion_matrix": confusion_matrix(predicted, ground_truth).tolist()
        }

    def run(self, debug: bool=False):
        self.train(debug)
        self.test()

    def save(self, method_name: str, path: str="./assets/"):
        """save the model, the loss plot..."""
        folder_path = path + method_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        self.metadata['params'] = {
            "learning_rate": self.learning_rate,
            "batchsize": self.batch_size,
            "classifier": str(type(self.classifier)),
            "method_name": method_name
        }

        with open(os.path.join(folder_path, 'results.json'), 'w') as f:
            json.dump(self.metadata, f)


        nb_epochs = len(self.metadata['train'][0]['train'])

        cmap = plt.get_cmap("tab20")
        plt.figure(figsize=(9, 5))
        for step in self.metadata['train'].keys():
            train_loss = [
                elem['train_loss']
                for elem in self.metadata['train'][step]['train']
            ]
            val_loss = [
                elem['val_loss']
                for elem in self.metadata['train'][step]['train']
            ]
            plt.plot(range(nb_epochs), train_loss, linestyle='dashed', color=cmap(step), label=f"train loss - step {step}")
            plt.plot(range(nb_epochs), val_loss, color=cmap(step), label=f"validation loss - step {step}")
        
        plt.legend(fontsize=12)
        plt.title("Loss functions accross epochs and steps")
        plt.xlabel("epochs")
        plt.savefig(os.path.join(folder_path, "loss.png"))

class SemiSupervisedClustering(Method):

    def __init__(
            self,
            encoder: ImageEncoder,
            device: str = "cpu",
            algo: str = "kmeans",
            nb_epochs: int = 7,
            batch_size: int = 32,
            learning_rate: float = 0.01,
            classifier: nn.Module = None,
            load_encoded_from_hf: bool = False, #The encoding is not run, data is fetched from Hugging Face
            hf_encoded_dataset_id: str = None,
            **kwargs
    ):
        self.device = device
        self.encoder = encoder
        self.clustering_model = None
        self.learning_rate = learning_rate
        self.algo = algo
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.classifier = classifier
        self.load_encoded_from_hf = load_encoded_from_hf
        self.hf_encoded_dataset_id = hf_encoded_dataset_id

        if self.load_encoded_from_hf:
            assert self.hf_encoded_dataset_id, "If you want to load encoded images from huggingface, please provide a dataset id as hf_encoded_dataset_id attribute."

        self.sub_method = SupervisedClassifier(
            classifier=self.classifier,
            device=self.device,
            nb_epochs=self.nb_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate
        )

        self.kwargs = kwargs

        self.metadata = {"train": {}, "test": {}, "params": {}}

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
            dataset = load_dataset(self.hf_encoded_dataset_id, token=HF_TOKEN)
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
            dataset.push_to_hub(self.hf_encoded_dataset_id, token=HF_TOKEN, private=False)
        
        print("")
        return encoded_images, true_labels

    def generate_synthetic_annotation(self, debug: bool):

        print(f">> Encoder : {type(self.encoder)}")
        if self.load_encoded_from_hf and not debug:
            encoded_images, true_labels = self.load_data_from_hf()
        else:
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

        clustering_score = davies_bouldin_score(encoded_images, predicted_clusters)
        print("silouhettescore")
        self.metadata["clustering"] = {
            "davies_bouldin_score": float(clustering_score),
            "clusters": []
        }
        print(f">> davies bouldin score : {clustering_score :.2f}")

        cluster2label = self.assign_label_to_cluster(n_clusters, true_labels, predicted_clusters)

        for i, (cluster, true_label) in enumerate(zip(predicted_clusters, true_labels)):
            if true_label == -1:
                new_label = cluster2label[cluster]
                self.train_dataset.update_label(i, new_label)

    def run(self, debug: bool=False):
        self.generate_synthetic_annotation(debug)

        self.sub_method.process_data(
            self.train_dataset.dataframe,
            self.val_dataset.dataframe,
            self.test_dataset.dataframe,
            use_train_data=True,
        )
        self.sub_method.run()

        self.metadata['train'] = self.sub_method.metadata['train']
        self.metadata['test'] = self.sub_method.metadata['test']

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

                self.metadata["clustering"]["clusters"].append({
                    "id": cluster,
                    "label": int(np.argmax(count_labels)),
                    "perc. of labeled data": float(nb_labeled_data/nb_data*100),
                    "label homogeneity": float(label_homogeneity),
                    "nb data": int(nb_data)
                })

            else:
                print(f"WARNING : cluster {cluster} has no labled data.")
                cluster2label[cluster] = 1 # randomly assigned
        
        print("")
        return cluster2label

    def save(self, method_name: str, path: str="./assets/", debug: bool=False):
        """save the model, the loss plot..."""
        folder_path = path + method_name + "_debug" if debug else path + method_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        self.metadata['params'] = {
            "learning_rate": self.learning_rate,
            "batchsize": self.batch_size,
            "classifier": str(type(self.classifier)),
            "method_name": method_name
        }

        with open(os.path.join(folder_path, 'results.json'), 'w') as f:
            json.dump(self.metadata, f)

        epochs = [
            elem["epoch"]
            for elem in self.metadata['train']
        ]
        train_loss = [
            elem["train_loss"]
            for elem in self.metadata['train']
        ]
        val_loss = [
            elem["val_loss"]
            for elem in self.metadata['train']
        ]

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, train_loss, label="train loss")
        plt.plot(epochs, val_loss, label="validation loss")
        plt.legend(fontsize=12)
        plt.title("Loss functions accross epochs")
        plt.xlabel("epochs")
        plt.savefig(os.path.join(folder_path, "loss.png"))

## Unsupervised methods

class UnsupervisedClustering(Method):

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
        self.test_dataset = WildfireDataset(test_df, transform)
        
    def run(self, debug: bool=False):
        """
        Apply a clustering algorithm to the extracted embeddings.
        """

        self.encoded_images, self.labels_true = self.encoder.encode_images(self.test_dataset)

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
        
        self.labels_kmeans = self.clustering_model.fit_predict(self.encoded_images)

        print(f"labels kmean: {self.labels_kmeans}")

        self.accuracy = sum(self.labels_true==self.labels_kmeans)/len(self.labels_true)

        print(f"Accuracy: {self.accuracy * 100:.2f}%")

    def save(self, method_name: str, path: str="./assets/"):
        pass
