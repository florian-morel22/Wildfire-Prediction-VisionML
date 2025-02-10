import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
from utils import load_data
from utils import WildfireDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from models import Net, ImageEncoder
from sklearn.model_selection import train_test_split

# ViT
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from utils import compute_metrics
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from abc import abstractmethod


# ViT unsupervised: get embeddings
import numpy as np
from sklearn.cluster import KMeans, DBSCAN



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
            batch_size: int = 50,
            learning_rate: float = 1e-2
            ):

        self.device = device
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if network is None:
            self.network = Net(num_classes=1)
            self.network = self.network.to(self.device)

        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

    def process_data(
            self,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            test_df: pd.DataFrame,
        ) -> None:
        
        transform = v2.Compose([
            v2.ToImage(), # Convert into Image tensor
            v2.ToDtype(torch.float32, scale=True)
        ])

        train, val = train_test_split(valid_df, train_size=0.8, random_state=42, shuffle=True)

        self.train_dataset = WildfireDataset(train, transform)
        self.val_dataset = WildfireDataset(val, transform)
        self.test_dataset = WildfireDataset(test_df, transform)

    def train(self, debug: bool=False) -> None:
        
        trainloader: DataLoader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)

        print(">>> TRAIN")

        nb_epochs = self.nb_epochs if not debug else 2
        for epoch in range(nb_epochs):

            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"epoch {epoch}")
            for i, data in pbar:
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss/(i+1)}, refresh=True)

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

        self.train(debug)
        self.test()

    def save(self):
        """save the model, the loss plot..."""
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

    def run(self, debug: bool=False) -> None:
        
        print(">>> TRAIN")

        # default: adamW optimizer
        training_args = TrainingArguments(
            # use_mps_device=True,
            output_dir="./vit-fire-detection",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            num_train_epochs=self.nb_epochs if not debug else 1,

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

        self.train_dataset = WildfireDataset(valid_df, transform)
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
