import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from models import Net

# ViT
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification, TrainingArguments, Trainer
from utils import compute_metrics
from datasets import DatasetDict
import os
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image

# ViT unsupervised: get embeddings
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

class BasicCNN():

    def __init__(
            self,
            network: nn.Module = None,
            device: str = "cpu",
            ):

        self.device = device

        if network is None:
            self.network = Net(num_classes=1)
            self.network = self.network.to(self.device)

        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(
            self,
            train_dataset,
            nb_epochs: int,
            batch_size=50
    ) -> None:
        
        trainloader: DataLoader = train_dataset.__dataloader__(batch_size, num_workers=4)

        print(">>> TRAIN")
        
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


    def test(
            self,
            test_dataset: DataLoader,
            batch_size: int = 50
    ) -> None:
        correct = 0
        total = 0

        print(">>> TEST")
        print("TODO: Voir autres scores")
        testloader: DataLoader = test_dataset.__dataloader__(batch_size, num_workers=4)
        
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

    def train_and_test(
            self,
            train_dataset,
            test_dataset,
            nb_epochs: int = 7,
            batch_size: int = 50,
            learning_rate: float = 1e-2
        ) -> None:

        self.train(train_dataset, nb_epochs=nb_epochs, batch_size=batch_size)
        self.test(test_dataset, batch_size=batch_size)
            
    def save(self):
        """save the model, the loss plot..."""
        pass


class ViT():

    def __init__(
            self,
            model_name: str = "google/vit-base-patch16-224-in21k",
            device: str = "cpu",
            ):

        self.device = device
        print(device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels = 2, 
            id2label = {0:'Normal', 1:'Fire'}, 
            label2id = {'Normal':0, 'Fire':1}
            )
        self.model.to(device)


    def train_and_test(
            self,
            train_dataset,
            test_dataset,
            nb_epochs: int,
            batch_size: int = 50,
            learning_rate: float = 1e-2
        ) -> None:
        
        print(">>> TRAIN")

        # default: adamW optimizer
        training_args = TrainingArguments(
            # use_mps_device=True,
            output_dir="./vit-fire-detection",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            num_train_epochs=nb_epochs,

            learning_rate=learning_rate,
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
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=self.feature_extractor
        )

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
       

class ViTUnsupervised():

    def __init__(
            self,
            model_name: str = "google/vit-base-patch16-224-in21k",
            device: str = "cpu",
            ):
        """
        Initialize the ViT class for unsupervised tasks.
        The model extracts embeddings instead of performing classification.
        """
        self.device = device
        print(f"Using device: {device}")
        
        # Load the feature extractor and the base ViT model (without classification head)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)  # Base ViT model
        self.model.to(device)

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract the embedding for a single image using the pre-trained ViT model.
        :param image_path: Path to the image file.
        :return: The image embedding as a numpy array.
        """
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Pass the image through the model to get embeddings
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the embedding from the [CLS] token
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return embedding

    def extract_embeddings_from_folder(self, folder_path: str) -> np.ndarray:
        """
        Extract embeddings for all images in a folder.
        :param folder_path: Path to the folder containing image files.
        :return: A matrix of embeddings for all images.
        """
        embeddings = []
        image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
        
        for image_path in image_paths:
            print(f"Processing {image_path}")
            embedding = self.extract_embedding(image_path)
            embeddings.append(embedding)
        
        return np.array(embeddings)

    def cluster_embeddings(self, embeddings: np.ndarray, method: str = "kmeans", **kwargs):
        """
        Apply a clustering algorithm to the extracted embeddings.
        :param embeddings: Matrix of embeddings (numpy array).
        :param method: Clustering method ('kmeans' or 'dbscan').
        :param kwargs: Additional parameters for the clustering algorithm (e.g., n_clusters for K-Means).
        :return: Cluster labels for the embeddings.
        """
        if method == "kmeans":
            n_clusters = kwargs.get("n_clusters", 2)
            print(f"Clustering with K-Means (n_clusters={n_clusters})")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
        
        elif method == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            print(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(embeddings)
        
        else:
            raise ValueError("Invalid clustering method. Choose 'kmeans' or 'dbscan'.")
        
        return labels
  