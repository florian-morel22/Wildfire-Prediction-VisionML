import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset 
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTImageProcessor, ViTModel

# Abstract class
class ImageEncoder():

    def encode_images(self):
        pass

class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class ViTEncoder(ImageEncoder):

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
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)  # Base ViT model
        self.model.to(device)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Extract the embedding for a single image using the pre-trained ViT model.
        :param image: image file.
        :return: The image embedding as a numpy array.
        """
        # Open and preprocess the image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Pass the image through the model to get embeddings
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the embedding from the [CLS] token
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return embedding

    def encode_images(self, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for all images in a dataset.
        :param dataset: dataset containing images.
        :return: A matrix of embeddings for all images.
        """

        embeddings = []
        labels = []
        for data in tqdm(dataset, desc="Process images"):
            img, label = data
            embedding = self.encode_image(img)
            embeddings.append(embedding)
            labels.append(label)

        return np.array(embeddings), np.array(labels)
  
  
class ResNetEncoder:
    def __init__(self, model_name="resnet50", device="cpu"):
        """
        Initialize ResNet for unsupervised tasks.
        The model extracts embeddings instead of performing classification.
        """
        self.device = device
        print(f"Using device: {device}")

        # Charger ResNet50 sans la dernière couche (FC layer)
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remplace la dernière couche FC par une identité
        self.model.to(device)
        self.model.eval()

        # Transformations nécessaires pour ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Extract the embedding for a single image using the pre-trained ResNet model.
        :param image: image file.
        :return: The image embedding as a numpy array.
        """
        # Appliquer les transformations
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Extraire les features avec ResNet
        with torch.no_grad():
            embedding = self.model(image)

        return embedding.squeeze(0).cpu().numpy()  # Convertir en numpy

    def encode_images(self, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for all images in a dataset.
        :param dataset: dataset containing images.
        :return: A matrix of embeddings for all images.
        """
        embeddings = []
        labels = []
        for data in tqdm(dataset, desc="Processing images"):
            img, label = data
            embedding = self.encode_image(img)
            embeddings.append(embedding)
            labels.append(label.item())

        return np.array(embeddings), np.array(labels)  