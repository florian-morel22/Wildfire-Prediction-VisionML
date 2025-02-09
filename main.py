import torch

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from methods import BasicCNN, ViT, ViTUnsupervised
from utils import WildfireDataset
from utils import split_val_dataset
import os

def main_supervised():
    method_name = "vit"

    # variables #
    batchsize = 10
    data_folder = Path("./data/valid")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    if method_name == "basic_cnn":
        method = BasicCNN(device=device)

        transform = v2.Compose([
            v2.ToImage(), # Convert into Image tensor
            v2.ToDtype(torch.float32, scale=True)
        ])

    elif method_name == "vit":
        method = ViT(device=device)

        feature_extractor_ = method.feature_extractor
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # Scale pixel values to [0, 1]
            v2.Resize((224, 224)),  # Resize images to 224x224
            v2.Normalize(
                mean=torch.tensor(feature_extractor_.image_mean, dtype=torch.float32).tolist(),
                std=torch.tensor(feature_extractor_.image_std, dtype=torch.float32).tolist()
            ),  # Normalize with float32
        ])
    #############


    train, test = split_val_dataset(data_folder)

    train_dataset = WildfireDataset(train, transform)
    test_dataset = WildfireDataset(test, transform)

    ####### METHOD #######
    import random
    from torch.utils.data import Subset
    fraction = 0.02
    train_indices = random.sample(range(len(train_dataset)), int(len(train_dataset)*fraction))
    test_indices = random.sample(range(len(test_dataset)), int(len(test_dataset)*fraction))
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    method.train_and_test(train_subset, test_subset, nb_epochs=50, batch_size=50, learning_rate=1e-2)
    method.save()

def main_unsupervised():
    method_name = "vit"

    # variables #
    data_folder = Path("./data/valid")
    nb_cluster = 2
    clustering_method = "kmeans" #dbscan #kmeans
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    if method_name == "basic_cnn":
        method = BasicCNN(device=device)

        transform = v2.Compose([
            v2.ToImage(), # Convert into Image tensor
            v2.ToDtype(torch.float32, scale=True)
        ])

    elif method_name == "vit":
        method = ViTUnsupervised(device=device)

        # Get embeddings of images
        embeddings = method.extract_embeddings_from_folder(folder_path=data_folder)

        # Clustering
        labels_kmeans = method.cluster_embeddings(embeddings=embeddings, method="kmeans", nb_cluster=nb_cluster)
        print(f"labels kmean: {labels_kmeans}")

        image_paths = []
        true_labels = []
        for subfolder in ['nowildfire', 'wildfire']:
            subfolder_path = os.path.join(data_folder, subfolder)
            
            for img in os.listdir(subfolder_path):
                if img.endswith(('png', 'jpg', 'jpeg')):
                    image_path = os.path.join(subfolder, img)
                    image_paths.append(image_path)
                    true_labels.append(1 if subfolder == "wildfire" else 0)

        # Map each image to its corresponding cluster label
        correct_predictions = 0
        for img_path, true_label, predicted_label in zip(image_paths, true_labels, labels_kmeans):
            if true_label == predicted_label:
                correct_predictions += 1
        # for img_path, label in zip(image_paths, labels_kmeans):
        #     print(f"Image: {img_path} -> Cluster: {label}")
        # Compute accuracy
        accuracy = correct_predictions / len(image_paths)
        print(f"Accuracy: {accuracy * 100:.2f}%")


def main():
    supervised = False

    if supervised:
        main_supervised()
    else:
        main_unsupervised()


if __name__ == '__main__':
    main()