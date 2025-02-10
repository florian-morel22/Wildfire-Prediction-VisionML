import os
import torch
import random
import argparse

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from utils import load_data
from models import ViTEncoder
from utils import WildfireDataset
from torch.utils.data import Subset
from methods import Method, BasicCNN, ViT, BasicClustering


def main(args):
    method_name = args.method
    data_path = Path(args.data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    sessions = []

    if method_name == "basic_cnn" or method_name == "all":
        method = BasicCNN(device=device)
        sessions.append(("basic_cnn", method))

    if method_name == "vit" or method_name == "all":
        method = ViT(device=device, nb_epochs=50, batch_size=50, learning_rate=1e-2)
        sessions.append(("vit", method))

    if method_name == "clustering_vit" or method_name == "all":
        encoder = ViTEncoder(device=device)
        method = BasicClustering(
            encoder=encoder,
            device=device,
            method=args.clustering_algo,
            nb_cluster=args.nb_clusters
        )

        sessions.append(("clustering_vit", method))

    train_df, valid_df, test_df = load_data(data_path, args.DEBUG)
    
    for session in sessions:

        method_name: str = session[0]
        method: Method = session[1]

        print(f"\033[32m\n>>>>>>>>>> {method_name} <<<<<<<<<<\n\033[0m")

        method.process_data(train_df, valid_df, test_df)
        method.run(args.DEBUG)
        method.save()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default=BasicCNN,
        choices=["basic_cnn", "vit", "clustering_vit", "all"],
        help="Method to run"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/valid",
    )
    parser.add_argument(
        "--nb_clusters",
        type=int,
        default=2,
        help="Number of clusters to use if the clustering method is chosen and kmeans is the clustering algo."
    )
    parser.add_argument(
        "--clustering_algo",
        default="kmeans",
        choices=["kmeans", "dbscan"],
        help="Algo to run if the clustering method is chosen."
    )

    args = parser.parse_args()

    main(args)