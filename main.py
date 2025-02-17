import torch
import argparse

from pathlib import Path

from utils import load_data 
from models import ViTEncoder, ResNetEncoder, SegFormerEncoder, resnet_classifier
from methods import Method, BasicCNN, ViT, BasicClustering, AdvancedClustering, SemiSupervisedCNN

torch.manual_seed(42)

def main(args):
    method_name = args.method
    data_path = Path(args.data_path)
    num_samples = int(args.num_samples)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    sessions = []

    if method_name == "basic_cnn" or method_name == "all":
        classifier = resnet_classifier(num_classes=1)
        method = BasicCNN(
            device=device,
            batch_size=32,
            network=classifier,
            learning_rate=1e-5,
            nb_epochs=10
        )
        sessions.append(("basic_cnn", method))

    if method_name == "semisupervised_cnn" or method_name == "all":
        # classifier = resnet_classifier(num_classes=1)
        method = SemiSupervisedCNN(
            network=None,
            device=device,
            batch_size=32,
            confidence_rate=0.9,
            max_pseudo_label=3000,
            learning_rate=1e-3,
            nb_epochs=5,
            steps=10
        )
        sessions.append(("semisupervised_cnn", method))

    if method_name == "vit" or method_name == "all":
        method = ViT(device=device, nb_epochs=50, batch_size=50, learning_rate=1e-2)
        sessions.append(("vit", method))

    if method_name == "clustering_vit" or method_name == "all":
        encoder = ViTEncoder(device=device)
        method = BasicClustering(
            encoder=encoder,
            device=device,
            algo=args.clustering_algo,
            n_clusters=args.n_clusters
        )

        sessions.append(("clustering_vit", method))
        
    if method_name == "clustering_resnet" or method_name == "all":
        encoder = ResNetEncoder(device=device)
        method = BasicClustering(
            encoder=encoder,
            device=device,
            method=args.clustering_algo, 
            n_cluster=args.n_clusters
        )

        sessions.append(("clustering_resnet", method))
    
    if method_name == "advanced_clustering" or method_name == "all":
        encoder = ViTEncoder(device=device)
        classifier = resnet_classifier(num_classes=1)
        method = AdvancedClustering(
            encoder=encoder,
            device=device,
            classifier=classifier,
            algo=args.clustering_algo,
            n_clusters=args.n_clusters,
            batch_size=32,
        )
        
        sessions.append(("advanced_clustering", method))

    if method_name == "advanced_clustering1" or method_name == "all":
        encoder = SegFormerEncoder(device=device)
        method = AdvancedClustering(
            encoder=encoder,
            device=device,
            algo=args.clustering_algo,
            n_clusters=args.n_clusters
        )
        
        sessions.append(("advanced_clustering1", method))

    train_df, valid_df, test_df = load_data(data_path, args.DEBUG, num_samples=num_samples)
    
    
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
        help="Method to run"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/valid",
    )
    parser.add_argument(
        "--n_clusters",
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

    parser.add_argument(
        "--num_samples",
        default=5,
        help="Number of images samples for DEBUG flag."
    )

    args = parser.parse_args()

    main(args)