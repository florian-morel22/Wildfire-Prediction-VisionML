import torch
import argparse

from pathlib import Path

from utils import load_data 
from models import ViTEncoder, ResNetEncoder, SegFormerEncoder, resnet_classifier
from methods import (
    Method,
    SupervisedClassifier,
    SupervisedViT,
    UnsupervisedClustering,
    SemiSupervisedClustering,
    SemiSupervisedSelfTraining
)

torch.manual_seed(42)


def get_methods(method_name: str, args) -> list[tuple[str, Method]]:

    """
    Get the methods to run from a method name.
    If method_name == 'all', all the methods are returned.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    mothods_list = []

    if method_name == "s_classifier_resnet" or method_name == "all":
        classifier = resnet_classifier(num_classes=1)
        method = SupervisedClassifier(
            device=device,
            batch_size=32,
            network=classifier,
            learning_rate=1e-5,
            nb_epochs=10
        )
        mothods_list.append(("s_classifier_resnet", method))

    if method_name == "ss_selftraining_resnet" or method_name == "all":
        classifier = resnet_classifier(num_classes=1)
        method = SemiSupervisedSelfTraining(
            network=classifier,
            device=device,
            batch_size=32,
            confidence_rate=0.9,
            max_pseudo_label=3000,
            learning_rate=1e-3,
            nb_epochs=5,
            steps=5
        )
        mothods_list.append(("ss_selftraining_resnet", method))

    if method_name == "s_vit" or method_name == "all":
        method = SupervisedViT(device=device, nb_epochs=50, batch_size=50, learning_rate=1e-2)
        mothods_list.append(("s_vit", method))

    if method_name == "us_clustering_vit" or method_name == "all":
        encoder = ViTEncoder(device=device)
        method = UnsupervisedClustering(
            encoder=encoder,
            device=device,
            algo=args.clustering_algo,
            n_clusters=args.n_clusters
        )

        mothods_list.append(("us_clustering_vit", method))
        
    if method_name == "us_clustering_resnet_net" or method_name == "all":
        encoder = ResNetEncoder(device=device)
        method = UnsupervisedClustering(
            encoder=encoder,
            device=device,
            method=args.clustering_algo, 
            n_cluster=args.n_clusters
        )

        mothods_list.append(("us_clustering_resnet_net", method))
    
    if method_name == "ss_clustering_vit_resnet" or method_name == "all":
        encoder = ViTEncoder(device=device)
        classifier = resnet_classifier(num_classes=1)
        method = SemiSupervisedClustering(
            encoder=encoder,
            device=device,
            classifier=classifier,
            algo=args.clustering_algo,
            n_clusters=args.n_clusters,
            batch_size=32,
        )
        
        mothods_list.append(("ss_clustering_vit_resnet", method))

    if method_name == "ss_clustering_segformer_net" or method_name == "all":
        encoder = SegFormerEncoder(device=device)
        method = SemiSupervisedClustering(
            encoder=encoder,
            device=device,
            algo=args.clustering_algo,
            n_clusters=args.n_clusters
        )
        
        mothods_list.append(("ss_clustering_segformer_net", method))

    return mothods_list

def main(args):
    
    method_name = args.method
    data_path = Path(args.data_path)
    num_samples = int(args.num_samples)

    train_df, valid_df, test_df = load_data(data_path, args.DEBUG, num_samples=num_samples)

    for method_name, method in get_methods(method_name, args):

        method_name: str
        method: Method

        print(f"\033[32m\n>>>>>>>>>> {method_name} <<<<<<<<<<\n\033[0m")

        method.process_data(train_df, valid_df, test_df)
        method.run(args.DEBUG)
        method.save(method_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default=SupervisedClassifier,
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