from pathlib import Path
from models import ViTEncoder
from utils import WildfireDataset, load_data

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def main():

    data_path = Path("./data")
    out_path = Path("./assets")

    encoder = ViTEncoder(device="cuda")

    train, val, test = load_data(Path("./data"))

    dataset = WildfireDataset(val)

    encoded_images, labels = encoder.encode_images(dataset)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(encoded_images, labels)

    x = reduced_data[:, 0]
    y = reduced_data[:, 1]
    c = labels

    plt.figure()
    plt.scatter(x, y, c=c, linewidths=0.5, marker="+")
    plt.title(f"Explained variance : {pca.explained_variance_ratio_}")
    plt.savefig(out_path / "pca.jpg")


if __name__ == '__main__':

    main()

