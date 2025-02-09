import torch

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from methods import BasicCNN, ViT
from utils import WildfireDataset
from utils import split_val_dataset


def main():
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


if __name__ == '__main__':
    main()