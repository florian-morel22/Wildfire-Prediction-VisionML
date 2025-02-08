import pandas as pd

from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset
from matplotlib import pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class WildfireDataset(VisionDataset):
    
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.filenames = dataframe["file"].reset_index(drop=True)
        self.labels = dataframe["label"].reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, index: int):
        img_path = self.filenames[index]
        try:
            X = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        target = self.labels[index]

        if self.transform:
            X = self.transform(X)

        return {"pixel_values": X, "labels": target}
    
    def __showitem__(self, index: int, mean=None, std=None):

        dict_ = self.__getitem__(index)
        image_tensor, label = dict_['pixel_values'], dict_['labels']

        # If normalization parameters are provided, create a denormalization transform
        if mean and std:
            denormalize = v2.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1 / s for s in std]
            )

        # Denormalize if needed
        if mean and std:
            image_tensor = denormalize(image_tensor)

        # Convert the tensor back to a PIL image
        if isinstance(image_tensor, torch.Tensor):
            image = to_pil_image(image_tensor)
        else:
            image = image_tensor

        # Display the image
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
    
    def __showitems__(self, indices: list, mean=None, std=None):
        plt.figure(figsize=(12, 12))

        # If normalization parameters are provided, create a denormalization transform
        if mean and std:
            denormalize = v2.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1 / s for s in std]
            )

        for i, index in enumerate(indices, start=1):
            dict_ = self.__getitem__(index)
            image_tensor, label = dict_['pixel_values'], dict_['labels']

            # Denormalize if needed
            if mean and std:
                image_tensor = denormalize(image_tensor)

            # Convert the tensor back to a PIL image
            if isinstance(image_tensor, torch.Tensor):
                image = to_pil_image(image_tensor)
            else:
                image = image_tensor

            # Plot the image
            plt.subplot(1, len(indices), i)
            plt.imshow(image)
            plt.title(f"Label: {label}")
            plt.axis("off")
        plt.show()

        
    def __len__(self):
        return len(self.filenames)
    
    def __dataloader__(self, batchsize=10, num_workers=4) -> DataLoader:
        return DataLoader(self, batch_size=batchsize, shuffle=True, num_workers=num_workers)



def split_val_dataset(data_folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:

    nowildfire_files = [file for file in (data_folder / "nowildfire").iterdir() if file.is_file()]
    wildfire_files = [file for file in (data_folder / "wildfire").iterdir() if file.is_file()]

    fulldataset = pd.DataFrame([
        {
            "file": file,
            "label": 0.
        }
        for file in nowildfire_files
    ] + [
        {
            "file": file,
            "label": 1.
        }
        for file in wildfire_files
    ])

    train, valid = train_test_split(fulldataset, train_size=0.8, random_state=42, shuffle=True)

    return train, valid