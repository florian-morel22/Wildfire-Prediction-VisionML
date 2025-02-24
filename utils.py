import torch
import pandas as pd

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import to_pil_image

class WildfireDataset(VisionDataset):
    """Create a Dataset with WildFire data
    
    :args:
        ...
        method_name : name of the method that will use this dataset.
        It is only required if the method needs a specific dataset formating
    """
    
    def __init__(
            self,
            dataframe: pd.DataFrame,
            transform: v2.Compose=None,
            target_transform: v2.Compose=None,
            method_name: str=None
        ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.filenames = self.dataframe["file"].copy()
        self.labels = self.dataframe["label"].copy()
        self.transform = transform
        self.target_transform = target_transform
        self.method_name = method_name

    def __getitem__(self, index: int) -> dict:
        img_path: Path = self.filenames[index]
        target: int = self.labels[index]
        
        try:
            X = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Image corrompue ignorée: {img_path} ({e})")
            return None

        if self.transform:
            X = self.transform(X)
        
        if self.target_transform:
            target = self.target_transform(target)

        if self.method_name == "s_vit":
            return {"pixel_values": X, "labels": target}
        elif self.method_name == "ss_selftraining":
            return X, target, index
        else:
            return X, target

    def __len__(self):
        return len(self.filenames)
    
    def __iter__(self):
        for i in range(len(self.filenames)):
            yield self.__getitem__(i)

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

    def __dataloader__(self, batchsize=10, num_workers=4) -> DataLoader:
        return DataLoader(self, batch_size=batchsize, shuffle=True, num_workers=num_workers)
    
    def update_label(self, index: int, value: int):
        assert value in [0, 1], "The new value must be 0 or 1."
        if self.labels[index] in [0, 1]:
            print(f"WARNING : label of {index} is already in [0, 1]. Are you sure you want to update it ?")

        self.labels.loc[index] = value
        self.dataframe.loc[index, "label"] = value

def load_data(data_folder: Path, debug: bool=False, num_samples: int=5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    corrupted_files = [
        data_folder / "test/wildfire/-73.15884,46.38819.jpg", 
        data_folder / "train/nowildfire/-114.152378,51.027198.jpg"
    ]

    train_path = data_folder / "train"
    train_nowildfire_files = [file for file in (train_path / "nowildfire").iterdir() if file.is_file() and file not in corrupted_files]
    train_wildfire_files = [file for file in (train_path / "wildfire").iterdir() if file.is_file()]

    valid_path = data_folder / "valid"
    valid_nowildfire_files = [file for file in (valid_path / "nowildfire").iterdir() if file.is_file()]
    valid_wildfire_files = [file for file in (valid_path / "wildfire").iterdir() if file.is_file()]

    test_path = data_folder / "test"
    test_nowildfire_files = [file for file in (test_path / "nowildfire").iterdir() if file.is_file()]
    test_wildfire_files = [file for file in (test_path / "wildfire").iterdir() if file.is_file() and file not in corrupted_files]



    train_df = pd.DataFrame([
        {
            "file": file,
            "label": -1. #Unknown label
        }
        for file in train_nowildfire_files + train_wildfire_files
    ])

    valid_df = pd.DataFrame([
        {
            "file": file,
            "label": 0.
        }
        for file in valid_nowildfire_files
    ] + [
        {
            "file": file,
            "label": 1.
        }
        for file in valid_wildfire_files
    ])

    test_df = pd.DataFrame([
        {
            "file": file,
            "label": 0.
        }
        for file in test_nowildfire_files
    ] + [
        {
            "file": file,
            "label": 1.
        }
        for file in test_wildfire_files
    ])

    if debug:
        train_df = train_df.iloc[:num_samples]
        valid_df = valid_df.iloc[:num_samples]
        test_df = test_df.iloc[:num_samples]

    return train_df, valid_df, test_df

def compute_metrics(pred):

    labels = pred.label_ids.argmax(-1)
    preds = pred.predictions.argmax(-1)

    print(preds)

    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}