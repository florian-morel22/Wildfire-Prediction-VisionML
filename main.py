import torch

from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from methods import BasicCNN
from utils import WildfireDataset
from utils import split_val_dataset


def main():

    # variables #
    batchsize = 10
    data_folder = Path("./data/valid")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = BasicCNN(device=device)
    #############


    train, test = split_val_dataset(data_folder)

    transform = v2.Compose([
        v2.ToImage(), # Convert into Image tensor
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = WildfireDataset(train, transform)
    test_dataset = WildfireDataset(test, transform)

    trainloader: DataLoader = train_dataset.__dataloader__(batchsize, num_workers=4)
    testloader: DataLoader = test_dataset.__dataloader__(batchsize, num_workers=4)

    ####### METHOD #######

    method.train(trainloader, nb_epochs=7)
    method.test(testloader)
    method.save()


if __name__ == '__main__':
    main()