import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from models import Net

    
class BasicCNN():

    def __init__(
            self,
            network: nn.Module = None,
            device: str = "cpu",
            ):

        self.device = device

        if network is None:
            self.network = Net(num_classes=1)
            self.network = self.network.to(self.device)

        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(
            self,
            trainloader: DataLoader,
            nb_epochs: int,
    ) -> None:
        
        print(">>> TRAIN")
        
        for epoch in range(nb_epochs):

            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"epoch {epoch}")
            for i, data in pbar:
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss/(i+1)}, refresh=True)


    def test(
            self,
            testloader: DataLoader
    ) -> None:
        correct = 0
        total = 0

        print(">>> TEST")
        print("TODO: Voir autres scores")
        
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.network(images)

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                labels = labels.view_as(predicted)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network : {100 * correct / total} %')

    def save(self):
        """save the model, the loss plot..."""
        pass