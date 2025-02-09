import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from models import Net

# ViT
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
from utils import compute_metrics
from datasets import DatasetDict
    
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
            train_dataset,
            nb_epochs: int,
            batch_size=50
    ) -> None:
        
        trainloader: DataLoader = train_dataset.__dataloader__(batch_size, num_workers=4)

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
            test_dataset: DataLoader,
            batch_size: int = 50
    ) -> None:
        correct = 0
        total = 0

        print(">>> TEST")
        print("TODO: Voir autres scores")
        testloader: DataLoader = test_dataset.__dataloader__(batch_size, num_workers=4)
        
        with torch.no_grad():
            for data in tqdm(testloader, total=len(testloader)):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.network(images)

                predicted = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                labels = labels.view_as(predicted)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network : {100 * correct / total} %')

    def train_and_test(
            self,
            train_dataset,
            test_dataset,
            nb_epochs: int = 7,
            batch_size: int = 50,
            learning_rate: float = 1e-2
        ) -> None:

        self.train(train_dataset, nb_epochs=nb_epochs, batch_size=batch_size)
        self.test(test_dataset, batch_size=batch_size)
            
    def save(self):
        """save the model, the loss plot..."""
        pass


class ViT():

    def __init__(
            self,
            model_name: str = "google/vit-base-patch16-224-in21k",
            device: str = "cpu",
            ):

        self.device = device
        print(device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels = 2, 
            id2label = {0:'Normal', 1:'Fire'}, 
            label2id = {'Normal':0, 'Fire':1}
            )
        self.model.to(device)


    def train_and_test(
            self,
            train_dataset,
            test_dataset,
            nb_epochs: int,
            batch_size: int = 50,
            learning_rate: float = 1e-2
        ) -> None:
        
        print(">>> TRAIN")

        # default: adamW optimizer
        training_args = TrainingArguments(
            use_mps_device=True,
            output_dir="./vit-fire-detection",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            num_train_epochs=nb_epochs,

            learning_rate=learning_rate,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            warmup_steps=100,

            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            load_best_model_at_end=True
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.feature_extractor
        )

        try:
        # Train the model
            trainer.train()
        except Exception as e:
            print(f"Exception. {e}")