# Importing the libraries
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import lightning as L
import torchmetrics


class TransformerLightning(L.LightningModule):
    def __init__(self, model:nn.Module, learning_rate:float, vocab_size:int):
        super().__init__()
        
        # Choosing the hyperparameters
        self.learning_rate = learning_rate
        self.model = model

        # Saving the hyperparameters
        self.save_hyperparameters(ignore=["model"])

        # Torchmetrics logging
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch:tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the input and labels
        features, true_labels = batch

        # Logits from the model
        logits = self(features)

        # Reshapping the label and logits
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        true_labels = true_labels.view(B * T)

        # Calculating the loss
        loss = F.cross_entropy(logits, true_labels)

        # Getting the label
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:torch.Tensor) -> torch.Tensor:
        # Get the loss, true labels and predicted labels
        loss, true_labels, predicted_labels = self._shared_step(batch)

        # Logging the loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch:tuple[torch.Tensor, torch.Tensor], batch_idx:torch.Tensor) -> torch.Tensor:
        # Get the loss, true labels and predicted labels
        loss, true_labels, predicted_labels = self._shared_step(batch)

        # Logging the loss and accuracy
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.valid_acc(predicted_labels, true_labels)
        self.log("val_acc", self.valid_acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Configuring the optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class MakeDataset(Dataset):
    def __init__(self, data:torch.Tensor, block_size:int=10):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index:int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index : index + self.block_size], self.data[index + 1 : index + self.block_size + 1]
    

class MakeDM(L.LightningDataModule):
    def __init__(self, data_dir:str='./', batch_size:int=64, block_size:int=10, num_workers:int=0) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Loading in the data
        with open(self.data_dir, mode='r', encoding='utf-8') as f:
            self.data_ = f.read()

        # Creating mapping from characters to integers and vice versa
        self.stoi_ = {ch:i for i,ch in enumerate(sorted(list(set(self.data_))))}
        self.itos_ = {i:ch for ch,i in self.stoi_.items()}

    def encode(self, text:str) -> list[int]:
        return [self.stoi_[ch] for ch in text]
    
    def decode(self, encoded:list[int]) -> str:
        return [self.itos_[i] for i in encoded]

    def setup(self, stage:str, seed:bool=True, train_size:float=0.9) -> None:
        # Convert the text into a integer representation
        self.encoded_ = torch.tensor(self.encode(self.data_))
        
        # Convert the data into a dataset
        self.dataset_ = MakeDataset(self.encoded_, block_size=self.block_size)

        # Creating a random generator
        generator = torch.Generator().manual_seed(0) if seed else None
        
        # Split the data into training and validation sets
        self.train_data_, self.valid_data_ = random_split(self.dataset_, lengths=[train_size, 1 - train_size], generator=generator)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data_, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data_, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_maps(self) -> tuple[dict, dict]:
        return self.stoi_, self.itos_