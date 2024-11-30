import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler=None, device=None):
        """
        Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader): DataLoader for the validation set.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            scheduler (optional): Learning rate scheduler.
            device (torch.device): The device to run the training on.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()  # Default loss function, can be replaced dynamically
        self.model.to(self.device)

    def set_criterion(self, criterion):
        """Set a custom loss function."""
        self.criterion = criterion

    def train_epoch(self):
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            # Move batch to device
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def validate_epoch(self):
        """Run one epoch of validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.

        Returns:
            dict: Training and validation losses per epoch.
        """
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val Accuracy = {val_accuracy:.4f}"
            )
        return history
