import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler=None, device=None, criterion=None):
        """
        Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The model to train.
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader): DataLoader for the validation set.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            scheduler (optional): Learning rate scheduler.
            device (torch.device): The device to run the training on.
            criterion (callable, optional): The loss function.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion or nn.CrossEntropyLoss()  # Default to CrossEntropyLoss
        self.model.to(self.device)

    def set_criterion(self, criterion):
        """Set a custom loss function."""
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            # Assume batch contains (input_ids, labels
            input_ids, labels = batch
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(input_ids=input_ids)

            # Extract logits
            logits = outputs.logits if isinstance(outputs, dict) else outputs

            # Compute loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            self.optimizer.zero_grad()
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
                # Assume batch contains (input_ids, labels)
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids)

                # Extract logits
                logits = outputs.logits if isinstance(outputs, dict) else outputs

                # Compute loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # Metrics
                total_loss += loss.item()
                _, predicted = logits.max(dim=-1)
                correct += (predicted == labels).sum().item()
                total += labels.numel()

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, num_epochs, save_best_model_path=None):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.

        Returns:
            dict: Training and validation losses per epoch.
        """
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        best_model_state = None

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                print(f"New best model found at epoch {epoch + 1} with validation loss {val_loss:.4f}")

                # Save the best model if a path is provided
                path_name = f"checkpoints/{self.model.__class__.__name__}_num_hidden_layers_{self.model.num_hidden_layers}_epoch_{epoch + 1}_loss_{best_val_loss}.pth"
                os.makedirs(os.path.dirname(path_name), exist_ok=True)
                torch.save(best_model_state, path_name)
                print(f"Best model saved to {path_name}")

        return history
    
    def plot_loss_curves(self, history, model_name="Model", save_path=None, show_plot=False):
        """
        Plot and save the training and validation loss curves.

        Args:
            history (dict): Dictionary containing 'train_loss' and 'val_loss'.
            model_name (str): Descriptive name of the model to include in the title.
            save_path (str, optional): Directory to save the plot. If None, does not save.
            show_plot (bool, optional): Whether to display the plot interactively. Default is False.
        """
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        if not train_loss or not val_loss:
            raise ValueError("History does not contain 'train_loss' or 'val_loss'.")

        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Training Loss", marker="o")
        plt.plot(val_loss, label="Validation Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {model_name}")
        plt.legend()
        plt.grid()

        # Save the plot if a path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, f"{model_name.replace(' ', '_')}_loss_curves.png")
            plt.savefig(plot_path)
            print(f"Loss curves saved to {plot_path}")

        # Show the plot if required
        if show_plot:
            plt.show()

        # Close the plot to free memory
        plt.close()