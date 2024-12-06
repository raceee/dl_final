import os
import torch

class Trainer_GNN:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, device, criterion):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.model.to(self.device)


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            # Be like a woman and Assume things
            data = batch.to(self.device)

            # Process masking
            mask = torch.rand(data.x.shape[0]) < 0.15 # 15% because BERT
            labels = data.x[mask].clone()
            data.x[mask] = 0

            # Forward pass
            outputs = self.model(data)

            # Compute loss
            loss = self.criterion(outputs[mask], labels.squeeze().long())

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
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Be like a woman and Assume things
                data = batch.to(self.device)

                # Process masking
                mask = torch.rand(data.x.shape[0]) < 0.15
                labels = data.x[mask].clone()
                data.x[mask] = 0

                # Forward pass
                outputs = self.model(data)

                # Compute loss
                loss = self.criterion(outputs[mask], labels.squeeze().long())

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs[mask], 1)
                correct += (predicted == labels.squeeze().long()).sum().item()
                total += labels.squeeze().size(0)

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_best_model_path=None):
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

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