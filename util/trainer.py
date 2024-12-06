import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from rdkit.Chem import CanonicalRankAtoms
from umap import UMAP
import random
from rdkit.Chem import rdmolops
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import silhouette_score

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
        self.criterion = criterion or nn.CrossEntropyLoss()  # default to CrossEntropyLoss
        self.model.to(self.device)

    def set_criterion(self, criterion):
        """Set a custom loss function."""
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)

            outputs = self.model(input_ids=input_ids)

            logits = outputs.logits if isinstance(outputs, dict) else outputs

            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def train(self, num_epochs, smiles_df_path, tokenizer, save_best_model_path=None):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            smiles_df_path (str): Path to the CSV file containing SMILES data for silhouette scoring.
            tokenizer: Tokenizer function or object for SMILES strings.
            save_best_model_path (str, optional): Path to save the best model.

        Returns:
            dict: Training and validation losses per epoch.
        """
        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "silhouette_score": []}
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy, silhouette = self.validate_epoch(smiles_df_path, tokenizer)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["silhouette_score"].append(silhouette)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val Accuracy = {val_accuracy:.4f}, "
                f"Silhouette Score = {silhouette:.4f}"
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

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Training Loss", marker="o")
        plt.plot(val_loss, label="Validation Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {model_name}")
        plt.legend()
        plt.grid()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, f"{model_name.replace(' ', '_')}_loss_curves.png")
            plt.savefig(plot_path)
            print(f"Loss curves saved to {plot_path}")

        if show_plot:
            plt.show()

        plt.close()

    def infer_clusters(self, path, tokenizer, method="umap"):
        """
        Ingest the path of the dataframe, tokenize the SMILES and their rotations,
        compute embeddings using the model, and visualize clusters.

        Args:
            path (str): Path to the dataframe
            tokenizer: Tokenizer function or object for tokenizing SMILES strings
            method (str): Dimensionality reduction method ('umap' or 'tsne')

        Returns:
            float: Silhouette score (higher is better).
        """

        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        df = pd.read_csv(path)

        df['rotated_smiles'] = df['molecule_smiles'].apply(self.generate_rotations)

        all_smiles = []
        labels = []  

        for idx, (original, rotations) in enumerate(zip(df['molecule_smiles'], df['rotated_smiles'])):
            all_smiles.append(original)
            labels.append(idx)
            for rotation in rotations:
                all_smiles.append(rotation)
                labels.append(idx)

        # Tokenize
        tokenized_smiles = tokenizer(
            all_smiles,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )['input_ids'].to(device)

        self.model.to(device)

        with torch.no_grad():
            outputs = self.model(tokenized_smiles)
            embeddings = outputs.logits

        embeddings_cls = embeddings[:, 0, :]  # Select the embedding at position 0 since it is the [CLS] token

        embeddings_np = embeddings_cls.cpu().numpy()

        # Apply dimensionality reduction
        if method.lower() == "umap":
            reducer = UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("Invalid method. Choose 'umap' or 'tsne'.")

        reduced_embeddings = reducer.fit_transform(embeddings_np)

        silhouette = silhouette_score(embeddings_np, labels)
        print(f"Silhouette Score: {silhouette:.4f}")

        plt.figure(figsize=(10, 8))
        unique_labels = list(set(labels))
        for label in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=f"Molecule {label + 1}",
                alpha=0.7,
                s=50
            )

        plt.title(f"Cluster Visualization ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Molecules")
        plt.tight_layout()
        plt.show()

        return silhouette

    def generate_rotations(self, smiles: str, num_rotations=10) -> list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string.")
            
            rotations = set()
            num_atoms = mol.GetNumAtoms()

            for _ in range(num_rotations * 2):
                atom_indices = list(range(num_atoms))
                random.shuffle(atom_indices)
                
                randomized_mol = AllChem.RenumberAtoms(mol, atom_indices)
                
                rotated_smiles = Chem.MolToSmiles(randomized_mol, canonical=False)
                
                rotations.add(rotated_smiles)
                
                if len(rotations) >= num_rotations:
                    break
            
            return list(rotations)[:num_rotations]

        except Exception as e:
            print(f"Error generating rotations: {e}")
            return []

    def validate_epoch(self, smiles_df_path, tokenizer):
        """Run one epoch of validation and compute silhouette score."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        embeddings_list = []
        labels_list = []

        device = self.device

        # Load the SMILES dataset for silhouette scoring
        df = pd.read_csv(smiles_df_path)
        df['rotated_smiles'] = df['molecule_smiles'].apply(self.generate_rotations)

        all_smiles = []
        labels = []

        for idx, (original, rotations) in enumerate(zip(df['molecule_smiles'], df['rotated_smiles'])):
            all_smiles.append(original)
            labels.append(idx)
            for rotation in rotations:
                all_smiles.append(rotation)
                labels.append(idx)

        # Tokenize
        tokenized_smiles = tokenizer(
            all_smiles,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )['input_ids'].to(device)

        with torch.no_grad():
            # Generate embeddings
            outputs = self.model(tokenized_smiles)
            embeddings = outputs.logits
            embeddings_cls = embeddings[:, 0, :]  # Select [CLS] token

            embeddings_np = embeddings_cls.cpu().numpy()

            # Calculate silhouette score
            silhouette = silhouette_score(embeddings_np, labels) if len(set(labels)) > 1 else float('nan')

        # Validation metrics
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits if isinstance(outputs, dict) else outputs

                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

                _, predicted = logits.max(dim=-1)
                correct += (predicted == labels).sum().item()
                total += labels.numel()

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        return avg_loss, accuracy, silhouette