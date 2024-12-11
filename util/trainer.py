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
from tqdm import tqdm
from torch.nn.functional import triplet_margin_loss
import time
from torch.cuda.amp import autocast, GradScaler
import pickle
import json
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, tokenizer, scheduler=None, device=None, criterion=None):
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
        self.tokenizer = tokenizer

    def set_criterion(self, criterion):
        """Set a custom loss function."""
        self.criterion = criterion

    def train(self, num_epochs, smiles_df_path, tokenizer, dataset_length):
        """
        Train the model for a specified number of epochs, saving plots, weights, and history.

        Args:
            num_epochs (int): Number of epochs to train.
            smiles_df_path (str): Path to the CSV file containing SMILES data for silhouette scoring.
            tokenizer: Tokenizer function or object for SMILES strings.
            dataset_length (int): Length of the training dataset.

        Returns:
            dict: Training and validation losses per epoch.
        """
        # Generate a unique run name based on the model's configuration and dataset length
        run_name = (
            f"MoleculeBERT_layers_{self.model.num_hidden_layers}_hidden_{self.model.hidden_size}_"
            f"intermediate_{self.model.intermediate_size}_dataset_{dataset_length}"
        )
        
        # Create a directory for this run
        base_dir = "results"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Results will be saved in: {run_dir}")

        history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "silhouette_score": []}
        best_val_loss = float("inf")

        for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
            train_loss = sum(self.train_epoch())
            val_loss, val_accuracy, silhouette = self.validate_epoch(smiles_df_path, tokenizer)
            self.calculate_silhouette_score(smiles_df_path, tokenizer, epoch=epoch + 1, save_dir=run_dir, show_plot=False)
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

            # Save the current model
            model_path = os.path.join(run_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)

            # Update and save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(run_dir, f"best_model_epoch_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")

        # Save training history
        history_path = os.path.join(run_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")

        # Plot loss curves
        self.plot_loss_curves(
            history,
            model_name=f"MoleculeBERT_layers_{self.model.num_hidden_layers}_hidden_{self.model.hidden_size}",
            save_path=run_dir
        )

        return history

    # def train_epoch(self):
    #     self.model.train()
    #     total_mlm_loss = 0
    #     total_triplet_loss = 0
    #     progress_bar = tqdm(self.train_dataloader, desc="Training Epoch Progress", unit="batch", leave=False)

    #     for batch_idx, batch in enumerate(progress_bar):
    #         # Unpack batch
    #         masked_input_ids = batch["anchor_input_ids"].to(self.device)
    #         anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
    #         mlm_labels = batch["mlm_labels"].to(self.device)
    #         positive_input_ids = batch["positive_input_ids"].to(self.device)
    #         positive_attention_mask = batch["positive_attention_mask"].to(self.device)
    #         negative_input_ids = batch["negative_input_ids"].to(self.device)
    #         negative_attention_mask = batch["negative_attention_mask"].to(self.device)

    #         # Forward pass for individual inputs
    #         anchor_outputs = self.model(input_ids=masked_input_ids, attention_mask=anchor_attention_mask)
    #         positive_outputs = self.model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
    #         negative_outputs = self.model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)

    #         # Extract logits and CLS embeddings
    #         logits = anchor_outputs["logits"]
    #         anchor_cls = anchor_outputs["hidden_states"][-1][:, 0, :]
    #         positive_cls = positive_outputs["hidden_states"][-1][:, 0, :]
    #         negative_cls = negative_outputs["hidden_states"][-1][:, 0, :]

    #         # Compute MLM loss
    #         mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    #         mlm_loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))

    #         # Compute Triplet Loss
    #         triplet_loss = triplet_margin_loss(anchor_cls, positive_cls, negative_cls, margin=1.0)

    #         # Scale losses for balance
    #         total_loss = 0.7 * mlm_loss + 0.3 * triplet_loss

    #         # Debug cosine similarity
    #         from torch.nn.functional import cosine_similarity
    #         anchor_positive_sim = cosine_similarity(anchor_cls, positive_cls).mean().item()
    #         anchor_negative_sim = cosine_similarity(anchor_cls, negative_cls).mean().item()
    #         print(f"Anchor-Positive Similarity: {anchor_positive_sim}")
    #         print(f"Anchor-Negative Similarity: {anchor_negative_sim}")

    #         # Backward pass
    #         self.optimizer.zero_grad()
    #         total_loss.backward()

    #         # Optimizer step
    #         self.optimizer.step()

    #         # Accumulate losses
    #         total_mlm_loss += mlm_loss.detach().cpu().item()
    #         total_triplet_loss += triplet_loss.detach().cpu().item()

    #         # Update progress bar
    #         progress_bar.set_postfix(
    #             mlm_loss=mlm_loss.item(),
    #             triplet_loss=triplet_loss.item(),
    #             total_loss=total_loss.item()
    #         )

    #     # Average losses
    #     avg_mlm_loss = total_mlm_loss / len(self.train_dataloader)
    #     avg_triplet_loss = total_triplet_loss / len(self.train_dataloader)

    #     return avg_mlm_loss, avg_triplet_loss
    def train_epoch(self):
        self.model.train()
        total_mlm_loss = 0
        total_triplet_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc="Training Epoch Progress", unit="batch", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            masked_input_ids = batch["anchor_input_ids"].to(self.device)
            anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
            mlm_labels = batch["mlm_labels"].to(self.device)
            positive_input_ids = batch["positive_input_ids"].to(self.device)
            positive_attention_mask = batch["positive_attention_mask"].to(self.device)
            negative_input_ids = batch["negative_input_ids"].to(self.device)
            negative_attention_mask = batch["negative_attention_mask"].to(self.device)

            # Forward pass for individual inputs
            anchor_outputs = self.model(input_ids=masked_input_ids, attention_mask=anchor_attention_mask)
            positive_outputs = self.model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
            negative_outputs = self.model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)

            # Extract logits and CLS embeddings
            logits = anchor_outputs["logits"]
            anchor_cls = anchor_outputs["hidden_states"][-1][:, 0, :]
            positive_cls = positive_outputs["hidden_states"][-1][:, 0, :]
            negative_cls = negative_outputs["hidden_states"][-1][:, 0, :]

            # Create a mask to exclude padding tokens (and optionally special tokens)
            padding_mask = mlm_labels != self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, "cls_token_id"):
                padding_mask &= mlm_labels != self.tokenizer.cls_token_id
            if hasattr(self.tokenizer, "sep_token_id"):
                padding_mask &= mlm_labels != self.tokenizer.sep_token_id

            # Mask logits and labels
            valid_logits = logits.view(-1, logits.size(-1))[padding_mask.view(-1)]
            valid_labels = mlm_labels.view(-1)[padding_mask.view(-1)]

            # Compute MLM loss
            mlm_loss_fn = nn.CrossEntropyLoss()
            print("Padding Mask Shape:", padding_mask.shape)
            print("Number of Valid Tokens:", valid_logits.shape[0], valid_labels.shape[0])
            mlm_loss = F.cross_entropy(valid_logits, valid_labels)

            # Compute Triplet Loss
            triplet_loss = triplet_margin_loss(anchor_cls, positive_cls, negative_cls, margin=1.0)

            # Scale losses for balance
            total_loss = 0.7 * mlm_loss + 0.3 * triplet_loss

            # Debug cosine similarity
            from torch.nn.functional import cosine_similarity
            anchor_positive_sim = cosine_similarity(anchor_cls, positive_cls).mean().item()
            anchor_negative_sim = cosine_similarity(anchor_cls, negative_cls).mean().item()
            print(f"Anchor-Positive Similarity: {anchor_positive_sim}")
            print(f"Anchor-Negative Similarity: {anchor_negative_sim}")

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate losses
            total_mlm_loss += mlm_loss.detach().cpu().item()
            total_triplet_loss += triplet_loss.detach().cpu().item()

            # Update progress bar
            progress_bar.set_postfix(
                mlm_loss=mlm_loss.item(),
                triplet_loss=triplet_loss.item(),
                total_loss=total_loss.item()
            )

        # Average losses
        avg_mlm_loss = total_mlm_loss / len(self.train_dataloader)
        avg_triplet_loss = total_triplet_loss / len(self.train_dataloader)

        return avg_mlm_loss #, avg_triplet_loss



    
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
        self.model.eval()
        total_mlm_loss = 0
        total_triplet_loss = 0
        total_correct = 0
        total_elements = 0

        # Calculate silhouette score
        silhouette = self.calculate_silhouette_score(smiles_df_path, tokenizer)

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Unpack batch dictionary
                masked_input_ids = batch["anchor_input_ids"].to(self.device)
                anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
                mlm_labels = batch["mlm_labels"].to(self.device)
                positive_input_ids = batch["positive_input_ids"].to(self.device)
                positive_attention_mask = batch["positive_attention_mask"].to(self.device)
                negative_input_ids = batch["negative_input_ids"].to(self.device)
                negative_attention_mask = batch["negative_attention_mask"].to(self.device)

                # Concatenate inputs
                combined_input_ids = torch.cat([masked_input_ids, positive_input_ids, negative_input_ids], dim=0)
                combined_attention_mask = torch.cat([anchor_attention_mask, positive_attention_mask, negative_attention_mask], dim=0)

                # Forward pass
                outputs = self.model(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
                logits = outputs["logits"]
                hidden_states = outputs["hidden_states"][-1]

                # Compute MLM loss
                mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                mlm_loss = mlm_loss_fn(logits[:masked_input_ids.size(0)].view(-1, logits.size(-1)), mlm_labels.view(-1))

                # Compute Triplet Loss
                batch_size = masked_input_ids.size(0)
                anchor_cls = hidden_states[:batch_size, 0, :]
                positive_cls = hidden_states[batch_size:2 * batch_size, 0, :]
                negative_cls = hidden_states[2 * batch_size:, 0, :]
                triplet_loss = triplet_margin_loss(anchor_cls, positive_cls, negative_cls, margin=1.0)

                # Accumulate losses
                total_mlm_loss += mlm_loss.item()
                total_triplet_loss += triplet_loss.item()

                # Compute accuracy (if logits correspond to masked tokens)
                preds = logits[:masked_input_ids.size(0)].argmax(dim=-1)
                total_correct += (preds == mlm_labels).sum().item()
                total_elements += mlm_labels.ne(-100).sum().item()  # Ignore padding/masked elements

        avg_mlm_loss = total_mlm_loss / len(self.val_dataloader)
        avg_triplet_loss = total_triplet_loss / len(self.val_dataloader)
        val_accuracy = total_correct / total_elements if total_elements > 0 else 0.0

        return avg_mlm_loss + avg_triplet_loss, val_accuracy, float(silhouette)


    def calculate_silhouette_score(self, smiles_df_path, tokenizer, epoch=None, save_dir=None, show_plot=False):
        """
        Calculate and visualize the silhouette score for the rotations of SMILES in last_unique_smiles.csv.

        Args:
            smiles_df_path (str): Path to the CSV file containing last_unique SMILES.
            tokenizer: Tokenizer function or object for SMILES strings.
            epoch (int, optional): Current epoch number for labeling the plot.
            save_dir (str, optional): Directory to save the plot. If None, does not save.
            show_plot (bool, optional): Whether to show the plot interactively.

        Returns:
            float: Silhouette score (higher is better).
        """
        self.model.eval()  # Ensure the model is in evaluation mode

        # Load the SMILES dataset
        df = pd.read_csv(smiles_df_path)
        df['rotated_smiles'] = df['molecule_smiles'].apply(self.generate_rotations)

        all_smiles = []
        labels = []

        for idx, (original, rotations) in enumerate(zip(df['molecule_smiles'], df['rotated_smiles'])):
            all_smiles.append(original)
            labels.append(idx)  # Original SMILES are labeled by their index
            for rotation in rotations:
                all_smiles.append(rotation)
                labels.append(idx)  # Rotations share the same label as the original

        # Tokenize SMILES
        tokenized_smiles = tokenizer(
            all_smiles,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_ids = tokenized_smiles['input_ids'].to(self.device)
        attention_mask = tokenized_smiles['attention_mask'].to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs["hidden_states"][-1]  # Use the last hidden states
            embeddings_cls = hidden_states[:, 0, :]  # Extract CLS token embeddings

        embeddings_np = embeddings_cls.cpu().numpy()  # Convert to numpy for silhouette scoring

        # Calculate silhouette score
        if len(set(labels)) > 1:  # Silhouette score requires at least 2 clusters
            silhouette = silhouette_score(embeddings_np, labels)
        else:
            silhouette = float('nan')  # Silhouette score is undefined for a single cluster

        print(f"Silhouette Score (last_unique_smiles): {silhouette:.4f}")

        # Apply UMAP for visualization
        reducer = UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)

        # Generate the scatter plot
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

        title = f"UMAP Visualization (Epoch {epoch}) - Silhouette: {silhouette:.4f}"
        plt.title(title)
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Molecules")
        plt.tight_layout()

        # Save plot if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = (
                f"scatter_epoch_{epoch}_layers_{self.model.num_hidden_layers}_hidden_{self.model.hidden_size}_"
                f"silhouette_{silhouette:.4f}.png"
            )
            plot_path = os.path.join(save_dir, filename)
            plt.savefig(plot_path)
            print(f"Scatter plot saved to {plot_path}")
        plt.close()

        return silhouette
