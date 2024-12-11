import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit.Chem import CanonicalRankAtoms
from umap import UMAP

from rdkit.Chem import rdmolops
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import silhouette_score

from util.gnn_utils import GNN_Utils


class Trainer_GNN:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, device, criterion):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        # self.class_weights = class_weights , class_weights
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
            data.x[mask] = -100 # was zero before

            # # Process masking with class weights
            # mask_probs = torch.ones(data.x.shape[0]).to(self.device)
            # for atomic_num, weight in enumerate(self.class_weights):
            #     mask_probs[data.x.flatten() == atomic_num] = weight

            # # Create mask based on probabilities
            # mask = torch.bernoulli(mask_probs).bool()

            # # Ensure at least one node is masked
            # if mask.sum() == 0:
            #     random_idx = torch.randint(0, data.x.shape[0], (1,))
            #     mask[random_idx] = True

            # # Masked labels and features
            # labels = data.x[mask].clone()
            # data.x[mask] = -100


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
    
    def validate_epoch(self, smiles_df_path, num_randos):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Load the SMILES dataset for silhousette scoring
        df = pd.read_csv(smiles_df_path)
        df = df.iloc[:num_randos]
        df['rotated_smiles'] = df['molecule_smiles'].apply(self.generate_rotations)

        all_smiles = []
        labels = []

        for idx, (original, rotations) in enumerate(zip(df['molecule_smiles'], df['rotated_smiles'])):
            all_smiles.append(original)
            labels.append(idx)
            for rotation in rotations:
                all_smiles.append(rotation)
                labels.append(idx)

        # Convert to graphs
        utils = GNN_Utils(self.device)

        with torch.no_grad():
            embeddings_list = []
            for smi in all_smiles:
                data = utils.process(smi)

                embeddings = self.model(data, return_embeddings=True)
                embeddings_list.append(embeddings.cpu().numpy())

            embeddings_np = np.vstack(embeddings_list)
            silhouette = silhouette_score(embeddings_np, labels) # if len(set(labels)) > 1 else float('nan')

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Be like a woman and Assume things
                data = batch.to(self.device)

                # Process masking
                mask = torch.rand(data.x.shape[0]) < 0.15
                labels = data.x[mask].clone()
                data.x[mask] = -100 # was zero before

                # # Process masking with class weights
                # mask_probs = torch.ones(data.x.shape[0]).to(self.device)
                # for atomic_num, weight in enumerate(self.class_weights):
                #     mask_probs[data.x.flatten() == atomic_num] = weight

                # # Create mask based on probabilities
                # mask = torch.bernoulli(mask_probs).bool()

                # # Ensure at least one node is masked
                # if mask.sum() == 0:
                #     random_idx = torch.randint(0, data.x.shape[0], (1,))
                #     mask[random_idx] = True

                # # Masked labels and features
                # labels = data.x[mask].clone()
                # data.x[mask] = -100

                # Forward pass
                outputs = self.model(data)

                # Compute loss
                loss = self.criterion(outputs[mask], labels.squeeze().long())
                total_loss += loss.item()

                # Metrics
                _, predicted = torch.max(outputs[mask], 1)
                correct += (predicted == labels.squeeze().long()).sum().item()
                total += labels.squeeze().size(0)

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total
        return avg_loss, accuracy, float(silhouette)
    
    def train(self, num_epochs, smiles_df_path, num_randos, save_best_model_path=None):
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'silhouette_score': []}
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy, silhouette = self.validate_epoch(smiles_df_path, num_randos)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['silhouette_score'].append(silhouette)

            print(
                f"\n\n\nEpoch {epoch + 1}/{num_epochs}: \n"
                f"Train Loss = {train_loss:.4f}, \n"
                f"Val Loss = {val_loss:.4f}, \n"
                f"Val Accuracy = {val_accuracy:.4f}, \n"
                f"Silhouette Score = {silhouette:.4f}"
            )
            
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_model_state = self.model.state_dict()
            #     print(f"New best model found at epoch {epoch + 1} with validation loss {val_loss:.4f}")

            #     # Save the best model if a path is provided
            #     path_name = f"checkpoints/{self.model.__class__.__name__}_num_hidden_layers_{self.model.num_hidden_layers}_epoch_{epoch + 1}_loss_{best_val_loss}.pth"
            #     os.makedirs(os.path.dirname(path_name), exist_ok=True)
            #     torch.save(best_model_state, path_name)
            #     print(f"Best model saved to {path_name}")

        return history
    
    def plot_loss_curves(self, history, model_name="Model", save_path=None, show_plot=False):
        train_loss = history['train_loss']
        val_loss = history['val_loss']

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

    def generate_rotations(self, smiles, num_rotations=10): # Change if you want
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string")
            
            rotations = set()
            num_atoms = mol.GetNumAtoms()

            for _ in range(num_rotations * 2):
                atom_indices = list(range(num_atoms))
                random.shuffle(atom_indices)

                randomized_mol = AllChem.RenumberAtoms(mol, atom_indices)

                rotated_smiles = Chem.MolToSmiles(randomized_mol, canonical=False)

                rotations.add(rotated_smiles)

                if len(rotations) == num_rotations:
                    break

            return list(rotations)[:num_rotations]
        
        except Exception as e:
            print(f"Error generating rotations: {e}")
            return []

    def infer_clusters(self, path, num_randos, method, show_plot=False):

        df = pd.read_csv(path)
        df = df.iloc[:num_randos]
        print(len(df))
        df['rotated_smiles'] = df['molecule_smiles'].apply(self.generate_rotations)
        
        all_smiles = []
        labels = []

        for idx, (original, rotations) in enumerate(zip(df['molecule_smiles'], df['rotated_smiles'])):
            all_smiles.append(original)
            labels.append(idx)
            for rotation in rotations:
                all_smiles.append(rotation)
                labels.append(idx)

        # Convert to graphs
        utils = GNN_Utils(self.device)

        embeddings_list = []
        with torch.no_grad():
            for smi in all_smiles:
                data = utils.process(smi)

                embeddings = self.model(data, return_embeddings=True)
                embeddings_list.append(embeddings.cpu().numpy())

            embeddings_np = np.vstack(embeddings_list)

        reducer = UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)

        silhouette = silhouette_score(embeddings_np, labels)
        print(f"Silhouette Score: {silhouette}")

        if show_plot:
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

        return float(silhouette)
