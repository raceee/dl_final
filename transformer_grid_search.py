import itertools
import json
import shutil
import torch
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from _transformer.model import MoleculeBERTModel
from _transformer.dataset import MoleculeDataset
from util.training_set import GetTrainingSet
from util.trainer import Trainer  # Import the Trainer class
import torch.nn as nn
import random
import os
import torch.distributed as dist

# Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def create_dataset(original_csv_path, train_size, output_dir):
    """Create a new dataset with the specified size."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete existing data directory
    
    os.makedirs(output_dir, exist_ok=True)
    
    training_df, _, _ = GetTrainingSet(original_csv_path).get_training_data()
    training_df = training_df.sample(frac=1).reset_index(drop=True)
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:]
    
    train_smiles_list = train_df['molecule_smiles'].tolist()
    val_smiles_list = val_df['molecule_smiles'].tolist()
    
    # Save the datasets
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    
    return train_smiles_list, val_smiles_list

def save_results(results, output_path):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def run_grid_search(original_csv_path, model_dir, plot_dir, result_dir, tokenizer, params):
    """Perform grid search."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    results = []
    
    for config in itertools.product(
        params['num_hidden_layers'], params['hidden_size'], params['train_sizes']
    ):
        num_hidden_layers, hidden_size, train_size = config
        print(f"Training with layers={num_hidden_layers}, hidden_size={hidden_size}, train_size={train_size}")
        
        # Create datasets
        train_smiles_list, val_smiles_list = create_dataset(
            original_csv_path, train_size, output_dir="./data"
        )
        
        # Create datasets and dataloaders
        train_dataset = MoleculeDataset(train_smiles_list, tokenizer, num_rotations=1)
        val_dataset = MoleculeDataset(val_smiles_list, tokenizer, num_rotations=1)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize model
        model = MoleculeBERTModel(
            vocab_size=tokenizer.vocab_size,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        total_steps = len(train_dataloader) * 10  # Assuming 10 epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            criterion=nn.CrossEntropyLoss()
        )
        
        # Train model
        history = trainer.train(
            num_epochs=10,
            smiles_df_path="./data/training_dataset.csv",
            tokenizer=tokenizer,
            save_best_model_path=f"{model_dir}/model_layers{num_hidden_layers}_hidden{hidden_size}_size{train_size}.pth"
        )
        
        # Save results
        results.append({
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "train_size": train_size,
            "train_loss": history['train_loss'],
            "val_loss": history['val_loss'],
            "silhouette_score": history['silhouette_score']
        })
        
        trainer.plot_loss_curves(
            history=history,
            model_name=f"Layers{num_hidden_layers}_Hidden{hidden_size}_Size{train_size}",
            save_path=plot_dir,
            show_plot=False
        )
    
    # Save all results to a JSON file
    save_results(results, os.path.join(result_dir, "grid_search_results.json"))

# Parameters for grid search
params = {
    "num_hidden_layers": [6, 12],  # Vary the number of layers
    "hidden_size": [348, 768, 1152],  # Vary the hidden size
    "train_sizes": [1000, 10000, 100000]  # Vary dataset sizes
}

# Run the grid search
run_grid_search(
    original_csv_path=r"C:\Users\Owner\repo\belka_util\train.csv",
    model_dir="models",
    plot_dir="plots",
    result_dir="results",
    tokenizer=AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR'),
    params=params
)
