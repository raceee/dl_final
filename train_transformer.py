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
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def save_results(history, summary, output_dir="results"):
    """
    Save training history and summary to disk.

    Args:
        history (dict): Training history (losses, scores).
        summary (dict): Summary of model run (dataset size, hyperparameters).
        output_dir (str): Directory to save the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save history as JSON
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Save history as CSV
    history_csv_path = os.path.join(output_dir, "training_history.csv")
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")

def main():
    # Define parameter grids
    dataset_fractions = [0.1]  # Fractions of the dataset to use
    # hidden_layer_options = [6, 12]  # Number of hidden layers
    # hidden_size_options = [348, 768]  # Hidden sizes
    hidden_layer_options = [12]  # Number of hidden layers
    hidden_size_options = [768]  # Hidden sizes
    # Load the full training data
    full_training_df, _, _ = GetTrainingSet(r"C:\Users\Owner\repo\belka_util\train.csv").get_training_data()
    full_training_df = full_training_df.sample(frac=1).reset_index(drop=True)

    # Loop over parameter combinations
    for dataset_fraction in dataset_fractions:
        # Create a smaller dataset based on the fraction
        training_df = full_training_df.sample(frac=dataset_fraction).reset_index(drop=True)
        print(f"Using {len(training_df)} samples from the dataset (fraction: {dataset_fraction})")

        train_size = int(0.8 * len(training_df))
        train_df = training_df.iloc[:train_size]
        val_df = training_df.iloc[train_size:]

        train_smiles_list = train_df['molecule_smiles'].tolist()
        val_smiles_list = val_df['molecule_smiles'].tolist()

        tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

        train_dataset = MoleculeDataset(train_smiles_list[:10], tokenizer, num_rotations=1)
        val_dataset = MoleculeDataset(val_smiles_list, tokenizer, num_rotations=1)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        for num_hidden_layers in hidden_layer_options:
            for hidden_size in hidden_size_options:
                # Generate timestamp
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                print(f"Training with {num_hidden_layers} hidden layers and hidden size {hidden_size} at {timestamp}")

                # Initialize the model with current parameters
                model = MoleculeBERTModel(
                    vocab_size=tokenizer.vocab_size,
                    num_hidden_layers=num_hidden_layers,
                    hidden_size=hidden_size
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
                total_steps = len(train_dataloader) * 10  # Assuming 10 epochs
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(0.1 * total_steps),
                    num_training_steps=total_steps
                )
                loss_fn = nn.CrossEntropyLoss()
                trainer = Trainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    scheduler=scheduler,
                    device=device,
                    criterion=loss_fn
                )

                history = trainer.train(
                    num_epochs=100,
                    smiles_df_path="data/random_unique_smiles.csv",
                    tokenizer=tokenizer,
                    dataset_length=len(training_df)
                )
                plt.close("all")
                # Save training results
                summary = {
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "model": {
                        "num_hidden_layers": num_hidden_layers,
                        "hidden_size": hidden_size
                    },
                    "optimizer": {
                        "learning_rate": 5e-5,
                        "scheduler": "linear"
                    },
                    "num_epochs": 3
                }

                # # Create unique result directory for each combination
                # output_dir = f"results/{timestamp}_dataset_{len(training_df)}_layers_{num_hidden_layers}_hidden_{hidden_size}"
                # save_results(history, summary, output_dir=output_dir)

                # trainer.plot_loss_curves(
                #     history=history,
                #     model_name=f"MoleculeBERT_layers_{num_hidden_layers}_hidden_{hidden_size}",
                #     save_path=os.path.join(output_dir, "plots"),
                #     show_plot=False
                # )

                # trainer.infer_clusters(
                #     "./data/last_unique_smiles.csv",
                #     tokenizer,
                #     method="umap"
                # )

if __name__ == "__main__":
    main()