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


# Main function
def main():
    # Load and prepare data
    training_df, _, _ = GetTrainingSet("data/train_sample.csv").get_training_data()
    training_df = training_df.sample(frac=1).reset_index(drop=True)

    train_size = int(0.8 * len(training_df))
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:]

    train_smiles_list = train_df['molecule_smiles'].tolist()
    val_smiles_list = val_df['molecule_smiles'].tolist()

    tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

    # Datasets and DataLoaders
    train_dataset = MoleculeDataset(train_smiles_list, tokenizer)
    val_dataset = MoleculeDataset(val_smiles_list, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = MoleculeBERTModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * 10  # Assuming 10 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    # Initialize and run Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=loss_fn
    )

    # Train the model
    history = trainer.train(num_epochs=10)

    # Plot and save the loss curves
    trainer.plot_loss_curves(
        history=history,
        model_name="MoleculeBERT with AdamW",
        save_path="plots",  # Directory to save the plot
        show_plot=True      # Set to True to display the plot
    )

# Entry point
if __name__ == "__main__":
    main()
