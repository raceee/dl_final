import torch
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from _transformer.model import MoleculeBERTModel
from _transformer.dataset import MoleculeDataset
from util.training_set import GetTrainingSet
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

# Setup function for DDP
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup function for DDP
def cleanup():
    dist.destroy_process_group()

# Contrastive loss function
def contrastive_loss(hidden_states, labels):
    # Normalize hidden states
    batch_size = hidden_states.size(0)
    hidden_states = hidden_states.view(batch_size, -1)
    hidden_states = hidden_states / torch.norm(hidden_states, dim=1, keepdim=True)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(hidden_states, hidden_states.t())
    labels = labels.unsqueeze(1)

    # Positive and negative masks
    mask = torch.eq(labels, labels.t()).float()
    neg_mask = 1 - mask
    neg_similarity = torch.exp(similarity_matrix * neg_mask).sum(dim=1)
    pos_similarity = torch.exp(similarity_matrix * mask).sum(dim=1)

    # Calculate loss
    loss = -torch.log(pos_similarity / (pos_similarity + neg_similarity)).mean()
    return loss

# Main function
def main(rank, world_size):
    # Setup DDP
    setup(rank, world_size)

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

    # Training loop
    model.train()
    for epoch in range(10):  # Number of epochs
        for batch in train_dataloader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Cleanup DDP
    cleanup()

# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()

    world_size = args.world_size
    rank = args.rank

    main(rank, world_size)
