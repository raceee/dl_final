import torch
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from _transformer.model import MoleculeBERTModel
from _transformer.dataset import MoleculeDataset
from util.training_set import GetTrainingSet
import torch.nn as nn
import random

# contrastive loss on the batch
def contrastive_loss(hidden_states, labels):
    # hidden_states: (batch_size, seq_len, hidden_size)
    # labels: (batch_size)
    batch_size = hidden_states.size(0)
    hidden_states = hidden_states.view(batch_size, -1)
    hidden_states = hidden_states / torch.norm(hidden_states, dim=1, keepdim=True)
    
    similarity_matrix = torch.matmul(hidden_states, hidden_states.t())

    labels = labels.unsqueeze(1)

    mask = torch.eq(labels, labels.t()).float()
    neg_mask = 1 - mask
    neg_similarity = similarity_matrix * neg_mask
    neg_similarity = torch.exp(neg_similarity)
    neg_similarity = neg_similarity.sum(dim=1)
    pos_similarity = similarity_matrix * mask
    pos_similarity = torch.exp(pos_similarity)
    pos_similarity = pos_similarity.sum(dim=1)
    loss = -torch.log(pos_similarity / (pos_similarity + neg_similarity)).mean()
    return loss

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
training_df, last_unique_df, random_in_training_df = GetTrainingSet("data/train_sample.csv").get_training_data()

training_df = training_df.sample(frac=1).reset_index(drop=True)

train_size = int(0.8 * len(training_df))
train_df = training_df.iloc[:train_size]
val_df = training_df.iloc[train_size:]

train_smiles_list = train_df['molecule_smiles'].tolist()
val_smiles_list = val_df['molecule_smiles'].tolist()

tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

train_dataset = MoleculeDataset(train_smiles_list, tokenizer)
val_dataset = MoleculeDataset(val_smiles_list, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MoleculeBERTModel(vocab_size=tokenizer.vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * 10  # Assuming 10 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

loss_fn = nn.CrossEntropyLoss()



num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        hidden_sattes = outputs.hidden_states


        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
