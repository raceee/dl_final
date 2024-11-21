import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import torch.nn as nn

class MoleculeDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles = self.dataframe.iloc[idx]['molecule_smiles']
        tokenized_smiles = self.tokenizer(smiles, return_tensors='pt', padding='max_length', truncation=True, max_length=128)['input_ids'].squeeze()
        labels = tokenized_smiles.clone()

        # Masking 15% of the tokens for MLM
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix = torch.tensor([0.0 if mask else prob for mask, prob in zip(special_tokens_mask, probability_matrix)])
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        tokenized_smiles[masked_indices] = self.tokenizer.mask_token_id

        return tokenized_smiles, labels

class MoleculeBERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, num_hidden_layers=12, hidden_size=768, intermediate_size=3072):
        super(MoleculeBERTModel, self).__init__()
        # Create custom configuration
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        
        # Initialize model with custom configuration
        self.bert = BertForMaskedLM(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs
    
    def get_cls_embedding(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding
        return cls_embedding