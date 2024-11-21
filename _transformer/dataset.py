'''
classes that are needed to transform the dataset into a format that can be used by the transformer 

this will include:
- tokenization of the SMILES strings can be found here: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html
- returning PyTorch DataLoader to load the data into the model

'''

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, mask_probability=0.5):
        # assume smiles_list is always a list of SMILES strings
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):  # handle tensor indices, usually used in DataLoader batches
            idx = idx.item()
        smiles = self.smiles_list[idx]

        tokenized_smiles = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )['input_ids'].squeeze()

        labels = tokenized_smiles.clone()

        probability_matrix = torch.full(labels.shape, self.mask_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix = torch.tensor(
            [0.0 if mask else prob for mask, prob in zip(special_tokens_mask, probability_matrix)]
        )
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # only compute loss on masked tokens

        tokenized_smiles[masked_indices] = self.tokenizer.mask_token_id

        return tokenized_smiles, labels
