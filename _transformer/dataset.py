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

# class MoleculeDataset(Dataset):
#     def __init__(self, smiles_list, tokenizer, mask_probability=0.5):
#         # assume smiles_list is always a list of SMILES strings
#         self.smiles_list = smiles_list
#         self.tokenizer = tokenizer
#         self.mask_probability = mask_probability

#     def __len__(self):
#         return len(self.smiles_list)

#     def __getitem__(self, idx):
#         if isinstance(idx, torch.Tensor):  # Handle tensor indices, usually used in DataLoader batches
#             idx = idx.item()
#         smiles = self.smiles_list[idx]

#         # Tokenize the SMILES string and include attention mask
#         tokenized_output = self.tokenizer(
#             smiles,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             max_length=128
#         )

#         input_ids = tokenized_output['input_ids'].squeeze(0)  # Remove batch dimension
#         attention_mask = tokenized_output['attention_mask'].squeeze(0)  # Remove batch dimension

#         labels = input_ids.clone()

#         # Create the masking probability matrix
#         probability_matrix = torch.full(labels.shape, self.mask_probability)
#         special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
#         probability_matrix = torch.tensor(
#             [0.0 if mask else prob for mask, prob in zip(special_tokens_mask, probability_matrix)]
#         )
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100  # Only compute loss on masked tokens

#         input_ids[masked_indices] = self.tokenizer.mask_token_id

#         return input_ids, attention_mask, labels


from rdkit import Chem
from rdkit.Chem import AllChem
import random
import time

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, mask_probability=0.15, num_rotations=10):
        # Assume smiles_list is always a list of SMILES strings
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.num_rotations = num_rotations

    def __len__(self):
        return len(self.smiles_list)

    def generate_rotations(self, smiles):
        """
        Generate rotations for a SMILES string using RDKit.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]  # Return the original SMILES if invalid

            rotations = set()
            num_atoms = mol.GetNumAtoms()
            for _ in range(self.num_rotations * 2):  # Generate up to num_rotations unique rotations
                atom_indices = list(range(num_atoms))
                random.shuffle(atom_indices)
                randomized_mol = AllChem.RenumberAtoms(mol, atom_indices)
                rotated_smiles = Chem.MolToSmiles(randomized_mol, canonical=False)
                rotations.add(rotated_smiles)
                if len(rotations) >= self.num_rotations:
                    break
            return list(rotations)[:self.num_rotations]
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error generating rotations for SMILES {smiles}: {e}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            return [smiles]  # Return original SMILES if an error occurs

    def mask_tokens(self, input_ids, tokenizer, mask_probability=0.15):
        """
        Mask tokens for MLM without splitting into categories.
        Args:
            input_ids (torch.Tensor): Input token IDs.
            tokenizer: Tokenizer object with special tokens like [MASK].
            mask_probability (float): Probability of masking a token.
        Returns:
            masked_input_ids (torch.Tensor): Token IDs with masks applied.
            labels (torch.Tensor): Original token IDs.
        """
        labels = input_ids.clone()  # Keep original tokens as labels
        probability_matrix = torch.full(labels.shape, mask_probability)
        special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Replace all masked tokens with [MASK]
        input_ids[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        return input_ids, labels


    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # Anchor
        anchor_smiles = self.smiles_list[idx]

        # Generate rotations
        rotations = self.generate_rotations(anchor_smiles)
        positive_smiles = random.choice(rotations)

        # Negative example
        negative_idx = random.randint(0, len(self.smiles_list) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.smiles_list) - 1)
        negative_smiles = self.smiles_list[negative_idx]

        # Tokenize
        anchor = self.tokenizer(anchor_smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        positive = self.tokenizer(positive_smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        negative = self.tokenizer(negative_smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Apply masking to anchor
        masked_input_ids, labels = self.mask_tokens(anchor["input_ids"].squeeze(0), self.tokenizer)
        return {
            "anchor_input_ids": masked_input_ids,
            "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
            "mlm_labels": labels,
            "positive_input_ids": positive["input_ids"].squeeze(0),
            "positive_attention_mask": positive["attention_mask"].squeeze(0),
            "negative_input_ids": negative["input_ids"].squeeze(0),
            "negative_attention_mask": negative["attention_mask"].squeeze(0),
        }