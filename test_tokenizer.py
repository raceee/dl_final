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
# Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# SMILES string and tokenizer
negative_smiles = "C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)N[Dy])n2)cc1"
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

# Tokenize the SMILES string
negative = tokenizer(negative_smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# Extract input IDs
input_ids = negative["input_ids"][0]  # Get the 1D tensor (first sequence in the batch)

# Convert input IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Print the tokens
print(tokens)
