'''
this script is the main script for training the models using torchrun

torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 train.py



'''

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
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup function for DDP
def cleanup():
    dist.destroy_process_group()
