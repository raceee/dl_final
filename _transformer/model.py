import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn

class MoleculeBERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, num_hidden_layers=12, hidden_size=768, intermediate_size=3072):
        super(MoleculeBERTModel, self).__init__()
        # Create custom configuration
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            output_hidden_states=True,  # Needed for triplet loss
            return_dict=True  # Ensure outputs are returned as a dictionary
        )
        
        # Initialize model with custom configuration
        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)  # Single-layer MLM head

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
        
        Returns:
            dict: A dictionary containing logits and hidden states.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        logits = self.mlm_head(sequence_output)  # Project hidden states to vocab size
        return {
            "logits": logits,  # For MLM loss
            "hidden_states": outputs.hidden_states  # For triplet loss
        }