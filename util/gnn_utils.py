import time
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
from rdkit import Chem

class GNN_Utils:
    def __init__(self, device):
        self.device = device

    def process(self, smile):
        """Processes the SMILES strings into graph data."""
        
        # for idx in range(len(smiles_list)):
        #     smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smile}")

        return self._mol_to_graph_data(mol)

    def _mol_to_graph_data(self, mol):
        """Converts an RDKit mol object into a torch_geometric Data object."""
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atoms, dtype=torch.long).unsqueeze(1).to(self.device)

        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)

        data = Data(x=x, edge_index=edge_index)
        return data