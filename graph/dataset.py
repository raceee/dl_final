
import time
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
from rdkit import Chem

class GraphNN_Dataset(Dataset):
    def __init__(self, smiles_list, root=None, transform=None, pre_transform=None):
        """
        Args:
            data (list): A list of SMILES strings.
        """
        self.smiles_list = smiles_list
        self.data_list = []
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        """Processes the SMILES strings into graph data."""
        start_time = time.time()
        
        for idx in range(len(self.smiles_list)):
            smiles = self.smiles_list[idx]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Unable to parse SMILES at index {idx}: {smiles}")
                continue

            self.data_list.append(self._mol_to_graph_data(mol))

        end_time = time.time()
        print(f"Data processing time: {end_time - start_time:.2f} seconds")

    def _mol_to_graph_data(self, mol):
        """Converts an RDKit mol object into a torch_geometric Data object."""
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atoms, dtype=torch.long).unsqueeze(1)

        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        return data
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]