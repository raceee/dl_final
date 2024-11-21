'''
classes that are needed to transform the dataset into a format that can be used by the gnn 

this will include:
- RDKit transformations of our SMILES strings
- Those RDKit graphs put into Pytorch Geometric format
- finally returning PyTorch DataLoader to load the data into the model

'''

class GraphDataset:
    def __init__(self):
        
        pass

    def __getitem__(self, idx):
        # get smiles string with idx
        # get target with idx
        # get graph of the SMILES string with RDKit
        # get graph in PyTorch Geometric format
        # return the graph and target
    
    def __len__(self):
        # return the length of the dataset
    
    def make_graph(self, smiles):
        # make a graph from the SMILES string w/ RDKit
        pass