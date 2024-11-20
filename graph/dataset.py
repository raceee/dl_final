'''
classes that are needed to transform the dataset into a format that can be used by the gnn 

this will include:
- RDKit transformations of our SMILES strings
- Those RDKit graphs put into Pytorch Geometric format
- finally returning PyTorch DataLoader to load the data into the model

'''