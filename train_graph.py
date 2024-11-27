import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from util.training_set import GetTrainingSet
from graph.dataset import GraphNN_Dataset
from graph.model import GraphNN_Model


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    


if __name__ == '__main__':

    path = "data/train_sample.csv"
    training_df, last_unique_df, random_in_training_df = GetTrainingSet(path).get_training_data()

    train_size = int(0.8 * len(training_df))
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:]

    column_name = 'molecule_smiles'
    train_smiles_list = train_df[column_name].tolist()
    val_smiles_list = val_df[column_name].tolist()

    train_smiles_list = train_df['molecule_smiles'].tolist()
    val_smiles_list = val_df['molecule_smiles'].tolist()

    train_dataset = GraphNN_Dataset(train_smiles_list)
    val_dataset = GraphNN_Dataset(val_smiles_list)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # graph_data = train_dataset[0]
    # print(graph_data.x)
    # print(graph_data.edge_index)

    # input_dim = 1
    # hidden_dim = 128
    # output_dim = 1
    # model = GraphNN_Model(input_dim, hidden_dim, output_dim)

    
