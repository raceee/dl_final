import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from util.training_set import GetTrainingSet
from graph.dataset import GraphNN_Dataset
from graph.model import GraphNN_Model


    

    


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    input_dim = 1
    hidden_dim = 64 # 32, 64, 128, 256
    output_dim = 118
    learning_rate = 0.001 # 0.01 - 0.0001
    mask_rate = 0.15 # 10% - 30% test range
    # add number of GNN layers as hp if possible later 2-4
    epochs = 10
    batch_size = 32 # 16, 32, 64, 128

    model = GraphNN_Model(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_dataloader:
            data = data.to(device)

            mask = torch.rand(data.x.shape[0]) < mask_rate
            original_features = data.x[mask].clone()
            data.x[mask] = 0

            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out[mask], original_features.squeeze().long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")

    torch.save(model.state_dict(), 'model_gnn.pth')