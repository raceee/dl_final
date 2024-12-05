import math
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from util.trainer import Trainer

from util.training_set import GetTrainingSet
from graph.dataset import GraphNN_Dataset
from graph.model import GraphNN_Model

# Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
def main():
    print(f"Using device: {device}")

    input_dim = 1
    hidden_dim = 64 # 32, 64, 128, 256
    output_dim =  118 # 118 because of the number of elements in the periodic table
    num_layers = 2
    learning_rate = 0.001 # 0.01 - 0.0001
    gamma = 0.1

    # add number of GNN layers as hp if possible later 2-4
    epochs = 10
    batch_size = 32 # 16, 32, 64, 128

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

    # graph_data = train_dataset[0]
    # print(len(train_dataset))
    # for data in train_dataset:
    #     print(data.num_nodes)
    #     print(data.x.shape)
        
    # print(graph_data.x)
    # print(graph_data.edge_index)
    # print(f"Number of nodes: {graph_data.num_nodes}")
    # print(f"Number of edges: {graph_data.num_edges}")
    # print(f"Shape of node features: {graph_data.x.shape}")
    # print(f"Shape of edge indices: {graph_data.edge_index.shape}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = GraphNN_Model(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Calculate the step size for the scheduler
    num_epochs_per_decay = 30
    batches_per_epoch = math.ceil(len(train_dataloader) / batch_size)
    step_size = num_epochs_per_decay * batches_per_epoch

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            data = batch.to(device)

            # print(f"Number of nodes: {data.num_nodes}") # 118
            # print(f"Number of edges: {data.num_edges}") # 250
            # print(f"Shape of node features: {data.x.shape}") # 118, 1
            # print(f"Shape of edge indices: {data.edge_index.shape}") # 2, 250


            mask = torch.rand(data.x.shape[0]) < 0.15 # 15% because BERT
            labels = data.x[mask].clone()
            data.x[mask] = 0

            # print(f"Mask: {mask}")
            # print(f"Original Features: {original_features}")
            # print(f"Mask Size: {mask.shape}") # 118
            # print(f"Original Features Size: {original_features.shape}") # 18, 1 then 17, 1, then 19 then 20 then 14
            

            # Forward pass
            outputs = model(data)

            # print(f"Outputs Size: {outputs.shape}")
            # print(f"Outputs Masked Size: {outputs[mask].shape}")
            # print()

            loss = criterion(outputs[mask], labels.squeeze().long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.2f}")


    # # torch.save(model.state_dict(), 'model_gnn.pth') 


if __name__ == '__main__':
    main()