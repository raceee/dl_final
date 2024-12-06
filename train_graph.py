import math
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from util.training_set import GetTrainingSet
from graph.dataset import GraphNN_Dataset
from graph.model import GraphNN_Model
from util.trainer_gnn import Trainer_GNN

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
    output_dim =  118
    num_hidden_layers = 2
    learning_rate = 0.001 # 0.01 - 0.0001
    gamma = 0.1

    num_epochs = 10
    batch_size = 32 # 16, 32, 64, 128

    training_df, _, _ = GetTrainingSet("data/train_sample.csv").get_training_data()

    train_size = int(0.8 * len(training_df))
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:]

    train_smiles_list = train_df['molecule_smiles'].tolist()
    val_smiles_list = val_df['molecule_smiles'].tolist()

    # Datasets and DataLoader
    train_dataset = GraphNN_Dataset(train_smiles_list, root='data/graph_data')
    val_dataset = GraphNN_Dataset(val_smiles_list, root='data/graph_data')
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = GraphNN_Model(input_dim, hidden_dim, output_dim, num_hidden_layers=num_hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Calculate the step size for the scheduler
    num_epochs_per_decay = 30
    batches_per_epoch = math.ceil(len(train_dataloader) / batch_size)
    step_size = num_epochs_per_decay * batches_per_epoch

    scheduler = StepLR(optimizer,
                       step_size=step_size,
                       gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize and run Trainer
    trainer = Trainer_GNN(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=criterion
    )

    # Train the model
    history = trainer.train(num_epochs=num_epochs,
                            save_best_model_path=f"gnn_checkpoints/hidden_dim_{hidden_dim}.pth") # need another name most likely
    
    print(history)
    # Plot and save the loss curves
    



if __name__ == '__main__':
    main()