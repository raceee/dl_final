import math
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from util.training_set import GetTrainingSet
from util.trainer import Trainer
from graph.dataset import GraphNN_Dataset
from graph.model import GraphNN_Model


# Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Main function
def main():
    # Hyperparameters
    batch_size = 32 # 16, 32, 64, 128
    num_layers = 2 # 2, 3, 4, 5
    input_dim = 1 # what does this input_dim represent? 1 was a suggestion
    hidden_dim = 64 # 32, 64, 128, 256
    output_dim = 118 # what is the exact amount of classes? 118 was a suggestion
    learning_rate = 0.001 # 0.01 - 0.0001
    gamma = 0.1
    epochs = 10

    # Load and prepare data
    path = "data/train_sample.csv"
    training_df, last_unique_df, random_in_training_df = GetTrainingSet(path).get_training_data()
    
    train_size = int(0.8 * len(training_df))
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:]

    train_smiles_list = train_df['molecule_smiles'].tolist()
    val_smiles_list = val_df['molecule_smiles'].tolist()

    # Datasets and DataLoaders
    train_dataset = GraphNN_Dataset(train_smiles_list)
    val_dataset = GraphNN_Dataset(val_smiles_list)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and criterion
    model = GraphNN_Model(input_dim=input_dim,
                          hidden_dim=hidden_dim,
                          output_dim=output_dim, 
                          num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Calculate the step size for the scheduler
    num_epochs_per_decay = 30
    batches_per_epoch = math.ceil(len(train_dataloader) / batch_size)
    step_size = num_epochs_per_decay * batches_per_epoch

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize and run Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        criterion=criterion,
        model_name="GNN"
    )

    # Train the model
    history = trainer.train(num_epochs=epochs, save_best_model_path="graph_checkpoints/model.pth")

    # Plot and save the loss curves
    trainer.plot_loss_curves(
        history=history,
        model_name="Graph Neural Network",
        save_path="graph_checkpoints/loss_curves.png",
        show=True
    )
    
    # Not this far yet
    # path = ""
    # trainer.infer_clusters(path, tokenizer, method="umap")

# Entry point
if __name__ == "__main__":
    main()