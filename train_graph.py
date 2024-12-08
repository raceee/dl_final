import json
import math
import torch
import pandas as pd
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

from util.training_set import GetTrainingSet
from graph.dataset import GraphNN_Dataset
from graph.model import GraphNN_Model
from graph.model_wo import GraphNN_Model_WO
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

    # Not interesting
    learning_rate = 0.01
    gamma = 0.1

    num_epochs = 10
    batch_size = 128
    
    input_dim = 1
    output_dim =  118

    # Interesting to grid search on
    hidden_dims = [32, 64, 128, 256]
    num_hidden_layerss = [2, 3, 4]

    # Load the training data
    # data_path = "data/train_sample.csv"
    data_path = "data/train.csv"
    training_df, last_unique_df, random_unique_df = GetTrainingSet(data_path).get_training_data()

    train_size = int(0.8 * len(training_df))
    train_df = training_df.iloc[:train_size]
    val_df = training_df.iloc[train_size:]

    train_smiles_list = train_df['molecule_smiles'].tolist()
    val_smiles_list = val_df['molecule_smiles'].tolist()

    # Datasets and DataLoader
    train_dataset = GraphNN_Dataset(train_smiles_list, root='gnn_root/graph_data')
    val_dataset = GraphNN_Dataset(val_smiles_list, root='gnn_root/graph_data')
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Grid search loop here
    hp_results = {}
    for hidden_dim in hidden_dims:
        for num_hidden_layers in num_hidden_layerss:
            print(f"Training model with hidden_dim={hidden_dim} and num_hidden_layers={num_hidden_layers}")
            

            # Initialize model, optimizer, and scheduler
            # model = GraphNN_Model(input_dim, hidden_dim, output_dim, num_hidden_layers=num_hidden_layers).to(device)
            model = GraphNN_Model_WO(input_dim, hidden_dim, output_dim, num_hidden_layers=num_hidden_layers).to(device)
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
                                    smiles_df_path="data/last_unique_smiles.csv",
                                    save_best_model_path=f"gnn_checkpoints/hidden_dim_{hidden_dim}.pth")
            
            # Plot and save the loss curves
            # trainer.plot_loss_curves(
            #     history=history,
            #     model_name="GraphNN with Adam",
            #     save_path="plots",
            #     show_plot=False
            # )

            # trainer.infer_clusters("data/last_unique_smiles.csv", method="umap", show_plot=False)

            # Save the results
            hp_results[f"{hidden_dim}, {num_hidden_layers}"] = history

    # Save the results to a JSON file
    now = datetime.now().strftime("%y%m_%d%H")
    with open(f"gs_results/{now}.json", "w") as f:
        json.dump(hp_results, f)

if __name__ == '__main__':
    main()