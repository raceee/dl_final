import os
import json
import matplotlib.pyplot as plt

import _globals

def plot_once(results, hidden_dims, num_hidden_layerss):
    plt.figure(figsize=(10, 8))
    if isinstance(hidden_dims, list):
        title = f"hid_layers_{num_hidden_layerss}"
        plot_title = f"Num Hidden Layers: {num_hidden_layerss}"
        legend_title = "Hidden Dimension"

        for hidden_dim in hidden_dims:
            line = results[f"{hidden_dim}, {num_hidden_layerss}"]["val_loss"]
            plt.plot(line, label=f"{hidden_dim}")

    elif isinstance(num_hidden_layerss, list):
        title = f"hdim_{hidden_dims}"
        plot_title = f"Hidden Dimension: {hidden_dims}"
        legend_title = "Num Hidden Layers"

        for num_hidden_layers in num_hidden_layerss:
            line = results[f"{hidden_dims}, {num_hidden_layers}"]["val_loss"]
            plt.plot(line, label=f"{num_hidden_layers}")


    plt.title(plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title=legend_title)
    plt.tight_layout()

    save_path = "gs_plots"
    if save_path:
        plot_path = os.path.join(save_path, f"{title}.png")
        plt.savefig(plot_path)
        print(f"Loss curves saved to {plot_path}")

    # Close the plot to free memory
    plt.close()

def plot_gs_loss_curves(path, hidden_dims, num_hidden_layerss):
    with open(path, "r") as f:
        results = json.load(f)

    # hidden_dim fixed
    for hidden_dim in hidden_dims:
        plot_once(results, hidden_dim, num_hidden_layerss)

    # num_hidden_layers fixed
    for num_hidden_layers in num_hidden_layerss:
        plot_once(results, hidden_dims, num_hidden_layers)

if __name__ == "__main__":
    path = ""
    plot_gs_loss_curves(path,
                        _globals.hidden_dims,
                        _globals.num_hidden_layerss)