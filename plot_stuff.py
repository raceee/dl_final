import os
import json
import matplotlib.pyplot as plt

import _globals

def plot_gs_loss_curves(results_path):
    # Load the results
    with open(results_path, "r") as f:
        hp_results = json.load(f)

    # Plot the loss curves
    plt.figure(figsize=(10, 8))
    for key, value in hp_results.items():
        plt.plot(value["val_loss"], label=f"{key}")
    
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curves")
    plt.legend(title="Architectures")
    plt.show()