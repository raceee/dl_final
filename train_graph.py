import os
import pandas as pd
from util.training_set import GetTrainingSet
from graph.model import GraphModel
from graph.dataset import Dataset
from torch.utils.data import DataLoader

training_df, last_unique_df, random_in_training_df = GetTrainingSet("data/train_sample.csv")

training_dataset = Dataset(training_df)
last_unique_dataset = Dataset(last_unique_df)
random_in_training_dataset = Dataset(random_in_training_df)

training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
last_unique_dataloader = DataLoader(last_unique_dataset, batch_size=32, shuffle=True)
random_in_training_dataloader = DataLoader(random_in_training_dataset, batch_size=32, shuffle=True)

for i in hypers:
    model = GraphModel(num_layers=i["num_layers"], num_features=i["num_features"], num_classes=i["num_classes"])
    model.train(training_df, last_unique_df, random_in_training_df, i["epochs"], i["batch_size"], i["learning_rate"], i["weight_decay"], i["device"])
    model.save_model(f"model_{i['num_layers']}_{i['num_features']}_{i['num_classes']}")