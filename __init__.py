from util.training_set import GetTrainingSet
import os
import pandas as pd

def make_datasets():
    # check if training set is made and Load the training set if "training_dataset.csv" does not exists

    if not os.path.exists("./training_dataset.csv"):
        print("Creating training dataset")
        GetTrainingSet("data/train_sample.csv")
        training_set = pd.read_csv("training_dataset.csv")
        non_training_set = pd.read_csv("data/test_sample.csv")
        random_unique_smiles = pd.read_csv("random_unique_smiles.csv")

    else:
        print("Loading training dataset")
        training_set = pd.read_csv("training_dataset.csv")
        non_training_set = pd.read_csv("data/test_sample.csv")
        random_unique_smiles = pd.read_csv("random_unique_smiles.csv")
    return training_set, non_training_set, random_unique_smiles # this are each a pandas dataframe of one column called molecule_smiles