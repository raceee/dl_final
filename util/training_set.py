import os
import pandas as pd

class GetTrainingSet:
    def __init__(self, path):
        self.path = path
        self.training_set = None
        self.last_unique_smiles = None
        self.random_unique_smiles = None

    def get_training_data(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        training_dataset_path = os.path.join(data_dir, "training_dataset.csv")
        last_unique_smiles_path = os.path.join(data_dir, "last_unique_smiles.csv")
        random_unique_smiles_path = os.path.join(data_dir, "random_unique_smiles.csv")

        # Check if the main dataset exists
        if not os.path.exists(training_dataset_path):
            print("Training dataset not found. Generating it...")
            self.training_set = self.load_training_set()
        else:
            print("Training dataset found. Loading it...")
            self.training_set = pd.read_csv(training_dataset_path)

        # Check if last_unique_smiles.csv exists
        if not os.path.exists(last_unique_smiles_path):
            print("Last unique smiles file not found. Generating it...")
            self.last_unique_smiles = self.load_last_unique_smiles()
        else:
            print("Last unique smiles file found. Loading it...")
            self.last_unique_smiles = pd.read_csv(last_unique_smiles_path)

        # Check if random_unique_smiles.csv exists
        if not os.path.exists(random_unique_smiles_path):
            print("Random unique smiles file not found. Generating it...")
            self.random_unique_smiles = self.load_random_unique_smiles()
        else:
            print("Random unique smiles file found. Loading it...")
            self.random_unique_smiles = pd.read_csv(random_unique_smiles_path)

        # Return all datasets
        return self.training_set, self.last_unique_smiles, self.random_unique_smiles
        
    def check_dataset_exists(self):
        # Check if the dataset exists in the current or sibling 'data' directory
        if os.path.exists(self.path):
            return True
        sibling_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', os.path.basename(self.path))
        if os.path.exists(sibling_data_path):
            self.path = sibling_data_path
            return True
        return False

    def load_training_set(self):
        unique_smiles = set()
        smiles_list = []

        desired_unique_count = 100000

        chunksize = 100000

        try:
            shuffled_df = pd.read_csv(self.path, usecols=['molecule_smiles']).sample(frac=1, random_state=42)

            # Read the CSV file in chunks from the beginning
            for chunk in pd.read_csv(self.path, usecols=['molecule_smiles'], chunksize=chunksize):
                print(f"Processing chunk with {len(chunk)} rows")
                # Iterate over each molecule_smiles in the chunk
                for smile in chunk['molecule_smiles']:
                    if smile not in unique_smiles:
                        unique_smiles.add(smile)
                        smiles_list.append(smile)
                        if len(unique_smiles) >= desired_unique_count:
                            break
                if len(unique_smiles) >= desired_unique_count:
                    break

            print(f"Total unique smiles collected from the beginning: {len(unique_smiles)}")

            # Convert the list of unique smiles to a DataFrame
            result_df = pd.DataFrame({'molecule_smiles': smiles_list})

            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            result_df.to_csv(os.path.join(data_dir, "training_dataset.csv"), index=False)
            print("Data successfully written to data/training_dataset.csv")

            return result_df

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def load_last_unique_smiles(self):
        try:
            df = pd.read_csv(self.path, usecols=['molecule_smiles'])
            last_smiles = df.tail(10000)['molecule_smiles'].tolist()  # Convert to list for reversing

            unique_smiles = set()
            smiles_list = []

            # Collect up to 10 unique molecules from the end
            for smile in reversed(last_smiles):  # Now reversed works correctly
                if smile not in unique_smiles:
                    unique_smiles.add(smile)
                    smiles_list.append(smile)
                    if len(unique_smiles) >= 10:
                        break

            print(f"Total unique smiles collected from the end: {len(unique_smiles)}")

            result_df = pd.DataFrame({'molecule_smiles': smiles_list})

            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            result_df.to_csv(os.path.join(data_dir, "last_unique_smiles.csv"), index=False)
            print("Data successfully written to data/last_unique_smiles.csv")

            return result_df

        except Exception as e:
            print(f"An error occurred while loading the last unique smiles: {e}")
            return None


    def load_random_unique_smiles(self):
        # Load 10 random unique smiles from the dataset
        try:
            df = pd.read_csv(self.path, usecols=['molecule_smiles'])
            
            unique_smiles = set(df['molecule_smiles'])
            random_smiles = list(unique_smiles)[:10]

            print(f"Total random unique smiles collected: {len(random_smiles)}")

            result_df = pd.DataFrame({'molecule_smiles': random_smiles})

            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            result_df.to_csv(os.path.join(data_dir, "random_unique_smiles.csv"), index=False)
            print("Data successfully written to data/random_unique_smiles.csv")

            return result_df

        except Exception as e:
            print(f"An error occurred while loading the random unique smiles: {e}")
            return None
