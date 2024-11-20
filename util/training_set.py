import os
import pandas as pd

class GetTrainingSet:
    def __init__(self, path):
        self.path = path
        if self.check_dataset_exists():
            self.training_set = self.load_training_set()
            self.last_unique_smiles = self.load_last_unique_smiles()
            self.random_unique_smiles = self.load_random_unique_smiles()
        else:
            print("Dataset not found in the current or sibling directories.")
            self.training_set = None
            self.last_unique_smiles = None
            self.random_unique_smiles = None
    
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

        # Define the desired number of unique entries
        desired_unique_count = 100000

        # Define a suitable chunksize (adjust based on your system's memory)
        chunksize = 100000

        try:
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

            # Save the result to a CSV file without the index in the 'data' directory
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
        # Load the last rows of the CSV file to get unique smiles from the end
        try:
            # Read the entire CSV file to get the last 10 unique smiles
            # Alternatively, you could use a larger chunk for efficiency if needed
            df = pd.read_csv(self.path, usecols=['molecule_smiles'])
            last_smiles = df.tail(10000)['molecule_smiles']  # Read the last 10,000 rows for efficiency
            
            unique_smiles = set()
            smiles_list = []

            # Collect up to 10 unique molecules from the end
            for smile in reversed(last_smiles):
                if smile not in unique_smiles:
                    unique_smiles.add(smile)
                    smiles_list.append(smile)
                    if len(unique_smiles) >= 10:
                        break

            print(f"Total unique smiles collected from the end: {len(unique_smiles)}")

            # Convert the list of unique smiles to a DataFrame
            result_df = pd.DataFrame({'molecule_smiles': smiles_list})

            # Save the result to a CSV file without the index in the 'data' directory
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
            # Read the entire CSV file to sample random unique smiles
            df = pd.read_csv(self.path, usecols=['molecule_smiles'])
            
            unique_smiles = set(df['molecule_smiles'])
            random_smiles = list(unique_smiles)[:10]  # Get the first 10 unique smiles (randomized if needed)

            print(f"Total random unique smiles collected: {len(random_smiles)}")

            # Convert the list of random unique smiles to a DataFrame
            result_df = pd.DataFrame({'molecule_smiles': random_smiles})

            # Save the result to a CSV file without the index in the 'data' directory
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            result_df.to_csv(os.path.join(data_dir, "random_unique_smiles.csv"), index=False)
            print("Data successfully written to data/random_unique_smiles.csv")

            return result_df

        except Exception as e:
            print(f"An error occurred while loading the random unique smiles: {e}")
            return None