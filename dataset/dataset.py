import os
import pandas as pd
from torch.utils.data import Dataset

class PowerConsumptionDataset(Dataset):
    def __init__(self, plugs_path, sm_path, num_rows, sequence_length, transform=None):
        self.plugs_path = plugs_path
        self.sm_path = sm_path
        self.num_rows = num_rows
        self.sequence_length = sequence_length
        self.transform = transform

        self.list_sm = os.listdir(self.sm_path)
        self.list_plugs_path = os.listdir(self.plugs_path)
        self.num_appliances = len(self.list_plugs_path)

    def __len__(self):
        return (len(self.list_sm) * self.num_rows) // self.sequence_length

    def __getitem__(self, index):
        file_id = index // self.num_rows
        sm_file = self.list_sm[file_id]
        plug_folders = [os.path.join(self.plugs_path, plug) for plug in self.list_plugs_path]

        # Get the dataframe for the smart meter
        sm_df = pd.read_csv(os.path.join(self.sm_path, sm_file), header=None)

        # Store the dataframes for each plug
        plug_dfs = {}
        for i, plug_folder in enumerate(plug_folders):
            plug_dfs[i] = pd.read_csv(os.path.join(plug_folder, sm_file), header=None)

        # Calculate the start and end index for the sequence
        start_index = index % self.num_rows
        end_index = start_index + self.sequence_length

        # Get the sequence for the smart meter
        sm_sequence_X = sm_df.iloc[start_index:end_index, 1].values
        sm_sequence_y = sm_df.iloc[end_index, 1].values

        # Get the sequence for each plug
        plug_sequences_X = {}
        plug_sequences_y = {}
        for i in range(self.num_appliances):
            plug_sequences_X[i] = plug_dfs[i].iloc[start_index:end_index, 1].values
            plug_sequences_y[i] = plug_dfs[i].iloc[end_index, 1].values

        print(sm_sequence_X, sm_sequence_y)
