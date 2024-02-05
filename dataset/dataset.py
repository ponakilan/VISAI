import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class PowerConsumptionDataset(Dataset):
    def __init__(self, sm_path, plugs_path, num_rows, sequence_length):
        self.sm_path = sm_path
        self.plugs_path = plugs_path
        self.num_rows = num_rows
        self.sequence_length = sequence_length

    def __len__(self):
        return (self.num_rows * len(os.listdir(self.sm_path))) - self.sequence_length
    
    def get_plug_sequence(self, filename, idx):
        pass
    
    def get_sm_sequence(self, idx: int):
        """
        Retrieve the sequence of the specified length from the smart meter data.

        Args:
            idx (int): Index of the sequence
        Returns:
            sequence (tuple): Tuple containing the sequence and the label
        """

        # Calculate the file index, start index, and end index
        file_idx = idx // self.num_rows
        start_idx = idx % self.num_rows
        end_idx = start_idx + self.sequence_length

        # Get the file name and load the file as pandas as dataframe
        sm_files = os.listdir(self.sm_path)
        sm_files.sort()
        sm_file = sm_files[file_idx]
        sm_df = pd.read_csv(os.path.join(self.sm_path, sm_file), header=None)

        # If the end index is greater than the number of rows in the file, then we need to load the next file
        if end_idx + 1 > self.num_rows:
            file_idx_2 = file_idx + 1
            sm_file_2 = sm_files[file_idx_2]
            sm_df_2 = pd.read_csv(os.path.join(self.sm_path, sm_file_2), header=None)
            sm_df = pd.concat([sm_df, sm_df_2], axis=0)

        # Get the sequence
        sequence_X = sm_df.iloc[start_idx:end_idx, 0].values
        sequence_y = sm_df.iloc[end_idx, 0]

        return sequence_X, sequence_y

    def __getitem__(self, idx):
        sm_sequence = self.get_sm_sequence(idx)
        return sm_sequence
