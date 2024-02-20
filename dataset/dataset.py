import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import functools


class PowerConsumptionDataset(Dataset):
    def __init__(self, data_dir: str, sm_path: str, plugs_path: str, num_rows: int, sequence_length: int, max_len: int):
        self.data_dir = data_dir
        self.sm_path = sm_path
        self.plugs_path = plugs_path
        self.num_rows = num_rows
        self.sequence_length = sequence_length
        self.max_len = max_len

        # Generate the dataset
        self.generate_dataset()

    # def get_plug_df(self, filename: str, folder: str) -> pd.DataFrame:
    #     """
    #     Open the specified plug file and return it as a pandas dataframe. If the file is not found, 
    #     then return a dummy dataframe.

    #     Args:
    #         filename (str): Name of the file
    #         folder (str): Name of the folder

    #     Returns:
    #         df (pandas.DataFrame): Pandas dataframe containing the plug data
    #     """

    #     try:
    #         file_path = os.path.join(self.plugs_path, os.path.join(folder, filename))
    #         return pd.read_csv(file_path, header=None)
        
    #     except FileNotFoundError:
    #         return pd.read_csv(os.path.join(self.plugs_path, 'plugs_dummy.csv'), header=None)

    def generate_dataset(self):
        """
        Generate the dataset by concatenating the smart meter data and the plug data. 
        The concatenated data is saved in the concat folder.
        """
        sm_files = os.listdir(self.sm_path)
        sm_files.sort()
        self.data = pd.DataFrame()
        for file in sm_files:
            sm_df = pd.read_csv(f'{self.sm_path}/{file}').iloc[:, 0]
            self.data = pd.concat([self.data, sm_df], ignore_index=True)
            if self.data.shape[0] > self.max_len:
                break

    def __len__(self):
        return self.data.shape[0] - self.sequence_length - 1
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.sequence_length

        # Get the sequence and target
        sequence = self.data.iloc[start_idx:end_idx, :].values.astype(np.float32)
        target = np.array(self.data.iloc[end_idx, 0], dtype=np.float32)

        return sequence, target
