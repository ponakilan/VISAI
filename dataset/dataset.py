import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class PowerConsumptionDataset(Dataset):
    def __init__(self, sm_path: str, plugs_path: str, num_rows: int, sequence_length: int):
        self.sm_path = sm_path
        self.plugs_path = plugs_path
        self.num_rows = num_rows
        self.sequence_length = sequence_length

        for i, file in enumerate(os.listdir(self.sm_path)):
            if file.endswith(".csv"):
                # Load the smart meter data and the plug data as pandas dataframes
                sm_df = pd.DataFrame(pd.read_csv(os.path.join(self.sm_path, file)).iloc[:, 0])
                plugs_01 = self.get_plug_df(file, '01')
                plugs_02 = self.get_plug_df(file, '02')
                plugs_03 = self.get_plug_df(file, '03')
                plugs_04 = self.get_plug_df(file, '04')
                plugs_05 = self.get_plug_df(file, '05')
                plugs_06 = self.get_plug_df(file, '06')
                plugs_07 = self.get_plug_df(file, '07')

                if i == 0:
                    # Concatenate all the dataframes in axis 1
                    self.df = pd.concat([sm_df, plugs_01, plugs_02, plugs_03, plugs_04, plugs_05, plugs_06, plugs_07], axis=1)
                else:
                    # Concatenate all the dataframes in axis 0
                    self.df = pd.concat([self.df, sm_df, plugs_01, plugs_02, plugs_03, plugs_04, plugs_05, plugs_06, plugs_07], axis=0)

    def __len__(self):
        return (self.num_rows * len(os.listdir(self.sm_path))) - self.sequence_length
    
    def get_plug_df(self, filename: str, folder: str) -> pd.DataFrame:
        """
        Open the specified plug file and return it as a pandas dataframe. If the file is not found, 
        then return a dummy dataframe.

        Args:
            filename (str): Name of the file
            folder (str): Name of the folder

        Returns:
            df (pandas.DataFrame): Pandas dataframe containing the plug data
        """

        try:
            file_path = os.path.join(self.plugs_path, os.path.join(folder, filename))
            return pd.read_csv(file_path, header=None)
        
        except FileNotFoundError:
            return pd.read_csv(os.path.join(self.plugs_path, 'plugs_dummy.csv'), header=None)
    
    def __getitem__(self, idx):
        sm_sequence = self.get_sm_sequence(idx)
        return sm_sequence
