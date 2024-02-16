import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PowerConsumptionDataset(Dataset):
    def __init__(self, data_dir: str, sm_path: str, plugs_path: str, num_rows: int, sequence_length: int, max_len: int):
        self.data_dir = data_dir
        self.sm_path = sm_path
        self.plugs_path = plugs_path
        self.num_rows = num_rows
        self.sequence_length = sequence_length

        # Generate the dataset
        self.generate_dataset()

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

    def generate_dataset(self):
        """
        Generate the dataset by concatenating the smart meter data and the plug data. 
        The concatenated data is saved in the concat folder.
        """
        sm_files = os.listdir(self.sm_path)
        sm_files.sort()
        for file in sm_files:
            if not os.path.exists(os.path.join(self.data_dir, f'concat/{file}')) and file.endswith(".csv"):
                # Load the smart meter data and the plug data as pandas dataframes
                sm_df = pd.DataFrame(pd.read_csv(os.path.join(self.sm_path, file)).iloc[:, 0])
                plugs_01 = self.get_plug_df(file, '01')
                plugs_02 = self.get_plug_df(file, '02')
                plugs_03 = self.get_plug_df(file, '03')
                plugs_04 = self.get_plug_df(file, '04')
                plugs_05 = self.get_plug_df(file, '05')
                plugs_06 = self.get_plug_df(file, '06')
                plugs_07 = self.get_plug_df(file, '07')

                # Concatenate all the dataframes in axis 1 and save it as the final dataframe
                df = pd.concat([sm_df, plugs_01, plugs_02, plugs_03, plugs_04, plugs_05, plugs_06, plugs_07], axis=1)
                df.to_csv(os.path.join(self.data_dir, f'concat/{file}'), index=False)

    def __len__(self):
        if self.max_len > len(os.listdir(os.path.join(self.data_dir, 'concat')))*86400 - self.sequence_length - 1:
            return len(os.listdir(os.path.join(self.data_dir, 'concat')))*86400 - self.sequence_length - 1
        else:
            return self.max_len
    
    def __getitem__(self, idx):
        # Get the list of files
        files = os.listdir(os.path.join(self.data_dir, 'concat'))

        # Calculate the file index
        file_idx = idx // self.num_rows

        # Calculate the start and end index
        start_idx = idx % self.num_rows
        end_idx = start_idx + self.sequence_length

        # Open the file
        df = pd.read_csv(os.path.join(self.data_dir, f'concat/{files[file_idx]}'))

        # Check if there is need to open another file
        if start_idx > (end_idx % self.num_rows):
            file_idx_2 = file_idx + 1
            df_2 = pd.read_csv(os.path.join(self.data_dir, f'concat/{files[file_idx_2]}'))
            df = pd.concat([df, df_2], axis=0)

        # Get the sequence and target
        sequence = df.iloc[start_idx:end_idx, :].values.astype(np.float32)
        target = np.array(df.iloc[end_idx, 0], dtype=np.float32)

        return sequence, target
