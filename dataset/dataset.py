import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


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
        Generate the dataset by concatenating the smart meter data and the plugself.data. 
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
        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data.values))
        pickle.dump(scaler, open(f'{self.data_dir}/cons_scaler.pkl', 'wb'))

    def __len__(self):
        return self.data.shape[0] - self.sequence_length - 1
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.sequence_length

        # Get the sequence and target
        sequence = torch.tensor(self.data.iloc[start_idx:end_idx, :].values.astype(np.float32))
        target = torch.tensor(self.data.iloc[end_idx, 0].astype(np.float32))

        return sequence, target
    

class Recommendations(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path).drop(columns=['Formatted Date'])

        # Encode the text data
        encoder = LabelEncoder()
        self.data['Summary'] = encoder.fit_transform(self.data['Summary'])
        self.data['Precip Type'] = encoder.fit_transform(self.data['Precip Type'])
        self.data['Daily Summary'] = encoder.fit_transform(self.data['Daily Summary'])
        self.data['lights'] = encoder.fit_transform(self.data['lights'])

        # Separate X and y
        self.X =self.data.iloc[:, :-4]
        self.y =self.data.iloc[:, -4:]

        # Standardize the data
        X_scaler = StandardScaler().fit(X=self.X)
        y_scaler = StandardScaler().fit(X=self.y)
        self.X = X_scaler.transform(self.X)
        self.y = y_scaler.transform(self.y)

        # Save the scaler objects
        pickle.dump(X_scaler, open('data/X_scaler.pkl', 'wb'))
        pickle.dump(y_scaler, open('data/y_scaler.pkl', 'wb'))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        return X, y