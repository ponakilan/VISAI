import pandas as pd
from dataset.dataset import PowerConsumptionDataset


dataset = PowerConsumptionDataset(
    plugs_path="../data/01_plugs",
    sm_path="../data/01_sm",
    num_rows=pd.read_csv("data/01_sm/2012-06-01.csv", header=None).shape[0],
    sequence_length=10
)

print(len(dataset))