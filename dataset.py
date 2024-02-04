import os
import pandas as pd
from torch.utils.data import Dataset

class PowerConsumption(Dataset):
    def __init__(self, plugs_path, sm_path, transform=None):
        