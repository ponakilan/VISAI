import torch
import torch.nn as nn

class ConsumptionPredictor(nn.Module):
    def __init__(self):
        super(ConsumptionPredictor, self).__init__()
    
        self.lstm = nn.LSTM(input_size=8, hidden_size=5, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=60, out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
