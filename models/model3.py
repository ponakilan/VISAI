import torch
import torch.nn as nn

class ConsumptionPredictor(nn.Module):
    def __init__(self, sequence_length):
        super(ConsumptionPredictor, self).__init__()
    
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(in_features=sequence_length*5, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
