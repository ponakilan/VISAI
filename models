import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.lstm = nn.LSTM(input_size=64, hidden_size=5, batch_first=True)
        self.linear = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
       
        x = self.conv1d_1(x)
        x = torch.relu(x)
        
        x = self.conv1d_2(x)
        x = torch.relu(x)
        
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
