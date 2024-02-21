from torch import nn
import torch

class RecomModel(nn.Module):
    def __init__(self):
        super(RecomModel, self).__init__()
        self.fc1 = nn.Linear(11, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x