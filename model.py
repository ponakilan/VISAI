from torch import nn

class ConsumptionEstimator(nn.module):
    def __init__(self):
        super().__init__()
        
        # Define the layers
        self.conv1 = nn.Conv1d()
