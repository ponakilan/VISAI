import torch

model = torch.load('/home/akilan/Downloads/model_13496.pt', map_location=torch.device('cpu'))
print(model)
