import streamlit as st
import torch
import numpy as np
import pickle
from dataset.dataset import PowerConsumptionDataset
from torch.utils.data import DataLoader

model = torch.load('model_33.pt', map_location=torch.device('cpu'))
model.eval()
dataset = PowerConsumptionDataset(
    data_dir="data",
    sm_path="data/01_sm",
    plugs_path="data/01_plugs",
    num_rows=24*3600,
    sequence_length=220,
    max_len=86400
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

forecast = []
if not 'last_update' in st.session_state:
    st.session_state.last_update = 0
last_update = st.session_state.last_update
st.title("Power consumption forecast")

for i, (sequences, _) in enumerate(dataloader):
    if i >= st.session_state.last_update:
        outputs = model(sequences)
        forecast.extend(outputs.detach().numpy().flatten().tolist())
    if i == (last_update + 2000):
        break

scaler = pickle.load(open('data/scaler.pkl', 'rb'))
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten().tolist()

st.line_chart({"forecast": forecast})
st.text(f'Average consumption: {sum(forecast)/len(forecast):.2f} Wh')
st.session_state.last_update = i

st.button('Update forecast')

