import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
from dataset.dataset import PowerConsumptionDataset, Recommendations
from torch.utils.data import DataLoader

st.title("Eco-Energetix")

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
st.subheader("Power Consumption Forecast")

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

# Recommendations
st.subheader("Recommended Setting")

recommender = torch.load('trained_models/model_49.pt', map_location=torch.device('cpu'))
dataset = Recommendations('data/recommendation.csv')
dataloader = DataLoader(
    dataset=dataset,
    batch_size=10,
    shuffle=False
)

X, y = next(iter(dataloader))
predictions = recommender(X).detach().numpy()

# Rescale the outputs
y_scaler = pickle.load(open('data/y_scaler.pkl', 'rb'))
predictions = y_scaler.inverse_transform(predictions)

for i in range(len(predictions)):
    predictions[i][2] = abs(predictions[i][2])*10000
    if predictions[i][2] > 100:
        predictions[i][2] = 100.00
    elif predictions[i][2] < 50:
        predictions[i][2] = 50.00

    predictions[i][0] = int(predictions[i][0])
    predictions[i][1] = int(predictions[i][1])
    predictions[i][2] = int(predictions[i][2])
    predictions[i][3] = int(predictions[i][3])

df = pd.DataFrame(
    predictions,
    columns=["Air Conditioner", "Refridgerator", "Light", "Heater"]
)

st.table(df)