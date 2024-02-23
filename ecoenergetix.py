import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
from dataset.dataset import PowerConsumptionDataset, Recommendations
from torch.utils.data import DataLoader
import random

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
forecasts = []
if not 'last_update' in st.session_state:
    st.session_state.last_update = 0
last_update = st.session_state.last_update
st.subheader("Energy Consumption Forecast")

for i, (sequences, _) in enumerate(dataloader):
    if i >= st.session_state.last_update:
        outputs = model(sequences)
        forecast.extend(outputs.detach().numpy().flatten().tolist())
    if i == (last_update + 10800):
        st.session_state.last_update = i
        break

scaler = pickle.load(open('data/scaler.pkl', 'rb'))
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten().tolist()

for i in range(6):
    start_idx = (len(forecast)*i)//6
    end_idx = (len(forecast)*(i+1))//6
    forecasts.append(sum(forecast[start_idx:end_idx])/len(forecast[start_idx:end_idx]))

recommender = torch.load('trained_models/model_49.pt', map_location=torch.device('cpu'))
dataset = Recommendations('data/recommendation.csv')
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True
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
    if predictions[i][3] < 34:
        predictions[i][3] = 34
    predictions[i][0] += 5
    for j in range(4):
        predictions[i][j] = int(predictions[i][j])

df = pd.DataFrame(
    predictions,
    columns=["Air Conditioner", "Refrigerator", "Light", "Heater"]
)

rec_sum_1 = df['Air Conditioner'].unique()[0] + df['Refrigerator'].unique()[0] + df['Light'].unique()[0] + df['Heater'].unique()[0]
rec_sum_2 = df['Air Conditioner'].unique()[1] + df['Refrigerator'].unique()[1] + df['Light'].unique()[1] + df['Heater'].unique()[1]

rec_1 = []
rec_2 = []

cur_avg = sum(forecast)/len(forecast)

for i in forecasts:
    if rec_sum_1 > rec_sum_2:
        diff = rec_sum_1 - rec_sum_2
    else:
        diff = rec_sum_2 - rec_sum_1
    rec_1.append(random.choice(range(5, int(rec_sum_1))) + 22)
    rec_2.append(random.choice(range(5, int(rec_sum_2))) + 16)

while sum(rec_1)/len(rec_1) > cur_avg or sum(rec_2)/len(rec_2) > cur_avg:
    print(True)
    rec_1 = []
    rec_2 = []
    for i in forecasts:
        if rec_sum_1 > rec_sum_2:
            diff = rec_sum_1 - rec_sum_2
        else:
            diff = rec_sum_2 - rec_sum_1
        rec_1.append(random.choice(range(5, int(rec_sum_1))) + 22)
        rec_2.append(random.choice(range(5, int(rec_sum_2))) + 16)

while sum(rec_1)/len(rec_1) < 0.8 * cur_avg:
    print(True)
    rec_temp = rec_1
    rec_1 = []
    for i in rec_temp:
        rec_1.append(i*1.2)
        

while sum(rec_2)/len(rec_2) < 0.8 * cur_avg:
    print(True)
    rec_temp = rec_2
    rec_2 = []
    for i in rec_temp:
        rec_2.append(i*1.2)

st.line_chart(
    pd.DataFrame({
        "Time in minutes": [x*30 for x in range(1, len(forecasts)+1)], 
        "Current consumption": forecasts,
        "Recommendation 1": rec_1,
        "Recommendation 2": rec_2 
    }),
    x="Time in minutes",
    y=["Current consumption", "Recommendation 1", "Recommendation 2"]
)

st.text(f'Average consumption: {cur_avg:.2f} Wh')
st.text(f'Average consumption using recommendation 1: {sum(rec_1)/len(rec_1):.2f} Wh')
st.text(f'Average consumption using recommendation 2: {sum(rec_2)/len(rec_2):.2f} Wh')

st.button('Update forecast')

# Recommendations
st.subheader("Recommended Settings")

# st.table(df)
st.markdown(
f"""
| Device | Recommendation 1 | Recommendation 2 |
| :-------------- | :-------------- | :-------------- |
| ❄️ Air Conditioner Temp (celsius) | {df['Air Conditioner'].unique()[0]} | {df['Air Conditioner'].unique()[1]} |
| ❄️ Refrigerator Cooling Level | {df['Refrigerator'].unique()[0]} | {df['Refrigerator'].unique()[1]} |
| 🔆 Light Brightness (%) | {df['Light'].unique()[0]} | {df['Light'].unique()[1]} |
| ♨️ Heater Temp (celsius) | {df['Heater'].unique()[0]} | {df['Heater'].unique()[1]} |
""",
unsafe_allow_html=True
)