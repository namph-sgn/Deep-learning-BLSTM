from datetime import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Make function to get data from api web
import requests

# change for your GCP key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "flask-app-test-317210-bdec872c665d.json"
PROJECT = "flask-app-test-317210"  # change for your GCP project
# change for your GCP region (where your model is hosted)
REGION = "us-central1"


_predict_data_url = "https://flask-app-test-317210.de.r.appspot.com/get_predict_result"

r = requests.get(url=_predict_data_url)
web_data = r.json()
past_data = pd.DataFrame({'time': web_data['past_prediction_time'],
                         'AQI_h_predict': web_data['past_prediction'], 'AQI_h': web_data['past_real_data']})
past_data = past_data.astype(
    {'time': 'datetime64[ns]', 'AQI_h': 'float', 'AQI_h_predict': 'float'})
current_data = pd.DataFrame(
    {'time': web_data['current_prediction_time'], 'AQI_h': web_data['current_prediction']})
current_data = current_data.astype({'AQI_h': 'float'})

st.write("""
# Air quality prediction app

This app predicts air quality in a day in Ho Chi Minh City

Data obtained from (https://moitruongthudo.vn/) and AirNow API (https://www.airnow.gov/).
These data are **not fully verified or validated** and should be considered preliminary and subject to change. 
Data and information reported to AirNow are for the express purpose of reporting and forecasting the AQI. 
As such, they should not be used to formulate or support regulation, trends, guidance, or any other government or public decision making.
""")

# st.sidebar.header('User Input Features')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)

st.write(current_data)

fig_bar = plt.figure(figsize=(14, 7))
ax_bar = fig_bar.add_subplot()


ax_bar.bar(current_data['time'].values,
           current_data['AQI_h'].values, width=0.4)

ax_bar.set_title("Next 5 hours prediction", fontsize=18)

st.write(fig_bar)

fig_line = plt.figure(figsize=(14, 7))
ax_line = fig_line.add_subplot()

ax_line.plot(past_data['time'].values, past_data['AQI_h'].values, label='Actual')
ax_line.plot(past_data['time'].values, past_data['AQI_h_predict'].values, label='Prediction')

ax_line.set_title("Past 30 hours prediction vs actual", fontsize=18)
fig_line.legend()

st.write(fig_line)