from datetime import time
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import altair as alt
import os

# Import custom models
from src.data.create_load_transform_processed_data import load_reshaped_array, create_tensorflow_dataset
from src.models import load_model

st.write("""
# Air quality prediction app

This app predicts air quality in a day in Ho Chi Minh City

Data obtained from (https://moitruongthudo.vn/) and AirNow API (https://www.airnow.gov/).
These data are **not fully verified or validated** and should be considered preliminary and subject to change. 
Data and information reported to AirNow are for the express purpose of reporting and forecasting the AQI. 
As such, they should not be used to formulate or support regulation, trends, guidance, or any other government or public decision making.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# First we must load the models
# Then we take latest data for models
# Then we must predict with the models for latest data
# Then we plot them to 1 line_chart with 2 lines.
# Then we plot the predicted AQI for the next 5 hours on a bar chart
PROJ_ROOT = os.path.abspath('./')

# Data path example
model_input_data_path = os.path.join(PROJ_ROOT,
                             "data",
                             "model_input",
                             "hanoi")

models, model_path = load_model.load_combined_model(timesteps=5, hour=1, PROJ_ROOT=PROJ_ROOT)

data, steps = load_reshaped_array(timesteps=5, target_hour=1, folder_path=model_input_data_path, data_type='test')

if len(data) % 700 != 0:
    remain_count = len(data)%700
    test = data[remain_count:]
    y_test = data[remain_count:]
test_data_tf, test_steps_per_epochs = create_tensorflow_dataset(test, y_test, 700)

predict_data = models.predict(test_data_tf, steps=test_steps_per_epochs)
st.write(predict_data)

st.line_chart(predict_data, width=100, use_container_width=True)

# uploaded_file = st.sidebar.file_uploader(
#     "Upload your input CSV file", type=['csv'])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         island = st.sidebar.selectbox(
#             'Island', ('Biscoe', 'Dream', 'Torgesen'))
#         sex = st.sidebar.selectbox('Sex', ('male', 'female'))
#         bill_length_mm = st.sidebar.slider(
#             'Bill length (mm)', 32.1, 59.6, 43.9)
#         bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
#         flipper_length_mm = st.sidebar.slider(
#             'Flipper length (mm)', 172.0, 231.0, 201.0)
#         body_mass_g = st.sidebar.slider(
#             'Body mass (g)', 2700.0, 6300.0, 4207.0)
#         data = {
#             'island': island,
#             'bill_depth_mm': bill_depth_mm,
#             'bill_length_mm': bill_length_mm,
#             'flipper_length_mm': flipper_length_mm,
#             'body_mass_g': body_mass_g,
#             'sex': sex
#         }
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_df = user_input_features()

# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# X = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df, X], axis=0)

# encode = ['sex', 'island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df, dummy], axis=1)
#     del df[col]
# df = df[:1] # Select only user input

# st.subheader('User input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write("Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).")
#     st.write(df)

# st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# st.write(pd.DataFrame(prediction_proba, columns=['Adelie','Chinstrap','Gentoo']))