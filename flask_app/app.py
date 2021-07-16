from pprint import pprint as pp
import requests
from flask import Flask, flash, redirect, url_for, Response, request
import os
import pandas as pd
import numpy as np
from query_from_models import predict_json, create_input_for_model
from access_gcp_data import concat_past_and_new_data, concat_past_and_new_prediction, delete_past_data_from_bucket, create_new_file_in_bucket, get_data_from_bucket_as_dataframe
import json

app = Flask(__name__)

# change for your GCP key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "flask-app-test-317210-bdec872c665d.json"
PROJECT = "flask-app-test-317210"  # change for your GCP project
# change for your GCP region (where your model is hosted)
REGION = "us-central1"
MODEL = ['hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5']


# This predict function need to: get past_data.csv
# transform past_data to model input
# Call 5 models to predict
# Get data from 48 hours then create input from those data then take latest 5 hours to send to the models.
# Concat all 5 prediction to a list
# After predict, Save a csv file called current_prediction.csv save 5 hours only. This is overwritten any time the predict is called.
# Then save the first prediction to a file called past_prediction.csv. This file contains all prediction made from the time the program run.
# return the prediction list

@app.route('/predict_five', methods=['GET'])
def predict_five():
    """
    Run prediction with new data, run every one hour
    """
    data_df = get_data_from_bucket_as_dataframe()
    data_df = data_df.tail(25)
    data_df = data_df.astype({'time': 'datetime64[ns]', 'AQI_h': 'float'})
    data_df.set_index(['site_id', 'time'], inplace=True)
    all_result = np.array([])
    predict_data, label = create_input_for_model(
        data_df, timesteps=[5], target_hour=[1])
    tmp_predict_data = predict_data[-1].copy()
    tmp_predict_data = np.reshape(
        tmp_predict_data, (1, predict_data.shape[1], predict_data.shape[2]))
    for target_hour in range(0, 5):
        preds = predict_json(project=PROJECT,
                             region=REGION,
                             model=MODEL[target_hour],
                             instances=tmp_predict_data)
        all_result = np.append(all_result, preds)

    daterange = pd.date_range(
        start=data_df.iloc[-1].name[1], end=data_df.iloc[-1].name[1] + pd.Timedelta(hours=4), freq='H', name="time")
    all_result_df = pd.DataFrame(
        all_result, index=daterange, columns=['AQI_h'])
    all_result_df.to_csv('current_prediction.csv')
    delete_past_data_from_bucket(delete_file_name="current_prediction.csv")
    create_new_file_in_bucket(upload_file='current_prediction.csv')

    prediction_file = concat_past_and_new_prediction(all_result_df.tail(1))
    prediction_file.to_csv('past_prediction.csv')
    delete_past_data_from_bucket(delete_file_name="past_prediction.csv")
    create_new_file_in_bucket(upload_file='past_prediction.csv')
    return Response(json.dumps(list(all_result)),  mimetype='application/json')


# This get predict result need:
# data of past month label and past month prediction
# So first, get past_prediction.csv and get 30 latest spot
# Then get past_data.csv and get 30 latest spot.
# Get current_prediction.csv
# Return all those prediction


@app.route('/get_predict_result', methods=['GET'])
def get_predict_result():
    past_prediction = get_data_from_bucket_as_dataframe(
        filename="past_prediction.csv")
    past_prediction = past_prediction.astype({'AQI_h': 'float'})
    past_prediction = past_prediction.tail(30)
    past_real_data = get_data_from_bucket_as_dataframe(
        filename="past_data.csv")
    past_real_data = past_real_data.astype({'AQI_h': 'float'})
    past_real_data = past_real_data.tail(30)
    current_prediction = get_data_from_bucket_as_dataframe(
        filename="current_prediction.csv")
    json_dict = {
                'past_prediction_time': list(past_prediction['time'].values),
                'past_prediction': list(past_prediction['AQI_h'].values),
                 'past_real_data': list(past_real_data['AQI_h'].values),
                 'current_prediction_time': list(current_prediction['time'].values),
                 'current_prediction': list(current_prediction['AQI_h'].values)}
    return Response(json.dumps(json_dict),  mimetype='application/json')


@app.route('/update', methods=['GET'])
def update():
    """update Update past_data.csv file in bucket to store new data gotten from AQINow API, this will be called every one hour

    Returns:
        None
    """
    if request.method == 'GET':
        data = concat_past_and_new_data()
        data.to_csv('past_data.csv')
        delete_past_data_from_bucket()
        create_new_file_in_bucket(upload_file='past_data.csv')
    return Response(json.dumps(json.loads(data.to_json())),  mimetype='application/json')


@app.route('/')
def hello():
    return 'Hello World!'


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
