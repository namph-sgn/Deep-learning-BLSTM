from pprint import pprint as pp
import requests
from flask import Flask, flash, redirect, url_for, Response, request
import os
from query_from_models import predict_json, create_input_for_model
from access_data import concat_past_and_new_data, delete_past_data_from_bucket, create_new_file_in_bucket, get_past_data_from_bucket_as_dataframe
import json

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "flask-app-test-317210-bdec872c665d.json" # change for your GCP key
PROJECT = "flask-app-test-317210" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)
MODEL = 'base_model'

@app.route('/predict', methods=['GET'])
def predict():
    """
    Take new 24 hour data from USEmbassy then make them into tensorflow dataset and run them through predict_json.
    What left is output the prediction
    """
    data_df = get_past_data_from_bucket_as_dataframe()
    print(data_df.shape)
    # data_df = data_df.tail(12)

    predict_data, label = create_input_for_model(data_df, timesteps=[5], target_hour=[1])
    print(predict_data)
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=MODEL,
                         instances=predict_data)
    print(preds)
    return Response(json.dumps(preds),  mimetype='application/json')


@app.route('/update', methods=['GET'])
def update():
    """update Update past_data.csv file in bucket to store new data gotten from AQINow API, this will be called every one hour

    Returns:
        None
    """
    if request.method == 'GET':
        new_data = concat_past_and_new_data()
        new_data.to_csv('new_data.csv')
        delete_past_data_from_bucket()
        create_new_file_in_bucket(filename='new_data.csv')
    return Response(json.dumps(json.loads(new_data.to_json())),  mimetype='application/json')

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)