import pandas as pd
import os
# import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import feedparser
import numpy as np
import extract_features
import create_load_transform_processed_data
from io import StringIO
from google.cloud import storage
import joblib

# def create_tensorflow_dataset(arr_data, arr_label, batch_size):
#     if len(arr_data) % batch_size != 0:
#         if len(arr_data) // batch_size != 0:
#             remain_count = len(arr_data) % batch_size
#             arr_data = arr_data[remain_count:]
#             arr_label = arr_label[remain_count:]
#         else:
#             batch_size = len(arr_data)
#     tf_dataset = tf.data.Dataset.from_tensor_slices((arr_data, arr_label))
#     tf_dataset = tf_dataset.repeat().batch(batch_size, drop_remainder=True)
#     steps_per_epochs = len(arr_data) // batch_size
#     print(arr_data)
#     return tf_dataset, steps_per_epochs


def categorize_AQI(AQI_data):
    """
    Input: Series of AQI_values
    Output: Series of AQI category
    7 categories [Good, Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy, Hazardous, Out of AQI]
    range of categories [0-50, 51-100, 101-150, 151-200, 201-300, 301-500, >500]
    """
    bins = [-1, 50, 100, 150, 200, 300, 500, np.inf]
    labels = ["Good", "Moderate", "Unhealthy for Sensitive",
              "Unhealthy", "Very Unhealthy", "Hazardous", "Beyond AQI"]
    return pd.cut(AQI_data, bins=bins, labels=labels)


def create_input_for_model(df, timesteps=[1], target_hour=[1], test_output=False, dev_output=False, output_path=None, PROJ_ROOT=os.pardir):
    """From interim dataframe:
        + add features
        + split into chunks according to timesteps
        + compressed and saved to output_path
        + estimate number of created dataset = timesteps * target_hour
    Parameters
    ----------
    df : pandas.DataFrame
        Contains interim data.
    timesteps : list of integer
        Each timestep represent 1 dataset
    target_hour : list of integer
        the label for each timesteps
    output_path : string
        Destination directory the dataset will be created
    """
    if output_path == None:
        output_path == os.path.join(PROJ_ROOT,
                                    "data",
                                    "model_input")
    for timesteps in timesteps:
        for target_hour in target_hour:
            # Create train, dev, test data
            # train_df = extract_features.create_and_save_scale_data(df, output_path=output_path).copy()
            # data scaled must be from the google cloud platform
            scaler = extract_features.load_scaler()
            train_df = df.copy()
            for col in ['AQI_h']:
                train_df[[col]] = scaler.transform(train_df[[col]])
            train_df = extract_features.add_features(train_df)
            if test_output is not False:
                train_df, test_df = extract_features.generate_train_test_set_by_time(
                    train_df)
                test, y_test, multiclass_y_test = extract_features.data_preprocessing(
                    test_df, target_hour, timesteps=timesteps)
                create_load_transform_processed_data.reshape_array_and_save_to_path(
                    test, y_test, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="test")
            if dev_output is not False:
                train_df, dev_df = extract_features.generate_train_test_set_by_time(
                    train_df)
                dev, y_dev, multiclass_y_dev = extract_features.data_preprocessing(
                    dev_df, target_hour, timesteps=timesteps)
                create_load_transform_processed_data.reshape_array_and_save_to_path(
                    dev, y_dev, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="dev")

            train, y_train, multiclass_y = extract_features.data_preprocessing(
                train_df, target_hour, timesteps=timesteps)

            # Save data to file
            if output_path is not None:
                create_load_transform_processed_data.reshape_array_and_save_to_path(
                    train, y_train, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="train")
    train = train.astype('float32')
    y_train = y_train.astype('float32')
    print("Input have been created")
    return train, y_train


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    # turn input into list (ML Engine wants JSON)
    instances_list = instances.tolist()

    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list}

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()

    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    scaler = extract_features.load_scaler()
    reverse_scaled_prediction = scaler.inverse_transform(response['predictions'])

    return reverse_scaled_prediction
