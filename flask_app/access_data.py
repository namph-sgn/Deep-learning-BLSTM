import feedparser
import pandas as pd
import numpy as np
from google.cloud import storage
from io import StringIO


def get_new_data():
    feed = "http://dosairnowdata.org/dos/RSS/HoChiMinhCity/HoChiMinhCity-PM2.5.xml"
    NewsFeed = feedparser.parse(feed)
    train = pd.DataFrame.from_dict(NewsFeed, orient='index')
    train2 = pd.DataFrame.from_dict(train.loc['entries', :].values[0])
    train2 = train2[['title', 'aqi']]
    train2.rename(columns={'title': 'time', 'aqi': 'AQI_h'}, inplace=True)
    train2 = train2.astype({'time': 'datetime64[ns]', 'AQI_h': 'float'})
    train2['site_id'] = 49
    train2.set_index(['site_id', 'time'], inplace=True)
    train2['AQI_h_label'] = categorize_AQI(train2['AQI_h'])
    train2['AQI_h_I'] = train2['AQI_h_label'].cat.codes + 1
    train2['Continous length'] = 0
    return train2


def get_past_data_from_bucket_as_dataframe():
    """Read a blob"""
    bucket_name = "deep_learning_model_bucket"
    blob_name = "past_data.csv"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists() == False:
        blob.upload_from_filename('new_data.csv')

    return_data = blob.download_as_text()
    return_data = StringIO(return_data)
    df = pd.read_csv(return_data, sep=",", header=0, index_col=False)
    df = df.astype({'time': 'datetime64[ns]', 'AQI_h': 'float'})
    df.set_index(['site_id', 'time'], inplace=True)
    return df


def concat_past_and_new_data():
    idx = pd.IndexSlice
    past_data = get_past_data_from_bucket_as_dataframe()
    new_data = get_new_data()
    max_time_past = past_data.index.get_level_values(
        1).max() + pd.Timedelta(hours=1)
    max_time_new = new_data.index.get_level_values(1).max()
    past_data = pd.concat(
        [past_data, new_data.loc[idx[:, max_time_past:max_time_new], :]])
    return past_data


def delete_past_data_from_bucket():
    bucket_name = "deep_learning_model_bucket"
    blob_name = "past_data.csv"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        return_data = blob.delete()
    return "Deleted"


def create_new_file_in_bucket(filename=None):
    bucket_name = "deep_learning_model_bucket"
    blob_name = "past_data.csv"
    if filename is None:
        filename = 'test.csv'

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return_data = blob.upload_from_filename(filename)
    print(return_data)
    return "Created"


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
