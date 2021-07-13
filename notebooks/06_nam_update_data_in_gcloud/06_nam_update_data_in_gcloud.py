#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Get source folder and append to sys directory

from __future__ import print_function
import os
import sys
PROJ_ROOT = os.path.join(os.pardir)
print(os.path.abspath(PROJ_ROOT))
src_dir = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_dir)
# Data path example
#pump_data_path = os.path.join(PROJ_ROOT,
#                              "data",
#                              "raw",
#                              "pumps_train_values.csv")


# ### Imports
# Import libraries and write settings here.

# Data manipulation
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import feedparser
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(PROJ_ROOT, 'flask_app',
                                                            "flask-app-test-317210-0c49e7d7d9cb.json")

# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30
# Display all cell outputs
InteractiveShell.ast_node_interactivity = 'all'
ipython = get_ipython()
# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
# Use %aimport module to reload each module

# Visualizations


# # Analysis/Modeling
# Do work here

def get_data_for_prediction():
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
    train, y_train = create_input_for_model(
        train2, timesteps=[5], target_hour=[1])
    print(train.shape)
    return train2, y_train

def get_past_data_from_bucket_as_dataframe():
    """Read a blob"""
    bucket_name = "deep_learning_model_bucket"
    blob_name = "past_data.csv"
    
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return_data = blob.download_as_text()
    return_data = StringIO(return_data)
    df = pd.read_csv(return_data, sep=",", header=0, index_col=False)

    return df


data_df, data = get_data_for_prediction()
past_data = get_past_data_from_bucket_as_dataframe()


# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here



