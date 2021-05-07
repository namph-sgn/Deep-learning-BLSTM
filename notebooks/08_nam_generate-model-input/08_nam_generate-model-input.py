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


sys.path


# ### Imports
# Import libraries and write settings here.

# Data manipulation
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
import glob
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
import holidays

# Extract Data
get_ipython().run_line_magic('aimport', 'features.extract_features')
from features import extract_features
get_ipython().run_line_magic('aimport', 'data.create_load_transform_processed_data')
from data import create_load_transform_processed_data
# Make dataset

# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30
# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from IPython import get_ipython
ipython = get_ipython()
# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
# Use %aimport module to reload each module

# Visualizations
import matplotlib.pyplot as plt


# # Analysis/Modeling
# Do work here

# ### Get interim data

_interim_data_path = os.path.join(PROJ_ROOT,
                                  "data",
                                  "interim")
_interim_files = glob.glob(_interim_data_path + '/*.csv')

interim_df = pd.DataFrame()
for file in _interim_files:
    print('Currently processing file \n{}'.format(file))
    interim_df = interim_df.append(pd.read_csv(file, parse_dates=True, index_col=['site_id', 'time'],
                                                     dtype={'CO': np.float64, 'NO2': np.float64, 'PM25': np.float64,
                                                            'AQI_h': np.float64, 'AQI_h_I': np.int, 'site_id': np.int}))
# Site 16 have many inconsistency in data so we remove it
interim_df = interim_df[(interim_df.index.get_level_values(0) != 16)]
# Get only columns we need
interim_df = interim_df[['PM25', 'AQI_h', 'AQI_h_Polutant', 'AQI_h_I',
       'AQI_h_label', 'Continous length']]
# Ho Chi Minh data is on site 49
hanoi_df = interim_df[(interim_df.index.get_level_values(0) != 49)].copy()
hcm_df = interim_df[(interim_df.index.get_level_values(0) == 49)].copy()


# ### Create input for model from interim data and put in to processed

_processed_data_path = os.path.join(PROJ_ROOT,
                                   "data",
                                   "processed")

def create_input_for_model(df, timesteps=[1], target_hour=[1], output_path=None):
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
                                   "processed")
    for timesteps in timesteps:
        for target_hour in target_hour:
            # Create train, dev, test data
            final_df = extract_features.add_features(df).copy()
            train_df, test_df = extract_features.generate_train_test_set_by_time(final_df)
            train_df, dev_df = extract_features.generate_train_test_set_by_time(train_df)
            train, y_train, multiclass_y = extract_features.data_preprocessing(train_df, target_hour, timesteps=timesteps)
            test, y_test, multiclass_y_test = extract_features.data_preprocessing(test_df, target_hour, timesteps=timesteps)
            dev, y_dev, multiclass_y_dev = extract_features.data_preprocessing(dev_df, target_hour, timesteps=timesteps)

            # Save data to file
            create_model_input.reshape_array_and_save_to_path(train, y_train, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="train")
            create_model_input.reshape_array_and_save_to_path(dev, y_dev, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="dev")
            create_model_input.reshape_array_and_save_to_path(test, y_test, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="test")
    print("Input have been created")
create_input_for_model(hanoi_df, output_path=_processed_data_path+"/hanoi")
create_input_for_model(hcm_df, output_path=_processed_data_path+"/hcm")


# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here
