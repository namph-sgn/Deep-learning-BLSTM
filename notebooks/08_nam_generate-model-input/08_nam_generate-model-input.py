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
from 

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

# Ho Chi Minh data is on site 49
hanoi_df = interim_df[(interim_df.index.get_level_values(0) != 49)].copy()
hcm_df = interim_df[(interim_df.index.get_level_values(0) == 49)].copy()


# ### Create input for model from interim data and put in to processed

print(lalalalal)
_processed_data_path = os.path.join(PROJ_ROOT,
                                   "data",
                                   "processed")

def create_input_for_model(df, timesteps_range=, target_hour=[1], output_path=None):
    """From interim dataframe:
        + add features
        + split into chunks according to timesteps
        + compressed and saved to output_path
    Parameters
    ----------
    df : pandas.DataFrame
        Contains interim data.
    """
    if output_path == None:
        output_path == os.path.join(PROJ_ROOT,
                                   "data",
                                   "processed"
    for timesteps in range(1, 13):
        for target_hour in target_hour:
            # Create train, dev, test data
            final_df = add_features(thudohanoi_df).copy()
            train_df, test_df = generate_train_test_set_by_time(final_df)
            train_df, dev_df = generate_train_test_set_by_time(train_df)
            train, y_train, multiclass_y = data_preprocessing(train_df, target_hour, timesteps=timesteps)
            test, y_test, multiclass_y_test = data_preprocessing(test_df, target_hour, timesteps=timesteps)
            dev, y_dev, multiclass_y_dev = data_preprocessing(dev_df, target_hour, timesteps=timesteps)

            # Save data to file
            reshape_array_and_save_to_path(train, y_train, path=_data_to_model_path, timesteps=timesteps, target_hour=target_hour, data_type="train")
            reshape_array_and_save_to_path(dev, y_dev, path=_data_to_model_path, timesteps=timesteps, target_hour=target_hour, data_type="dev")
            reshape_array_and_save_to_path(test, y_test, path=_data_to_model_path, timesteps=timesteps, target_hour=target_hour, data_type="test")
    print("Input have been created")


# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here
