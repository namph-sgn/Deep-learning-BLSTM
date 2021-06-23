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

import feedparser
feed = "http://dosairnowdata.org/dos/RSS/HoChiMinhCity/HoChiMinhCity-PM2.5.xml"
NewsFeed = feedparser.parse(feed)


get_ipython().run_line_magic('aimport', 'features.calculate_AQI')
from features.calculate_AQI import categorize_AQI
get_ipython().run_line_magic('aimport', 'features.extract_features')
from features.extract_features import data_preprocessing
get_ipython().run_line_magic('aimport', 'data.create_input_for_models')
from data.create_input_for_models import create_input_for_model
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
create_input_for_model(train2, timesteps=[5], target_hour=[1], output_path=os.path.join(PROJ_ROOT, "data", "model_input"))

# After calculating AQI, we must:
# Process data: Add site_id put to 0-1, encode to tensorflow,
# Put that data into model


# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here



