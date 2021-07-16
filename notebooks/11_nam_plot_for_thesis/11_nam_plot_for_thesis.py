#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook purpose is to plot everything needed for the thesis

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

# What we need right now is:
# Total system processes
# Data preparation processes
# Raw data description table
# Enhanced data table
# Result for each time steps table
# result of each hour model table
# model creation diagram
# model training process
# model deployment process
# Web app architecture

# Raw data line plot overview
# Dickey fuller test
# KDE plot or line plot for distribution of air level
# Plot seasonal decompose
# Plot PACF for most affect time
# Plot real data vs result for 5 hours
# Plot bar chart for the result of 12 models for each time window
# Plot bar chart for the result of each hour model (RMSE and MAE)

from data 


# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here



