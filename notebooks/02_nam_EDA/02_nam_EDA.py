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

# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here



