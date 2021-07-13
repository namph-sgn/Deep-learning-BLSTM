#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook is for collecting raw data and updating interim data with the previous

# ### Get source folder and append to sys directory

from __future__ import print_function
import os
import sys
PROJ_ROOT = os.path.join(os.pardir)
print(os.path.abspath(PROJ_ROOT))
src_dir = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_dir)
# Data path example
# pump_data_path = os.path.join(PROJ_ROOT,
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
import asyncio
import requests

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

import requests
import pandas as pd
_site_URL = "https://moitruongthudo.vn/api/site"

r = requests.get(url=_site_URL)
site_data = r.json()
site_data = pd.DataFrame(site_data)
site_data_path = os.path.join(PROJ_ROOT,
                              "data",
                              "raw",
                              "site_data",
                              "site_data.csv")
site_data[['id', 'name', 'address', 'latitude', 'longtitude',
           'ref_id']].to_csv(site_data_path, index=False)


import feedparser
feed = "http://dosairnowdata.org/dos/RSS/HoChiMinhCity/HoChiMinhCity-PM2.5.xml"
NewsFeed = feedparser.parse(feed)
print(NewsFeed)


API_KEY = "0E7F0D6A-3C0D-4A7C-BEFB-5CB7438FA91B"
request_link = "https://www.airnowapi.org/aq/observation/latLong/historical/?format=application/json&latitude=10.7115&longitude=106.6353&date={}&distance=25&API_KEY={}"

async def collect_data(date):
    global API_KEY
    global request_link
    global data
    link = request_link.format(date, API_KEY)
    r = requests.get(url=link)
    tmp_data = r.json()
    print(tmp_data)
    tmp_data = pd.DataFrame(tmp_data)
    # data = pd.concat(data, tmp_data)
    return tmp_data
lala = await collect_data('2021062001')
lala


API_KEY = "0E7F0D6A-3C0D-4A7C-BEFB-5CB7438FA91B"
request_link = "https://www.airnowapi.org/aq/observation/latLong/historical/?format=application/json&latitude=10.7115&longitude=106.6353&date={}&hour={}&distance=25&API_KEY={}"

data = pd.DataFrame()

def process_usembassy_data():
    # What we will do in process data:
    # Check for
    return "OK"

async def collect_data(date):
    global API_KEY
    global request_link
    global data
    link = request_link.format(date, API_KEY)
    r = requests.get(url=link)
    tmp_data = r.json()
    print(tmp_data)
    tmp_data = pd.DataFrame(tmp_data)
    # data = pd.concat(data, tmp_data)
    return tmp_data

async def get_data_usembassy():
    time_now = pd.Timestamp.now()
    daterange = pd.date_range(end=time_now-pd.Timedelta(hours=1),start=time_now-pd.Timedelta(hours=12), freq='H').round('H')
    daterange = daterange.strftime('%Y-%m-%dT%H-0000')
    daterange = pd.DataFrame(daterange)
    collecting_task = list(daterange.apply(lambda date: collect_data(date=date), axis=1))
    all_done = await asyncio.gather(*collecting_task)
    return None

await get_data_usembassy()


data


# ### Get raw data from moitruongthudo.vn

import datetime
import csv
import asyncio
import requests
import pandas as pd
import time
idx = pd.IndexSlice

tic = time.time()
_stat_URL = "https://moitruongthudo.vn/public/dailystat/"
_site_id = pd.DataFrame(site_data['id'])
changed_raw_file = []

async def get_indv_data(parameter, site_id):
    r = requests.get(url = _stat_URL + parameter + '/', params = {'site_id': site_id})
    data = r.json()
    data = pd.DataFrame(data)
    data['parameter'] = parameter
    return data

async def data_processing(all_data, site_id, latest_time, data_good=True):
    all_data = pd.concat(all_data)
    all_data['time'] = pd.to_datetime(all_data['time'], format="%Y-%m-%d %H:%M:%S")
    # all_data = all_data[all_data['time'] > latest_time]
    all_data['site_id'] = site_id
    all_data = pd.pivot_table(all_data, values = 'value', index = ['site_id', 'time'], columns=['parameter'],
            aggfunc='sum')
    all_data.rename({'PM2.5': 'PM25'}, axis=1, inplace=True)
    all_data.reset_index(inplace=True)
    return all_data

async def get_site_data(site_id, latest_time):
    parameters = ['NO2','SO2','CO','PM2.5','PM10','O3']
    site_data = pd.DataFrame()
    all_data = await asyncio.gather(*(get_indv_data(p, site_id) for p in parameters))
    site_data = await data_processing(all_data, site_id, latest_time)
    return site_data

async def update_raw_files(site_id):
    csv_path_name = os.path.join(PROJ_ROOT,
                                "data",
                                "raw",
                                "{}.csv".format(site_id))
    try:
        exist_data = pd.read_csv(csv_path_name)
    except FileNotFoundError: 
        print("No data for site {}".format(site_id))
        exist_data = []
    if len(exist_data) != 0:
        # Convert column to date
        exist_data['time'] = pd.to_datetime(exist_data['time'], format="%Y-%m-%d %H:%M:%S")
        # Find the latest datetime
        latest_time = exist_data['time'].max()
        # Get data for site
        site_data = await get_site_data(site_id, latest_time)
        # Check in exist data from mintime of site_data to latest_time of exist data have 0
        global new_data_min_time 
        new_data_min_time = site_data['time'].min()
        tmp_data = exist_data[exist_data['time']>=new_data_min_time].copy()
        tmp_data = tmp_data[tmp_data.PM25.isin([0])]
        # If there is 0 in old data, delete all data till mintime of new data and write new file
        if tmp_data.shape[0] != 0:
            exist_data = exist_data[exist_data['time']<new_data_min_time]
            new_site_data = pd.concat([exist_data, site_data])
            new_site_data.to_csv(csv_path_name, index=False, mode='w')
            global changed_raw_file
            changed_raw_file.append(site_id)
        else:
            site_data = site_data[site_data['time']>latest_time]
            site_data.to_csv(csv_path_name, header=False, index=False, mode='a')
        print('done: {} site'.format(site_id))
    return None

tasks = list(_site_id.apply(lambda site_id: update_raw_files(site_id.values[0]), axis=1))
all_done = await asyncio.gather(*tasks)
toc = time.time()
print('total time in ms: {}ms'.format(1000 * (toc - tic)))


def fix_file_and_data():
    # Sometime when pulling raw data from internet we have some string mixed in
    # This function search for those string and change them to float
    


# ### Update interim data with newly collected data

# Job: take last time from interim_data
# @aimport features
from features import calculate_AQI
import pandas as pd
import numpy as np
import glob
import xarray as xr

def exclusion(list1, list2):
    return list(set(list1) - set(list2))

_changed_raw_site_id = pd.DataFrame(changed_raw_file)
_site_id = pd.DataFrame(exclusion(_site_id.values.ravel(), changed_raw_file))


async def update_interim_files(site_id, changed_raw_file=False):
    _raw_path_name = os.path.join(PROJ_ROOT,
                                 "data",
                                 "raw",
                                 "{}.csv".format(site_id))
    _interim_path_name = os.path.join(PROJ_ROOT,
                                     "data",
                                     "interim",
                                     "{}.csv".format(site_id))
    try:
        raw_data = pd.read_csv(
            _raw_path_name, parse_dates=True, index_col=['site_id', 'time'])
        interim_data = pd.read_csv(
            _interim_path_name, parse_dates=True, index_col=['site_id', 'time'])
    except FileNotFoundError:
        print("No data for site {}".format(site_id))
        raw_data = []
        interim_data = []
    except:
        print("Error for site {}".format(site_id))
    if len(raw_data) != 0:
        # Find the latest datetime
        raw_latest_time = raw_data.index.get_level_values(1).max()

        if changed_raw_file==False:
            interim_latest_time = interim_data.index.get_level_values(1).max()
        else:
            global new_data_min_time
            interim_latest_time = new_data_min_time
        print("Site {}, min_time {}, interim {}".format(site_id, new_data_min_time, interim_latest_time))
        # Trim raw_data time/ features
        raw_data = raw_data[raw_data.index.get_level_values(
            1) >= (interim_latest_time - pd.Timedelta(hours=12))]
        raw_data = raw_data[['CO', 'NO2', 'PM25']]
        # Calculate AQI
        AQI = calculate_AQI.calculate_AQI_h(raw_data)
        # Trim AQI to interim latest time
        AQI = AQI[AQI.index.get_level_values(1) >= interim_latest_time]
        if changed_raw_file==False:
            AQI.to_csv(_interim_path_name, header=False, mode='a')
        else:
            # Trim interim_data to new latest time
            interim_data = interim_data[interim_data.index.get_level_values(1) < interim_latest_time]
            interim_data.reset_index(inplace=True)
            AQI.reset_index(inplace=True)
            new_AQI = pd.concat([interim_data, AQI])
            new_AQI.to_csv(_interim_path_name, index=False, mode='w')
        print('done: {} site'.format(site_id))
    return None


tasks = list(_site_id.apply(
    lambda site_id: update_interim_files(site_id.values[0]), axis=1))

all_done = await asyncio.gather(*tasks)

changed_file_tasks = list(_changed_raw_site_id.apply(
    lambda site_id: update_interim_files(site_id.values[0], changed_raw_file=True), axis=1))

all_done_changed_task = await asyncio.gather(*changed_file_tasks)

