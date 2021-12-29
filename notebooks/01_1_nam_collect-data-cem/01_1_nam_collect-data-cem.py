#!/usr/bin/env python
# coding: utf-8

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


# Data manipulation
import pandas as pd
import numpy as np
import sklearn
import asyncio
import requests
import re

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


# import requests
# _site_URL = "http://envisoft.cem.gov.vn/eos/services/call/json/get_stations"

# r = requests.post(url=_site_URL)
# site_data = r.json()
# site_data = pd.json_normalize(site_data['stations'])
# site_data = site_data[['off_time3', "station_name", "address", "id", 'latitude', 'longitude']]
# site_data.rename(columns={'id': 'dataid', 'longitude': 'longtitude'}, inplace=True)
# site_data['longtitude'] = site_data['longtitude'].round(4)
# site_data['latitude'] = site_data['latitude'].round(4)
# site_data = site_data[site_data['station_name'].str.contains("Hà Nội:")]
# # site_data
# site_data_path = os.path.join(PROJ_ROOT,
#                               "data",
#                               "raw",
#                               "site_data",
#                               "site_data_cem.csv")
# # site_data[['dataid', 'station_name', 'off_time3', 'address', 'latitude', 'longtitude']].to_csv(site_data_path, quoting=, index=False)


site_data_path = os.path.join(PROJ_ROOT,
                              "data",
                              "raw",
                              "site_data",
                              "site_data_cem.csv")
site_data = pd.read_csv(site_data_path, dtype={"dataid":"string"})
site_data


a = get_columns(headers=headers, dataid="29573032472660669416148855365")
a


def get_raw_aqi_data(headers, dataid, id, columns):
    url = "http://enviinfo.cem.gov.vn/eip/default/call/json/get_aqi_data%3Faqi_type%3D1"
    
    payload = "sEcho=1&iColumns=7&sColumns=%2C%2C%2C%2C%2C%2C&iDisplayStart=0&iDisplayLength=22814&mDataProp_0=0&sSearch_0=&bRegex_0=false&bSearchable_0=true&mDataProp_1=1&sSearch_1=&bRegex_1=false&bSearchable_1=true&mDataProp_2=2&sSearch_2=&bRegex_2=false&bSearchable_2=true&mDataProp_3=3&sSearch_3=&bRegex_3=false&bSearchable_3=true&mDataProp_4=4&sSearch_4=&bRegex_4=false&bSearchable_4=true&mDataProp_5=5&sSearch_5=&bRegex_5=false&bSearchable_5=true&mDataProp_6=6&sSearch_6=&bRegex_6=false&bSearchable_6=true&sSearch=&bRegex=false&station_id={}&added_columns=PM-10%2CPM-2-5".format(dataid)
    df = pd.DataFrame()
    try:
        columns[0:0] = [0, 'time', 'AQI']
        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()
        data = data['aaData']
        df = pd.DataFrame(data, columns=columns)
        df.drop([0], axis=1, inplace=True)
        df['site_id'] = id
        column_order = ['site_id', 'time', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'AQI']
        df = df.reindex(column_order, axis='columns', fill_value=0)
        df = df[column_order]
        df = df.iloc[::-1]
    except ValueError:
        print("error in site {}".format(id))
        print("columns: {}".format(columns))
    return df

def get_columns(headers, dataid):
    url = "http://enviinfo.cem.gov.vn/eip/default/call/json/get_indicators_have_data"
    payload = "station_id={}&from_public=1&station_type=4".format(dataid)
    response = requests.request("POST", url, headers=headers, data=payload)

    response = response.json()

    response = response['html']

    response = response.replace("PM-2-5", "PM2.5")
    response = response.replace("PM-10", "PM10")
    x = re.findall("selected>(.*?)</option>", response)
    return x

headers = {
  'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
  'Accept': 'application/json, text/javascript, */*; q=0.01',
  'Accept-Language': 'en-US,en;q=0.5',
  'Referer': 'http://enviinfo.cem.gov.vn/',
  'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
  'X-Requested-With': 'XMLHttpRequest',
  'Origin': 'http://enviinfo.cem.gov.vn',
  'Connection': 'keep-alive',
  'Cookie': 'session_id_eip=125.235.239.3-7aa0263d-aede-4574-94e8-19e09900c1aa',
  'Pragma': 'no-cache',
  'Cache-Control': 'no-cache'
}

# get_raw_aqi_data(site_data['dataid'][1])
for id, dataid in zip(site_data['id'], site_data['dataid']):
    # dataid = data['dataid']
    print("{} --- {}".format(id,dataid))
    columns = get_columns(headers=headers, dataid=dataid)
    response_data = get_raw_aqi_data(dataid=str(dataid), id=id, headers=headers, columns=columns)
    if not response_data.empty:
        cem_site_path = os.path.join(PROJ_ROOT,
                                "data",
                                "cem",
                                "{}.csv".format(id))
        response_data.to_csv(cem_site_path, index=False)

