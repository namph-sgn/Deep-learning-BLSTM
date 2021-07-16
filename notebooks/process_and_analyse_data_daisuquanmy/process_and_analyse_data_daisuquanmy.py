#!/usr/bin/env python
# coding: utf-8

# To push the split for training and developing data faster
# we need to skip the index when the timeline is not continuous. (This is covered by dropping all na values)
# The problem arise when there is not enough data after skipping.
# When that happen, the program skip again, leaving some valuable data behind.
# The best thing that can happen is when we skip the second time, if the timeline is not continuous, it will skip
# to the earliest not continous timeline. If we doesn't have any marker, we must find it manually by looping each index.
# So we need to have a marker. It will be put in the raw data files.

# Put timeskip marker in the raw datafiles. (New columns name Timeskip, value is the first time data appear again after the index)
# If after the skip, the timeline is not continous, find the time skip marker and put the position there.
# The refined file doesn't have data in PM (Missing data -> np.nan), findout why *** Not Done 2018 have some problem
# The split for training and developing data has index out of bound, fix it.
# The split for target have some difficulty:
# we can't shift the data so we have to merge it with splitting training data
# When splitting the data 


# We don't need to remove the data, we can just calculate like normal.
# Then in the finished file, we drop all data which deemed Missing. 
# When putting data to the model, we need to figure out a way to know which data is dropped


from __future__ import print_function
import os
import sys
PROJ_ROOT = os.path.join(os.pardir)
print(os.path.abspath(PROJ_ROOT))
src_dir = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_dir)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import seaborn as sns
# import plotly.express as px
from itertools import product
import warnings
# import statsmodels.api as sm
from features import calculate_AQI
plt.style.use('seaborn-darkgrid')


# TP HCM
import pandas as pd
import numpy as np
import glob
import xarray as xr

idx = pd.IndexSlice

_HCM_us_embass_raw_data_path = os.path.join(PROJ_ROOT,
                             "data",
                             "external",
                             "us_embass",
                             "raw")

_HCM_us_embass_data_files = glob.glob(_HCM_us_embass_raw_data_path + '/HoChiMinhCity_*YTD*.csv')
USEm_HCM_df = pd.DataFrame()
for file in _HCM_daisuquanmy_data_files:
    USEm_HCM_df = USEm_HCM_df.append(pd.read_csv(file, parse_dates=True))

# Remove site year 2015 because inconsistency data
# daisuquanmy_df = daisuquanmy_df[(daisuquanmy_df['Year'] != 2015)]
USEm_HCM_df.drop(columns=['Site', 'Parameter', 'AQI', 'AQI Category','NowCast Conc.', 'Conc. Unit', 'Duration'], inplace=True)
USEm_HCM_df['Date (LT)'] = pd.to_datetime(USEm_HCM_df['Date (LT)'])
USEm_HCM_df = USEm_HCM_df.rename(columns={'Date (LT)': 'time'})
USEm_HCM_df['site_id'] = 49
# Remove duplicate, fill in missing index and set ['Date (LT)'] as index and sort
USEm_HCM_df.drop_duplicates(subset ="time", 
                     keep = "first", inplace = True)
USEm_HCM_df = USEm_HCM_df.sort_values(by=['time'])
USEm_HCM_df = USEm_HCM_df.set_index('time').asfreq('H').sort_index()
USEm_HCM_df.reset_index(drop=False, inplace=True)
USEm_HCM_df = USEm_HCM_df.set_index(['site_id', 'time'])
# For all Raw Conc <= 0 QC Name must be Missing. Raw Conc change to -999
USEm_HCM_df.loc[USEm_HCM_df['Raw Conc.'] <= 0, 'Raw Conc.'] = -999
USEm_HCM_df.loc[USEm_HCM_df['Raw Conc.'] >= 600, 'Raw Conc.'] = -999b
USEm_HCM_df.loc[USEm_HCM_df['Raw Conc.'] <= 0, 'QC Name'] = 'Missing'

USEm_HCM_df = USEm_HCM_df.replace(-999, np.nan)
# Change Raw Conc to PM25
USEm_HCM_df.rename(columns={'Raw Conc.': 'PM25'}, inplace=True)
# Fill missing data
USEm_HCM_df.fillna(method='ffill', limit=2, inplace=True)
USEm_HCM_df.fillna(method='bfill', limit=2, inplace=True)

# For year 2015 only, take data after 2015-12-09 14:00:00
USEm_HCM_df = USEm_HCM_df.loc[USEm_HCM_df.index.get_level_values(1) >= pd.to_datetime("2016-02-05 02:00 PM")]
USEm_HCM_df = USEm_HCM_df[['PM25']]

USEm_HCM_df['PM25'] = USEm_HCM_df['PM25'].interpolate()

USEm_HCM_df


# Ha Noi
import pandas as pd
import numpy as np
import glob
import xarray as xr

idx = pd.IndexSlice

_HaNoi_daisuquanmy_data_path = root_path + r'/Data/dai_su_quan_my'
_HaNoi_daisuquanmy_data_files = glob.glob(_HaNoi_daisuquanmy_data_path + '/raw_data/Hanoi_*YTD*.csv')

USEm_HaNoi = pd.DataFrame()
for file in _HaNoi_daisuquanmy_data_files:
    print('Currently processing file \n{}'.format(file))
    USEm_HaNoi = USEm_HaNoi.append(pd.read_csv(file, parse_dates=True))

# Remove site year 2015 because inconsistency data
# daisuquanmy_df = daisuquanmy_df[(daisuquanmy_df['Year'] != 2015)]
USEm_HaNoi.drop(columns=['Site', 'Parameter', 'AQI', 'AQI Category','NowCast Conc.', 'Conc. Unit', 'Duration'], inplace=True)
USEm_HaNoi['Date (LT)'] = pd.to_datetime(USEm_HaNoi['Date (LT)'])
# Remove duplicate, fill in missing index and set ['Date (LT)'] as index and sort
USEm_HaNoi.drop_duplicates(subset ="Date (LT)", 
                     keep = "first", inplace = True)
USEm_HaNoi = USEm_HaNoi.sort_values(by=['Date (LT)'])
USEm_HaNoi = USEm_HaNoi.set_index('Date (LT)').asfreq('H')
USEm_HaNoi.sort_index(inplace=True)
# For all Raw Conc <= 0 QC Name must be Missing. Raw Conc change to -999
USEm_HaNoi.loc[USEm_HaNoi['Raw Conc.'] <= 0, 'Raw Conc.'] = -999
USEm_HaNoi.loc[USEm_HaNoi['Raw Conc.'] <= 0, 'QC Name'] = 'Missing'
USEm_HaNoi = USEm_HaNoi.replace(-999, np.nan)
# Change Raw Conc to PM25
USEm_HaNoi.rename(columns={'Raw Conc.': 'PM25'}, inplace=True)
# Fill missing data
USEm_HaNoi.fillna(method='ffill', limit=2, inplace=True)
USEm_HaNoi.fillna(method='bfill', limit=2, inplace=True)
USEm_HaNoi = USEm_HaNoi.replace(np.nan, 0)

# For year 2015 only, take data after 2015-12-09 14:00:00
USEm_HaNoi = USEm_HaNoi.loc[USEm_HaNoi.index >= pd.to_datetime("2015-12-09 14:00:00")]
USEm_HaNoi = USEm_HaNoi[['PM25']]
USEm_HaNoi


AQI = calculate_AQI.calculate_AQI_h(USEm_HCM_df)


AQI.to_csv(root_path + "/Data/dai_su_quan_my/refined_data/USHCM_refined.csv")


AQI =  pd.read_csv(root_path + "/Data/dai_su_quan_my/refined_data/USHCM_refined.csv")
AQI = AQI.set_index('Date (LT)')
AQI = AQI.iloc[1:]


series = AQI[['PM25']].copy()


sns.kdeplot(series, shade=True)


plt.figure(figsize=(15,12))
plt.suptitle('Lag Plots', fontsize=22)

plt.subplot(3,3,1)
pd.plotting.lag_plot(series, lag=1) #hourly lag
plt.title('1-Hour Lag')

plt.subplot(3,3,2)
pd.plotting.lag_plot(series, lag=24) #daily lag
plt.title('1-Day Lag')

plt.subplot(3,3,3)
pd.plotting.lag_plot(series, lag=168) #Weekly lag
plt.title('Weekly Lag')

plt.subplot(3,3,4)
pd.plotting.lag_plot(series, lag=720) #Monthly lag
plt.title('Monthly Lag')

plt.subplot(3,3,5)
pd.plotting.lag_plot(series, lag=2160) #Quarterly lag
plt.title('Quarterly Lag')

plt.subplot(3,3,6)
pd.plotting.lag_plot(series, lag=8760) #Yearly lag
plt.title('Yearly Lag')

plt.legend()
plt.show()


get_ipython().system(' pip install --upgrade statsmodels')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


plt.figure(figsize=(15,12))
series = AQI_non_indexed
result = seasonal_decompose(series, model='additive', period=1)
result.plot()


acf = plot_acf(series, lags=800, alpha=0.05)
plt.title("ACF for Weighted Price", size=20)
plt.show()


plot_pacf(series, lags=800, alpha=0.05, method='ols')
plt.title("PACF for Weighted Price", size=20)
plt.show()


# Data with lag >26 is statistically insignificant and the impact on model is minimal


stats, p, lags, critical_values = kpss(series, 'ct', nlags='auto')


print(f'Test Statistics : {stats}')
print(f'p-value : {p}')
print(f'Critical Values : {critical_values}')

if p < 0.05:
    print('Series is not Stationary')
else:
    print('Series is Stationary')


def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    
    print (dfoutput)
    
    if p > 0.05:
        print('Series is not Stationary')
    else:
        print('Series is Stationary')


adf_test(series)


# series = series.to_frame()
series.reset_index(drop=False, inplace=True)

window1 = 3
window2 = 24
window3 = 48
feature = 'PM25'

series_rolled_3d = series.rolling(window=window1, min_periods=0)
series_rolled_7d = series.rolling(window=window2, min_periods=0)
series_rolled_30d = series.rolling(window=window2, min_periods=0)

series_mean_3d = series_rolled_3d.mean().shift(1).reset_index()
series_mean_7d = series_rolled_7d.mean().shift(1).reset_index()
series_mean_30d = series_rolled_30d.mean().shift(1).reset_index()

series_std_3d = series_rolled_3d.std().shift(1).reset_index()
series_std_7d = series_rolled_7d.std().shift(1).reset_index()
series_std_30d = series_rolled_30d.std().shift(1).reset_index()

series[f"{feature}_mean_lag{window1}"] = series_mean_3d['PM25']
series[f"{feature}_mean_lag{window2}"] = series_mean_7d['PM25']
series[f"{feature}_mean_lag{window3}"] = series_mean_30d['PM25']

series[f"{feature}_std_lag{window1}"] = series_std_3d['PM25']
series[f"{feature}_std_lag{window2}"] = series_std_7d['PM25']
series[f"{feature}_std_lag{window3}"] = series_std_30d['PM25']

series.fillna(series.mean(), inplace=True)

series.set_index("Date (LT)", drop=False, inplace=True)
series.head()


df = series.copy()


df["month"] = df['Date (LT)'].dt.month


df["month"] = df['Date (LT)'].dt.month
df["hour_of_day"] = df['Date (LT)'].dt.hour
df["day"] = df['Date (LT)'].dt.day
df["day_of_week"] = df['Date (LT)'].dt.dayofweek
df.head()


df.to_csv(root_path + "tmp.csv")


# <a id="subsection-eight"></a>
# # AUTO ARIMA

get_ipython().system(' pip install pmdarima')
import pmdarima as pm


features = ['PM25_mean_lag3', 'PM25_mean_lag24',
       'PM25_mean_lag48', 'PM25_std_lag3', 'PM25_std_lag24', 'PM25_std_lag48',
       'month', 'hour_of_day', 'day', 'day_of_week']


df_train = df[df['Date (LT)'] < "2020"]
df_valid = df[df['Date (LT)'] >= "2020"]


model = pm.auto_arima(df_train.PM25, exogenous=df_train[features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.PM25, exogenous=df_train[features])

forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[features])
df_valid["Forecast_ARIMAX"] = forecast


model_pmdarima = model


df_valid["PM25"].plot(figsize=(14,7))


df_valid["Forecast_ARIMAX"].plot(figsize=(14, 7))


df_valid[["Forecast_ARIMAX", "PM25"]].plot(figsize=(14, 7))


from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

rmse_autoarima = sqrt(mean_squared_error(df_valid['PM25'],df_valid['Forecast_ARIMAX']))
# mape_autoarima = mean_absolute_percentage_error(df_valid['PM25'],df_valid['Forecast_ARIMAX'])
mae_autoarima = mean_absolute_error(df_valid['PM25'],df_valid['Forecast_ARIMAX'])
print("RMSE: {} / MAE: {}".format(rmse_autoarima, mae_autoarima))


# <a id="subsection-eight"></a>
# # CART

from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

from datetime import datetime


tree_reg = tree.DecisionTreeRegressor()


X_train, y_train = df_train[features], df_train.PM25
X_test, y_test = df_valid[features], df_valid.PM25


## Hyper Parameter Optimization Grid

params={
 "max_depth"        : [1, 3, 4, 5, 6, 7],
 "min_samples_split": [1, 3, 4, 5, 6, 7],
 "min_samples_leaf" : [1, 3, 4, 5, 6, 7]
}


model_CART  = RandomizedSearchCV(    
                tree_reg,
                param_distributions=params,
                n_iter=10,
                n_jobs=-1,
                cv=5,
                verbose=3,
)


model_CART.fit(X_train, y_train)


print(f"Model Best Score : {model_CART.best_score_}")
print(f"Model Best Parameters : {model_CART.best_estimator_.get_params()}")


df_train['CART_PM25'] = model_CART.predict(X_train)

df_train[['PM25','CART_PM25']].plot(figsize=(15, 5))


df_valid['CART_PM25'] = model_CART.predict(X_test)

df_valid[['PM25','CART_PM25']].plot(figsize=(16, 9))


df_valid['PM25'].plot(figsize=(16, 9))


df_valid['CART_PM25'].plot(figsize=(16, 9))


from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

rmse_tree = sqrt(mean_squared_error(df_valid['PM25'],df_valid['CART_PM25']))
# mape_autoarima = mean_absolute_percentage_error(df_valid['PM25'],df_valid['Forecast_ARIMAX'])
mae_tree = mean_absolute_error(df_valid['PM25'],df_valid['CART_PM25'])
print("RMSE: {} / MAE: {}".format(rmse_tree, mae_tree))


# <a id="subsection-eight"></a>
# # XG Boost

from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

from datetime import datetime


reg = xgb.XGBRegressor()


X_train, y_train = df_train[features], df_train.PM25
X_test, y_test = df_valid[features], df_valid.PM25


## Hyper Parameter Optimization Grid

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 "max_depth"        : [1, 3, 4, 5, 6, 7],
 "n_estimators"     : [int(x) for x in np.linspace(start=100, stop=2000, num=10)],
 "min_child_weight" : [int(x) for x in np.arange(3, 15, 1)],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
 "subsample"        : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
 "colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
 "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1],  
 
}


model_xgboost  = RandomizedSearchCV(    
                reg,
                param_distributions=params,
                n_iter=10,
                n_jobs=-1,
                cv=5,
                verbose=3,
)


model_xgboost.fit(X_train, y_train)


print(f"Model Best Score : {model_xgboost.best_score_}")
print(f"Model Best Parameters : {model_xgboost.best_estimator_.get_params()}")


model_xgboost.best_estimator_


df_train['Predicted_PM25'] = model_xgboost.predict(X_train)

df_train[['PM25','Predicted_PM25']].plot(figsize=(15, 5))


df_valid['Predicted_PM25'].plot(figsize=(16,9))


df_valid['Predicted_PM25'] = model_xgboost.predict(X_test)
df_valid[['PM25', 'Predicted_PM25']].plot(figsize=(16,9))


from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

rmse_xgboost = sqrt(mean_squared_error(df_valid['PM25'],df_valid['Predicted_PM25']))
# mape_autoarima = mean_absolute_percentage_error(df_valid['PM25'],df_valid['Forecast_ARIMAX'])
mae_xgboost = mean_absolute_error(df_valid['PM25'],df_valid['Predicted_PM25'])
print("RMSE: {} / MAE: {}".format(rmse_xgboost, mae_xgboost))


# We need to know the number of continuous values
# So when we use chunker, we can refer to the starting point to know how many to skip if timeline is not continous after skipping
# Basic idea: Loop through the dataframe, if it is continuous increase the continous counter and pos counter
# If encounter uncontinous timeline (Encounter a nan value), find the lastpoint with a loop,
# Then change the start position continous length column values to the continous counter.
# Reset the continous counter and start counter
# Result: at the start of each continous timeline, there will be a number which indicate how many step you need to skip if
# you need to get to the next timeline
from IPython.core.debugger import set_trace

def time_continous_marker(df):
    df_copy = df.copy()
    pos = 0
    start_pos = 0
    continous = 0
    size = 24
    length = df_copy.shape[0]
    while pos < length:
        try:
#             set_trace()
            if df_copy.iloc[pos : pos + size, 1].isnull().sum() == 0:
                while df_copy.iloc[pos : pos + size, 1].isnull().sum() == 0:
                    pos += size
                    continous += size
            else:
                while np.isnan(df_copy.iloc[pos, 1]) == False:
                    continous += 1
                    pos += 1
                tmp.iloc[start_pos, 7] = continous
                continous = 0
                while np.isnan(df_copy.iloc[pos, 1]) == True:
                    pos += 1
                start_pos = pos
#             print(pos)
        except IndexError:
            df_copy.iloc[start_pos, 7] = continous
            continous = 0
            print("Current position is: {}".format(pos))
            pos = length
    return df_copy


AQI = AQI.replace(0, np.nan).copy()
AQI['Continous length'] = 0
AQI = AQI.append(pd.Series(),ignore_index=True)
AQI = time_continous_marker(AQI)
AQI = AQI.dropna(subset=['AQI_h']).set_index("Date (LT)")
AQI.to_csv(_daisuquanmy_data_path + '/refined_data/refined.csv')


AQI = pd.read_csv(_daisuquanmy_data_path +'/refined_data/refined.csv')
AQI.loc[:, 'Date (LT)'] = pd.to_datetime(AQI['Date (LT)'], format="%Y-%m-%d %H:%M:%S")
AQI['site_id'] = 48
AQI.rename(columns={'Date (LT)': 'time'}, inplace=True)
AQI = AQI.set_index(['site_id', 'time'])
AQI_copy = AQI.copy()
AQI_copy


AQI_copy.to_csv(_daisuquanmy_data_path + '/refined_data/refined.csv')


test_df = AQI_copy.iloc[22090:22200].copy()
print(test_df.head(40))


# Now chunker need to be able to output chunks of train data and target in correct order
# Basic ideas: loop through each chunk of data (size is size with ,
# if the timeline is continous, add data to chunk_data [pos: pos+size]
# add data to target [pos+size+target_hour]
# if the timeline is not, skip pos according to size
# Repeat below till timeline is continous
# if after skipping, the timeline is still not continous, skip pos according to continous length colunmns
from IPython.core.debugger import set_trace
def chunker(seq, size, target_hour, debug = True):
    """
    Input: 
        + Dataframe with PM25, AQI_h, AQI_h_I, Continous length and some other columns
            - If data doesn't have Continous length columns, add 1 to data
            - Take only PM25, date columns as train
            - Take only AQI_h, AQI_h_I as target
        + size: length of each chunk
        + target_hour: labels for the hour which will be predicted
    Ouput: 2 lists: chunk_data, target
        + chunk_data: original data chunked to a specific timestep, have shape [..., timeframe, features]
        + target: Label for each chunked train/test data, have shape [..., 1]
    """
    if not 'Continous length' in seq.columns:
        seq['Continous length'] = 0

    timerange = pd.Timedelta(hours=size + target_hour)
    chunk_data = []
    target = []
    multiclass_target = []
    pos = 0
    length = seq.shape[0]
    while pos < (length - size):
        try:
            if seq.iloc[pos + size + target_hour].name[1] - seq.iloc[pos].name[1] == timerange:
                while seq.iloc[pos + size + target_hour].name[1] - seq.iloc[pos].name[1] == timerange:
                    chunk_data.append(seq.iloc[pos:pos + size].loc['PM25','Hour','Day of Week'])
                    target.append(seq.iloc[pos + size + target_hour].loc['AQI_h'])
                    multiclass_target.append(seq.iloc[pos + size + target_hour].loc['AQI_h_I'])
                    pos += 1
                    if pos + size + target_hour >= length - size:
                        print("Returned")
                        return np.array(chunk_data), np.array(target), np.array(multiclass_target)
            else:
                tmp_pos = pos + size + target_hour
                while seq.iloc[tmp_pos + size + target_hour].name[1] - seq.iloc[tmp_pos].name[1] != timerange:
                    tmp_pos += int(seq.iloc[tmp_pos, 6])
                print("Jump from {} to {}".format(pos, tmp_pos))
                pos = tmp_pos
        except IndexError:
            print("Current position is: {}".format(pos))
            print("Current tmp position is: {}".format(tmp_pos))
            pos = length - size
    print("Returned")
    return np.array(chunk_data), np.array(target), np.array(multiclass_target)
chunk_data, target, multiclass_target = chunker(AQI_copy, 12, 0)


type(chunk_data[0, 0])


# Test chunker: Need to get date of target and date of data, then if 


target


chunk_data


import numpy
numpy.version.version


chunk_data[0]


set_trace()


AQI_copy.shape


# Checking
for data, tar in zip(chunk_data, target):
    if chunk_data


def chunker(seq, size):
    timerange = pd.Timedelta(hours=size + 1)
    chunks = []
    target = []
    pos = 0
    while pos < len(seq)-size:
#         print(seq.iloc[pos + size].name)
        if seq.iloc[pos + size + 1].name - seq.iloc[pos].name == timerange:
            chunks.append(seq.iloc[pos:pos + size])
            target.append(seq.iloc[pos + size + 1])
            pos += 1
        else:
            print("Jump from {} to {}".format(pos, pos + size))
            pos += size
    return chunks, target

train = []
tmp = test_df.copy()
for train_chunk, target_chunk in chunker(tmp, 12):
    train = train + [list(train_chunk.values)]
    target = target + [list(target_chunk.values)]
#     if hour != 0:
#         train = train[:-hour]
train = np.array(train)
print("Train shape: ",train.shape)
target = np.array(target)
print("Target shape: ", target.shape)


train


y = []
target = AQI[['AQI_h', 'AQI_I']].copy()
multiclass_y = []
site_ids = [48]
for site in site_ids:
    site_label = target.loc[site]
    site_label = site_label.shift(-12 - hour).dropna()
    site_label_y = list(site_label[['AQI_h']].values.ravel())
    site_label_multiclass_y = list(site_label[['AQI_h_I']].values.ravel())
    y = y + site_label_y
    multiclass_y = multiclass_y + site_label_multiclass_y
y = np.array(y)
multiclass_y = np.array(multiclass_y)
print("Label shape: ",y.shape)


train





daisuquanmy_df_copy = daisuquanmy_df.copy()


daisuquanmy_df_copy[daisuquanmy_df_copy['Year'] == 2019]


daisuquanmy_df_copy = daisuquanmy_df_copy[daisuquanmy_df_copy['Year'] == 2015]
# daisuquanmy_df_copy[daisuquanmy_df_copy['QC Name'] == "Valid"].iloc[0]
# Remove all data before 2015-12-09 14:00:00
daisuquanmy_df_copy = daisuquanmy_df_copy.loc[daisuquanmy_df_copy['Date (LT)'] >= pd.to_datetime("2015-12-09 14:00:00")]
daisuquanmy_df_copy = daisuquanmy_df_copy.replace(-999, np.nan)
daisuquanmy_df_copy.fillna(method='ffill', limit=2, inplace=True)
daisuquanmy_df_copy.fillna(method='bfill', limit=2, inplace=True)


daisuquanmy_df_copy


# Downplaying the time from hour to date by mean


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,30))
gs = fig.add_gridspec(12,1)
axes = []
counter = 0
AQI_copy = AQI[AQI['Date (LT)'].dt.year == 2017].copy()
for month in AQI_copy['Date (LT)'].dt.month.unique():
    axes.append(fig.add_subplot(gs[counter,0]))
    axes[counter] = fig.add_subplot(gs[counter, 0])
    tmp = AQI_copy[AQI_copy['Date (LT)'].dt.month == month]
    axes[counter].plot(tmp['Date (LT)'], tmp['PM25'])
    axes[counter].set_title("Month {}".format(month))
    counter += 1
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(7,1)
axes = []
counter = 0
for year in daisuquanmy_df['Year'].unique():
    axes.append(fig.add_subplot(gs[counter,0]))
    axes[counter] = fig.add_subplot(gs[counter, 0])
    tmp = daisuquanmy_df[daisuquanmy_df['Year'] == year]
    axes[counter].plot(tmp.index, tmp['Raw Conc.'])
    axes[counter].set_title("Year {}".format(year))
    counter += 1
plt.tight_layout()
plt.show()


daisuquanmy_df_copy


daisuquanmy_df_copy[daisuquanmy_df_copy['Raw Conc.'].isnull()]


# Find first valid datapoint


daisuquanmy_df['AQI'].describe()


daisuquanmy_df['QC Name'].unique()


daisuquanmy_df[daisuquanmy_df['AQI Category'].isnull()]


daisuquanmy_df[daisuquanmy_df['QC Name'] == "Missing"]


daisuquanmy_df[daisuquanmy_df['Raw Conc.'] < 0]





daisuquanmy_df.info()





daisuquanmy_df['Date (LT)'].unique()


daisuquanmy_df[daisuquanmy_df['QC Name'] == "Missing"]


daisuquanmy_df[daisuquanmy_df.isnull().any(axis=1)]


thudohanoi_df_copy = thudohanoi_df.copy()
thudohanoi_df_copy = thudohanoi_df_copy.replace(0, np.nan)
thudohanoi_df_copy.head()


thudohanoi_df_copy.isnull().sum()


thudohanoi_df_copy = thudohanoi_df_copy.replace(0, np.nan)
print(thudohanoi_df_copy.isnull().sum())
thudohanoi_df_copy = thudohanoi_df_copy.fillna(method='ffill', limit=2)
print(thudohanoi_df_copy.isnull().sum())
thudohanoi_df_copy = thudohanoi_df_copy.fillna(method='bfill', limit=2)
print(thudohanoi_df_copy.isnull().sum())


from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
df_columns = thudohanoi_df_copy.columns
for column in df_columns:
    thudohanoi_df_copy.loc[:, column] = imp_median.fit_transform(thudohanoi_df_copy[[column]]).ravel()

