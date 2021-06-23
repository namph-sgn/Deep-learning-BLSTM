import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def data_preprocessing(df, hour=1, timesteps=12, debug=False):
    """
    Input: 
        + raw df contains all data 
            columns: ['PM25', 'AQI_h', 'AQI_h_I']
            index: ['time']
        + the predict hour to make data
        + timesteps
    Ouput: 
        + Processed train data:
        + Target for the train data:
        + Classes of target for spliting the data
    """

    def chunker(seq, size):
        return (seq.iloc[pos:pos + size] for pos in range(0, len(seq)-size))

    def chunker_special(seq, target_hour, size, feature_cols, target_cols=['AQI_h', 'AQI_h_I'], debug=False):
        """chunker_special [summary]

        [extended_summary]

        Args:
            seq (DataFrame): Dataframe with AQI_h, AQI_h_I, Continous length and some other columns.
                        - If data doesn't have Continous length columns, add 1 to data
                        - Take only PM25, date columns as train
                        - Take only AQI_h, AQI_h_I as target
                        - Data must only have time as index
            target_hour (int): labels for the hour which will be predicted
            size (int): length of each chunk
            feature_cols (list): columns which is considered as features
            target_cols (list, optional): [description]. Defaults to ['AQI_h', 'AQI_h_I'].
            debug (bool, optional): [description]. Defaults to False.

        Returns:
            chunk_data: original data chunked to a specific timestep, have shape [..., timeframe, features]
            target: Label for each chunked train/test data, have shape [..., 1]

        Yields:
            [type]: [description]
        """
        timerange = pd.Timedelta(hours=size + target_hour)
        pos = 0
        length = seq.shape[0]
        # If input dataframe doesn't have continous length (perfect data), add continous length =0
        try:
            continous_length_index = seq.columns.get_loc('Continous length')
        except:
            seq['Continous length'] = 0
            print(seq.columns)
            continous_length_index = seq.columns.get_loc('Continous length')
        while pos < (length - size):
            try:
                if seq.iloc[pos + size + target_hour].name - seq.iloc[pos].name == timerange:
                    while seq.iloc[pos + size + target_hour].name - seq.iloc[pos].name == timerange:
                        chunk_tmp = seq.iloc[pos:pos +
                                             size].loc[:, feature_cols]
                        target_tmp = seq.iloc[pos +
                                              size + target_hour].loc['AQI_h']
                        multiclass_tmp = seq.iloc[pos +
                                                  size + target_hour].loc['AQI_h_I']
                        yield chunk_tmp, target_tmp, multiclass_tmp
                        pos += 1
                        if pos + size + target_hour >= length - size:
                            return None
                else:
                    tmp_pos = pos + size + target_hour
                    while seq.iloc[tmp_pos + size + target_hour].name - seq.iloc[tmp_pos].name != timerange:
                        # set_trace()
                        tmp_pos += int(seq.iloc[tmp_pos,
                                       continous_length_index])
                    pos = tmp_pos
            except IndexError:
                print("Current position is: {}".format(pos))
                print("Current tmp position is: {}".format(tmp_pos))
                pos = length - size
        return None
# ===========================================================================================================================
    hour = hour - 1
    df_copy = df.copy()
    exclude_cols = ['Continous length', 'AQI_h_I', 'PM25']
    feature_cols = df_copy.drop(exclude_cols, axis=1, errors='ignore').columns
    # Remember: data shape is (datapoint, 12, 7)
    # Remember: target shape is (datapoint, 1, 1)
    site_ids = list(df_copy.index.get_level_values(0).unique())
    train = []
    y = []
    multiclass_y = []
    for site in site_ids:
        if (site == 48) or (site == 49):
            tmp = df_copy.loc[site]
            tmp = tmp.iloc[2:]
            generator = chunker_special(
                seq=tmp, target_hour=hour, size=timesteps, feature_cols=feature_cols)
            for train_tmp, y_tmp, multiclass_tmp in generator:
                train.append(list(train_tmp.values))
                y.append(y_tmp)
                multiclass_y.append(multiclass_tmp)
        else:
            # Data
            tmp_train = df_copy.loc[site, feature_cols].copy()
            tmp_train = tmp_train.iloc[2:]
            generator = chunker(tmp_train, timesteps)
            for i in generator:
                train.append(list(i.values))
            if hour != 0:
                train = train[:-hour]
            # Target
            tmp_target = df_copy.loc[site, ['AQI_h', 'AQI_h_I']].copy()
            tmp_target = tmp_target.iloc[2:]
            tmp_target = tmp_target.shift(-timesteps - hour).dropna()
            tmp_target_y = tmp_target[['AQI_h']].values.ravel()
            tmp_target_multiclass_y = tmp_target[['AQI_h_I']].values.ravel()
            y = y + list(tmp_target_y)
            multiclass_y = multiclass_y + list(tmp_target_multiclass_y)
    train = np.array(train)
    y = np.array(y)
    multiclass_y = np.array(multiclass_y)
    print("Feature shape: ", train.shape)
    print("Label shape: ", y.shape)

    import warnings
    warnings.filterwarnings('ignore')

    return train, y, multiclass_y


def extract_time_features(df):
    # Job: Expand time in data
    time_index = df.index.get_level_values(1)
    df_time_features = pd.DataFrame()
    df_time_features['Hour'] = time_index.hour.astype(float)
    df_time_features['Month'] = time_index.month.astype(float)
    df_time_features['Day of Week'] = time_index.dayofweek.astype(float)
    df_time_features['Day of Month'] = time_index.day.astype(float)
    df_time_features['Days in Month'] = time_index.daysinmonth.astype(float)
    df_time_features['Year'] = time_index.year.astype(float)

    # Job: Encode time cyclical data
    hour_in_day = 23
    df_time_features['sin_hour'] = np.sin(
        2*np.pi*df_time_features['Hour']/hour_in_day)
    df_time_features['cos_hour'] = np.cos(
        2*np.pi*df_time_features['Hour']/hour_in_day)
    month_in_year = 12
    df_time_features['sin_month'] = np.sin(
        2*np.pi*df_time_features['Month']/month_in_year)
    df_time_features['cos_month'] = np.cos(
        2*np.pi*df_time_features['Month']/month_in_year)
    day_in_week = 6
    df_time_features['sin_dayweek'] = np.sin(
        2*np.pi*df_time_features['Day of Week']/day_in_week)
    df_time_features['cos_dayweek'] = np.cos(
        2*np.pi*df_time_features['Day of Week']/day_in_week)
    df_time_features['sin_daymonth'] = np.sin(
        2*np.pi*df_time_features['Day of Month']/df_time_features['Days in Month'])
    df_time_features['cos_daymonth'] = np.cos(
        2*np.pi*df_time_features['Day of Month']/df_time_features['Days in Month'])
    # One hot encode year data
    one_hot_df = pd.get_dummies(
        df_time_features['Year'], drop_first=True, prefix='year')
    df_time_features = df_time_features.join(one_hot_df)
    # Input weekday/weekend/holiday data
    vn_holidays = np.array(
        list(holidays.VN(years=[2015, 2016, 2017, 2018, 2019, 2020, 2021]).keys()))
    holiday_mask = np.isin(time_index.date, vn_holidays)
    masks = (holiday_mask) | (df_time_features['Day of Week'].values == 5) | (
        df_time_features['Day of Week'].values == 6)
    df_time_features['day_off'] = np.where(masks == True, 1, 0)
    df_time_features = df_time_features.drop(
        columns=['Day of Month', 'Month', 'Day of Week', 'Days in Month', 'Year', 'Hour'])
#     Input lagged data
    windows = list(range(1, 13))
    windows.append(24)
    for window in windows:
        feature = 'AQI_h'
        series_rolled = df['AQI_h'].rolling(window=window, min_periods=0)
        series_mean = series_rolled.mean().shift(1).reset_index()
        series_std = series_rolled.std().shift(1).reset_index()
        df_time_features[f"{feature}_mean_lag{window}"] = series_mean['AQI_h'].values
#         df_time_features[f"{feature}_std_lag{window}"] = series_std['AQI_h'].values
        df_time_features.fillna(df_time_features.mean(), inplace=True)
        df_time_features.fillna(df['AQI_h'].mean(), inplace=True)

    return df_time_features.values, df_time_features.columns


def add_features(df):

    # Change all data to numpy, then concatenate those numpy.
    # Then construct the dataframe to old frame. This can work
    data_df = df[['AQI_h', 'AQI_h_I', 'Continous length']].copy()

    # Job: Normalize train data

    scaler = MinMaxScaler(feature_range=(-1, 1))

    for col in ['AQI_h']:
        data_df[[col]] = scaler.fit_transform(data_df[[col]])

    columns = ['site_id', 'time',
               'AQI_h', 'AQI_h_I', 'Continous length']
    df_numpy = data_df.reset_index().to_numpy()

    # Add onehot site label
    one_hot_site = pd.get_dummies(data_df.index.get_level_values(
        0), prefix='site', drop_first=True).astype(int)
    columns.extend(one_hot_site.columns)
    # Add onehot category
    one_hot_cat = pd.get_dummies(
        data_df['AQI_h_I'], drop_first=True, prefix='cat').astype(int)
    columns.extend(one_hot_cat.columns)
    # Add time features
    time_features, time_columns = extract_time_features(data_df)
    columns.extend(time_columns)
    df_numpy = np.concatenate(
        [df_numpy, one_hot_site.values, one_hot_cat.values, time_features], axis=1)

    final_df = pd.DataFrame(
        df_numpy, columns=columns).set_index(['site_id', 'time'])
    for float_col in final_df.loc[:, final_df.dtypes == float].columns:
        final_df.loc[:, float_col] = final_df.loc[:, float_col].values.round(6)
    return final_df


def generate_train_test_set_by_time(df, ratio=0.1):
    # Generate test set by taking the lastest 10% data from each site
    if 49 in df.index.get_level_values(0):
        train_df = df.copy()
        latest_time = train_df.index.get_level_values(1).max()
        oldest_time = train_df.index.get_level_values(1).min()
        cutoff_hour = (latest_time - oldest_time).total_seconds()
        cutoff_hour = cutoff_hour // 3600
        cutoff_hour = cutoff_hour * ratio
        test_df = train_df[train_df.index.get_level_values(
            1) >= (latest_time - pd.Timedelta(hours=cutoff_hour))]
        train_df = train_df[train_df.index.get_level_values(
            1) < (latest_time - pd.Timedelta(hours=cutoff_hour))]
    else:
        train_df = df.drop(index=(48), level=0).copy()
        latest_time = train_df.index.get_level_values(1).max()
        oldest_time = train_df.index.get_level_values(1).min()
        cutoff_hour = (latest_time - oldest_time).total_seconds()
        cutoff_hour = cutoff_hour // 3600
        cutoff_hour = cutoff_hour * ratio
        test_df = train_df[train_df.index.get_level_values(
            1) >= (latest_time - pd.Timedelta(hours=cutoff_hour))]
        train_df = train_df[train_df.index.get_level_values(
            1) < (latest_time - pd.Timedelta(hours=cutoff_hour))]
        # Generate train_test set for site 48 and 49
        train_df_48 = df[df.index.get_level_values(0) == 48].copy()
        latest_time = train_df_48.index.get_level_values(1).max()
        oldest_time = train_df_48.index.get_level_values(1).min()
        cutoff_hour = (latest_time - oldest_time).total_seconds()
        cutoff_hour = cutoff_hour // 3600
        cutoff_hour = cutoff_hour * ratio
        test_df = test_df.append(train_df_48[train_df_48.index.get_level_values(
            1) >= (latest_time - pd.Timedelta(hours=cutoff_hour))])
        train_df = train_df.append(train_df_48[train_df_48.index.get_level_values(
            1) < (latest_time - pd.Timedelta(hours=cutoff_hour))])
    return train_df, test_df


def generate_train_test_set_by_skfold(df):
    from sklearn.model_selection import StratifiedKFold
    # Generate test set by taking 10% data from everysite by stratified kfold method
    df_copy = df.copy()
    site_ids = list(df_copy.index.get_level_values(0).unique())
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for site in site_ids:
        site_df = df_copy.loc[site].copy()
        y_multiclass = site_df['AQI_h_I'].values
        for train_index, test_index in skf.split(np.zeros(len(y_multiclass)), y_multiclass):
            site_train, site_test = site_df.iloc[train_index], site_df.iloc[test_index]
            train_df = train_df.append(site_train)
            test_df = test_df.append(site_test)
            break
    return train_df, test_df
