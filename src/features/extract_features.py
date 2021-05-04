def data_preprocessing(df, hour = 1, timesteps = 12, debug = False):
    """
    Input: 
        + raw df contains all data 
            columns: ['PM25', 'AQI_h', 'AQI_h_I']
            index: ['time']
        + the predict hour to make data
    Ouput: 
        + Processed train data:
        + Target for the train data:
        + Classes of target for spliting the data
    """

    def chunker(seq, size):
        return (seq.iloc[pos:pos + size] for pos in range(0, len(seq)-size))

    def chunker_special(seq, target_hour, size, feature_cols, target_cols = ['AQI_h', 'AQI_h_I'], debug = False):
        """
        Input: 
            + Dataframe with PM25, AQI_h, AQI_h_I, Continous length and some other columns
                - If data doesn't have Continous length columns, add 1 to data
                - Take only PM25, date columns as train
                - Take only AQI_h, AQI_h_I as target
                - Data must only have time as index
            + size: length of each chunk
            + target_hour: labels for the hour which will be predicted
        Ouput: 2 lists: chunk_data, target
            + chunk_data: original data chunked to a specific timestep, have shape [..., timeframe, features]
            + target: Label for each chunked train/test data, have shape [..., 1]
        """

        timerange = pd.Timedelta(hours=size + target_hour)
        pos = 0
        length = seq.shape[0]
        continous_length_index = seq.columns.get_loc('Continous length')
        if debug == True:
            set_trace()
        while pos < (length - size):
            try:
                if seq.iloc[pos + size + target_hour].name - seq.iloc[pos].name == timerange:
                    while seq.iloc[pos + size + target_hour].name - seq.iloc[pos].name == timerange:
                        chunk_tmp = seq.iloc[pos:pos + size].loc[:,feature_cols]
                        target_tmp = seq.iloc[pos + size + target_hour].loc['AQI_h']
                        multiclass_tmp = seq.iloc[pos + size + target_hour].loc['AQI_h_I']
                        yield chunk_tmp, target_tmp, multiclass_tmp
                        pos += 1
                        if pos + size + target_hour >= length - size:
                            return None
                else:
                    tmp_pos = pos + size + target_hour
                    while seq.iloc[tmp_pos + size + target_hour].name - seq.iloc[tmp_pos].name != timerange:
                        # set_trace()
                        tmp_pos += int(seq.iloc[tmp_pos, continous_length_index])
                    pos = tmp_pos
            except IndexError:
                set_trace()
                print("Current position is: {}".format(pos))
                print("Current tmp position is: {}".format(tmp_pos))
                pos = length - size
        return None
# ===========================================================================================================================
    hour = hour - 1
    df_copy = df.copy()
    exclude_cols = ['Continous length', 'AQI_h_I', 'PM25']
    feature_cols = df_copy.drop(exclude_cols, axis=1).columns
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
            generator = chunker_special(seq=tmp, target_hour=hour, size=timesteps, feature_cols=feature_cols)
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
    print("Feature shape: ",train.shape)
    print("Label shape: ",y.shape)

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
    df_time_features['sin_hour'] = np.sin(2*np.pi*df_time_features['Hour']/hour_in_day)
    df_time_features['cos_hour'] = np.cos(2*np.pi*df_time_features['Hour']/hour_in_day)
    month_in_year = 12
    df_time_features['sin_month'] = np.sin(2*np.pi*df_time_features['Month']/month_in_year)
    df_time_features['cos_month'] = np.cos(2*np.pi*df_time_features['Month']/month_in_year)
    day_in_week = 6
    df_time_features['sin_dayweek'] = np.sin(2*np.pi*df_time_features['Day of Week']/day_in_week)
    df_time_features['cos_dayweek'] = np.cos(2*np.pi*df_time_features['Day of Week']/day_in_week)
    df_time_features['sin_daymonth'] = np.sin(2*np.pi*df_time_features['Day of Month']/df_time_features['Days in Month'])
    df_time_features['cos_daymonth'] = np.cos(2*np.pi*df_time_features['Day of Month']/df_time_features['Days in Month'])
    # One hot encode year data
    one_hot_df = pd.get_dummies(df_time_features['Year'], drop_first=True, prefix='year')
    df_time_features = df_time_features.join(one_hot_df)
    # Input weekday/weekend/holiday data
    vn_holidays = np.array(list(holidays.VN(years=[2015,2016,2017,2018,2019,2020,2021]).keys()))
    holiday_mask = np.isin(time_index.date, vn_holidays)
    masks = (holiday_mask) | (df_time_features['Day of Week'].values == 5) | (df_time_features['Day of Week'].values == 6)
    df_time_features['day_off'] = np.where(masks == True, 1, 0)
    df_time_features = df_time_features.drop(columns=['Day of Month', 'Month', 'Day of Week', 'Days in Month', 'Year', 'Hour'])
#     Input lagged data
    windows = list(range(1,13))
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
    data_df = df[['PM25', 'AQI_h', 'AQI_h_I', 'Continous length']].copy()

    # Job: Normalize train data

    scaler = MinMaxScaler(feature_range=(-1,1))

    for col in ['PM25', 'AQI_h']:
        data_df[[col]] = scaler.fit_transform(data_df[[col]])

    columns = ['site_id', 'time', 'PM25', 'AQI_h', 'AQI_h_I', 'Continous length']
    df_numpy = data_df.reset_index().to_numpy()

    # Add onehot site label
    one_hot_site = pd.get_dummies(data_df.index.get_level_values(0), prefix='site', drop_first=True).astype(int)
    columns.extend(one_hot_site.columns)
    # Add onehot air category
    one_hot_cat = pd.get_dummies(data_df['AQI_h_I'], drop_first=True, prefix='cat').astype(int)
    columns.extend(one_hot_cat.columns)
    # Add time features
    time_features, time_columns = extract_time_features(data_df)
    columns.extend(time_columns)
    df_numpy = np.concatenate([df_numpy, one_hot_site.values, one_hot_cat.values, time_features], axis=1)
    

    final_df = pd.DataFrame(df_numpy, columns=columns).set_index(['site_id', 'time'])
    for float_col in final_df.loc[:, final_df.dtypes == float].columns:
        final_df.loc[:, float_col] = final_df.loc[:, float_col].values.round(6)
    return final_df

def generate_train_test_set_by_time(df, ratio = 0.1):
    # Generate test set by taking the lastest 10% data from each site
#     train_df = df.drop(index=(48), level=0).copy()
    train_df = df.copy()
    latest_time = train_df.index.get_level_values(1).max()
    oldest_time = train_df.index.get_level_values(1).min()
    cutoff_hour = (latest_time - oldest_time).total_seconds()
    cutoff_hour = cutoff_hour // 3600
    cutoff_hour = cutoff_hour * ratio
    test_df = train_df[train_df.index.get_level_values(1) >= (latest_time - pd.Timedelta(hours=cutoff_hour))]
    train_df = train_df[train_df.index.get_level_values(1) < (latest_time - pd.Timedelta(hours=cutoff_hour))]
    # Generate train_test set for site 48 and 49
#     train_df_48 = df[df.index.get_level_values(0) == 48]
#     latest_time = train_df_48.index.get_level_values(1).max()
#     oldest_time = train_df_48.index.get_level_values(1).min()
#     cutoff_hour = (latest_time - oldest_time).total_seconds()
#     cutoff_hour = cutoff_hour // 3600
#     cutoff_hour = cutoff_hour * ratio
#     test_df = test_df.append(train_df_48[train_df_48.index.get_level_values(1) >= (latest_time - pd.Timedelta(hours=cutoff_hour))])
#     train_df = train_df.append(train_df_48[train_df_48.index.get_level_values(1) < (latest_time - pd.Timedelta(hours=cutoff_hour))])
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
def reshape_array_and_save_to_path(arr_data, arr_label, path, timesteps, target_hour, data_type="Train"):
    # reshaping the array from 3D 
    # matrice to 2D matrice. 
    arr_data_reshaped = arr_data.reshape(arr_data.shape[0], -1)
    arr_label_reshaped = arr_label.reshape(arr_label.shape[0], -1)
    
    # saving reshaped array to file.
    saved_data = np.savez_compressed(path + "/{}_{}_{}_data.npz".format(timesteps, target_hour, data_type), arr_data_reshaped)
    saved_label = np.savez_compressed(path + "/{}_{}_{}_label.npz".format(timesteps, target_hour, data_type), arr_label_reshaped)
    
    # retrieving data from file.
    loaded_arr_data_file = np.load(path + "/{}_{}_{}_data.npz".format(timesteps, target_hour, data_type), allow_pickle=True)
    loaded_arr_label_file = np.load(path + "/{}_{}_{}_label.npz".format(timesteps, target_hour, data_type), allow_pickle=True)
    loaded_arr_data = loaded_arr_data_file['arr_0']
    loaded_arr_data_file.close()
    loaded_arr_label = loaded_arr_label_file['arr_0'].ravel()
    loaded_arr_label_file.close()
    # This loadedArr is a 2D array, therefore
    # we need to convert it to the original 
    # array shape.reshaping to get original 
    # matrice with original shape. 
    loaded_arr_data = loaded_arr_data.reshape( 
        loaded_arr_data.shape[0], loaded_arr_data.shape[1] // arr_data.shape[2], arr_data.shape[2])
    
    features_save = np.save(path+"/features.npy", arr_data.shape[2])
    # check the shapes:
    print("Data array:")
    print("shape of arr: ", arr_data.shape) 
    print("shape of loaded_array: ", loaded_arr_data.shape)
    
    # check if both arrays are same or not: 
    if (arr_data == loaded_arr_data).all(): 
        print("Yes, both the arrays are same") 
    else: 
        print("No, both the arrays are not same")
    # check the shapes:
    print("Label array:")
    print("shape of arr: ", arr_label.shape) 
    print("shape of loaded_array: ", loaded_arr_label.shape)

    # check if both arrays are same or not: 
    if (arr_label == loaded_arr_label).all(): 
        print("Yes, both the arrays are same") 
    else: 
        print("No, both the arrays are not same")
    return None
def load_reshaped_array(timesteps, target_hour, folder_path, data_type="train"):
    features = np.load(folder_path + "/features.npy", allow_pickle=True).ravel()[0]
    loaded_file = np.load(folder_path + "/{}_{}_{}_data.npz".format(timesteps, target_hour, data_type), allow_pickle=True)
    loaded_data = loaded_file['arr_0']
    loaded_data = loaded_data.reshape( 
            loaded_data.shape[0], loaded_data.shape[1] // features, features).astype(float)
    loaded_file.close()
    loaded_file_label = np.load(folder_path + "/{}_{}_{}_label.npz".format(timesteps, target_hour, data_type), allow_pickle=True)
    loaded_label = loaded_file_label['arr_0'].ravel().astype(float)
    loaded_file_label.close()
    return loaded_data, loaded_label
def create_tensorflow_dataset(arr_data, arr_label, batch_size):
    tf_dataset = tf.data.Dataset.from_tensor_slices((arr_data, arr_label))
    tf_dataset = tf_dataset.repeat().batch(batch_size, drop_remainder=True)
    steps_per_epochs = len(arr_data) // batch_size
    return tf_dataset, steps_per_epochs
