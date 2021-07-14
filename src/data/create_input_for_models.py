import os
from src.features import extract_features
from src.data import create_load_transform_processed_data


def create(df, timesteps=[1], target_hour=[1], test_output=False, dev_output=False, output_path=None, PROJ_ROOT=os.pardir):
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
                                    "model_input")
    for timesteps in timesteps:
        for target_hour in target_hour:
            # Create train, dev, test data
            train_df = extract_features.create_and_save_scale_data(df, output_path=output_path).copy()
            train_df = extract_features.add_features(train_df).copy()
            if test_output is not False:
                train_df, test_df = extract_features.generate_train_test_set_by_time(
                    train_df)
                test, y_test, multiclass_y_test = extract_features.data_preprocessing(
                    test_df, target_hour, timesteps=timesteps)
                create_load_transform_processed_data.reshape_array_and_save_to_path(
                    test, y_test, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="test")
            if dev_output is not False:
                train_df, dev_df = extract_features.generate_train_test_set_by_time(
                    train_df)
                dev, y_dev, multiclass_y_dev = extract_features.data_preprocessing(
                    dev_df, target_hour, timesteps=timesteps)
                create_load_transform_processed_data.reshape_array_and_save_to_path(
                    dev, y_dev, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="dev")

            train, y_train, multiclass_y = extract_features.data_preprocessing(
                train_df, target_hour, timesteps=timesteps)

            # Save data to file
            create_load_transform_processed_data.reshape_array_and_save_to_path(
                train, y_train, path=output_path, timesteps=timesteps, target_hour=target_hour, data_type="train")
    train = train.astype('float32')
    y_train = y_train.astype('float32')
    print("Input have been created")
    return train, y_train
