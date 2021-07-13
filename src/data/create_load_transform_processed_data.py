
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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
    if len(arr_data) % batch_size != 0:
        if len(arr_data) // batch_size != 0:
            remain_count = len(arr_data)%batch_size
            arr_data = arr_data[remain_count:]
            arr_label = arr_label[remain_count:]
        else:
            batch_size = len(arr_data)
    tf_dataset = tf.data.Dataset.from_tensor_slices((arr_data, arr_label))
    tf_dataset = tf_dataset.repeat().batch(batch_size, drop_remainder=True)
    steps_per_epochs = len(arr_data) // batch_size
    print(arr_data)
    return tf_dataset, steps_per_epochs