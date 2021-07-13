import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics, initializers
import os


def get_model_name(timesteps, target_hour):
    return 'model_{}_{}.h5'.format(timesteps, target_hour)


def get_model_weigts_name(timesteps, target_hour):
    return 'weights_{}_{}.ckpt'.format(timesteps, target_hour)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def create_model(batch_size=128, timesteps=24, features=3, dropout=0.2, add_lstm_layer=1, add_layer=1, batch_normalization=True):
    """create_model Create LSTM model

    [extended_summary]

    Args:
        batch_size (int, optional): [description]. Defaults to 128.
        timesteps (int, optional): [description]. Defaults to 24.
        features (int, optional): [description]. Defaults to 3.
        dropout (float, optional): [description]. Defaults to 0.0.
        add_lstm_layer (int, optional): [description]. Defaults to 0.
        add_layer (int, optional): [description]. Defaults to 0.
        batch_normalization (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # var init
    initial_learning_rate = 1.0
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.85,
        staircase=True)

    inputs = keras.Input(batch_input_shape=(batch_size, timesteps, features))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(units=128, kernel_initializer=initializers.RandomNormal(stddev=0.1),
                                         bias_initializer='zeros', stateful=True, return_sequences=True, dropout=dropout))(x)
    for i in range(0, add_lstm_layer):
        x = layers.Bidirectional(layers.LSTM(units=128, kernel_initializer=initializers.RandomNormal(stddev=0.1),
                                             stateful=True, bias_initializer='zeros', return_sequences=True, dropout=dropout))(x)
    x = layers.Bidirectional(layers.LSTM(units=128, kernel_initializer=initializers.RandomNormal(stddev=0.1),
                                         stateful=True, bias_initializer='zeros', return_sequences=False, dropout=dropout))(x)
    if batch_normalization == True:
        x = layers.BatchNormalization()(x)
    if add_layer == 1:
        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01))(x)
    outputs = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.01))(x)
    # outputs = layers.Dense(1, activation="relu")(x)
    model = keras.Model(inputs, outputs)
    my_optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    # tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=my_optimizer, loss=root_mean_squared_error,
                  metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()])
    return model


def load_model_with_weight(model_path, timesteps, target_hour):
    model = keras.models.load_model("{}/{}/model_of_{}_hour/{}".format(model_path,
                                                                       timesteps, target_hour, get_model_name(timesteps=timesteps, target_hour=target_hour)),
                                    custom_objects={'LeakyReLU': layers.LeakyReLU(alpha=0.01),
                                                    'root_mean_squared_error': root_mean_squared_error})
    model.load_weights('{}/{}/model_of_{}_hour/{}'.format(model_path, timesteps, target_hour,
                                                          get_model_weigts_name(timesteps=timesteps, target_hour=target_hour))).expect_partial()
    return model


def load_combined_model(timesteps, hour, PROJ_ROOT):
    path = os.path.join(PROJ_ROOT, "models")
    model = tf.keras.models.load_model("{}/combined/hanoi/{}".format(path,
                                                                     get_model_name(timesteps=timesteps, target_hour=hour)),
                                       custom_objects={'LeakyReLU': layers.LeakyReLU(alpha=0.01),
                                                       'root_mean_squared_error': root_mean_squared_error})
    model_path = "{}/combined/hanoi/{}".format(path,
                                               get_model_name(timesteps=timesteps, target_hour=hour))
    return model, model_path
