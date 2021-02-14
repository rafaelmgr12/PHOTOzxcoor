import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regression Libraries

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, BaggingRegressor, StackingRegressor

# Other libraries

from sklearn.model_selection import cross_val_score, KFold, train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import norm, skew  # for some statistics

# Neural Network Libs
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import confusion_matrix
import keras as ks
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers



def clean_tab(tab, col, val):
    '''Function to clean the
 droping the row of values'''
    tab.drop(tab[tab[col] > val].index, inplace=True)


def tts_split(X, y, size, splits):
    '''Split the data in Train and
     test using the Shuffle split'''

    rs = ShuffleSplit(n_splits=splits, test_size=size)

    rs.get_n_splits(X)

    for train_index, test_index in rs.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def rmsle_cv(model, X_train, y_train):
    """Root mean Square using cross validation"""
    kf = KFold(3, shuffle=True, random_state=None).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train.ravel(),
                                    scoring="neg_mean_squared_error", cv=kf))
    return(rmse)


def rmsle(y, y_pred):
    '''Root mean Square'''
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse_ann(y_true, y_pred):
    '''root mean square erro for using as metric for the ANNs'''
    return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true), axis=-1))


def rmse_ann2(y_true, y_pred):
    return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true), axis=-1))


def rmse_ann3(y_true, y_pred):
    return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true), axis=-1))


def plot_history(history):
    '''Plot the graphics for the train_error and val_loss'''
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
# Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def get_features_targets_des1(data):
    '''Extract the colors using the mag.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['MAG_AUTO_G'].values - data['MAG_AUTO_R'].values
    features[:, 1] = data['MAG_AUTO_R'].values - data['MAG_AUTO_I'].values
    features[:, 2] = data['MAG_AUTO_I'].values - data['MAG_AUTO_Z'].values
    features[:, 3] = data['MAG_AUTO_Z'].values - data['MAG_AUTO_Y'].values
    # features[:, 4] = data['WAVG_MAG_PSF_G'] - data['WAVG_MAG_PSF_R']
    # features[:, 5] = data['WAVG_MAG_PSF_R'] - data['WAVG_MAG_PSF_I']
    # features[:, 6] = data['WAVG_MAG_PSF_I'] - data['WAVG_MAG_PSF_Z']
    # features[:, 7] = data['WAVG_MAG_PSF_Z'] - data['WAVG_MAG_PSF_Y']
    features[:, 4] = data["MAG_AUTO_I"].values

    targets = data['z'].values
    return features, targets


def get_features_targets_des2(data):
    '''Extract the colors using the mag_derred.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['MAG_AUTO_G_DERED'].values - \
        data['MAG_AUTO_R_DERED'].values
    features[:, 1] = data['MAG_AUTO_R_DERED'].values - \
        data['MAG_AUTO_I_DERED'].values
    features[:, 2] = data['MAG_AUTO_I_DERED'].values - \
        data['MAG_AUTO_Z_DERED'].values
    features[:, 3] = data['MAG_AUTO_Z_DERED'].values - \
        data['MAG_AUTO_Y_DERED'].values
    # features[:, 4] = data['WAVG_MAG_PSF_G_DERED'] - data['WAVG_MAG_PSF_R_DERED']
    # features[:, 5] = data['WAVG_MAG_PSF_R_DERED'] - data['WAVG_MAG_PSF_I_DERED']
    # features[:, 6] = data['WAVG_MAG_PSF_I_DERED'] - data['WAVG_MAG_PSF_Z_DERED']
    # features[:, 7] = data['WAVG_MAG_PSF_Z_DERED'] - data['WAVG_MAG_PSF_Y_DERED']
    features[:, 4] = data["MAG_AUTO_I_DERED"].values

    targets = data['z'].values
    return features, targets


def get_features_targets_gama1(data):
    '''Extract the colors using the mag_derred.
    Here I'm using the DES-mag, for other survey a change is needed'''
    features = np.zeros(shape=(len(data), 5))

    features[:, 0] = data['gKronMag'].values - \
        data['rKronMag'].values
    features[:, 1] = data['rKronMag'].values - \
        data['iKronMag'].values
    features[:, 2] = data['iKronMag'].values - \
        data['zKronMag'].values
    features[:, 3] = data['zKronMag'].values - \
        data['yKronMag'].values
    # features[:, 4] = data['WAVG_MAG_PSF_G_DERED'] - data['WAVG_MAG_PSF_R_DERED']
    # features[:, 5] = data['WAVG_MAG_PSF_R_DERED'] - data['WAVG_MAG_PSF_I_DERED']
    # features[:, 6] = data['WAVG_MAG_PSF_I_DERED'] - data['WAVG_MAG_PSF_Z_DERED']
    # features[:, 7] = data['WAVG_MAG_PSF_Z_DERED'] - data['WAVG_MAG_PSF_Y_DERED']
    features[:, 4] = data["iKronMag"].values

    targets = data['Z'].values
    return features, targets


def smote(X, y, n, k):
    '''Add values for the dataset using the smote technique'''

    if n == 0:
        return X, y

    knn = KNeighborsRegressor(k, "distance").fit(X, y)
    # choose random neighbors of random points
    ix = np.random.choice(len(X), n)
    nn = knn.kneighbors(X[ix], return_distanec=False)
    newY = knn.predict(X[ix])
    nni = np.random.choice(k, n)
    ix2 = np.array([n[i] for n, i in zip(nn, nni)])

    # synthetically generate mid-point between each point and a neighbor
    dif = X[ix] - X[ix2]
    gap = np.random.rand(n, 1)
    newX = X[ix] + dif*gap
    return np.r_[X, newX], np.r_[y, newY]


def gaussian_noise(X, y, sigma):
    """
    Add gaussian noise to the dataset
    """
    X = X.copy()
    y = y.copy()
    for _ in range(n):
        X = np.r_[X, _X + np.random.randn(*_X.shape)*sigma]
        y = np.r_[y, _y]
    return X, y


def plot_model(model):
    return SVG(keras.utils.vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def rmse_loss_keras(y_true, y_pred):
    diff = keras.backend.square(
        (y_pred - y_true) / (keras.backend.abs(y_true) + 1))
    return keras.backend.sqrt(keras.backend.mean(diff))

'''
def build_nn(input_dim, shape, l2_rate=1e-5, l1_rate=1e-4, kernel_initializer, act_type="tanh",
opt_type = ks.optimizers.RMSprop(),n_neurons = 10):

    ann_model = Sequential([Dense(n_inputs=input_dim, input_shape=shape, kernel_regularizer=regularizers.l1_l2(l1=l1_rate, l2=l2_rate),
                                  bias_regularizer=regularizers.l2(l2_rate), activity_regularizer=regularizers.l2(l2_rate)),
                            Dense(n_neurons, kernel_initializer=kernel_init,  kernel_constraint=max_norm(2.5), activation=act_type, kernel_regularizer=regularizers.l1_l2(l1=l1_rate, l2=l2_rate),
                                  bias_regularizer=regularizers.l2(l2_rate), activity_regularizer=regularizers.l2(l2_rate)),
                            BatchNormalization(),
                            Dense(n_neurons, kernel_initializer=kernel_init, kernel_constraint=max_norm(2.5), activation=act_type, kernel_regularizer=regularizers.l1_l2(l1=l1_rate, l2=l2_rate),
                                  bias_regularizer=regularizers.l2(l2_rate), activity_regularizer=regularizers.l2(l2_rate)),
                            Dense(1, activation=None, name="output")
                            ])
    opt = opt_type
    ann_model.compile(optimizer=opt, loss=rmse_ann, metrics=[
                      'mse', 'mae', rmse_ann])

    return ann_model

'''
def model_nn(input_dim, n_hidden_layers, dropout=0, batch_normalization=False,
             activation='relu', neurons_decay=0, starting_power=1, l2=0,
             compile_model=True, trainable=True):
    """Define an ANN with tanh activation"""

    assert dropout >= 0 and dropout < 1
    assert batch_normalization in {True, False}
    model = keras.models.Sequential()

    for layer in range(n_hidden_layers):
        n_units = 2**(int(np.log2(input_dim)) +
                      starting_power - layer*neurons_decay)
        if n_units < 8:
            n_units = 8
        if layer == 0:
            model.add(Dense(units=n_units, input_dim=input_dim, name='Dense_' + str(layer + 1),
                            kernel_regularizer=keras.regularizers.l2(l2)))
        else:
            model.add(Dense(units=n_units, name='Dense_' + str(layer + 1),
                            kernel_regularizer=keras.regularizers.l2(l2)))
        if batch_normalization:
            model.add(BatchNormalization(
                name='BatchNormalization_' + str(layer + 1)))
        model.add(Activation('tanh', name='Activation_' + str(layer + 1)))
        if dropout > 0:
            model.add(Dropout(dropout, name='Dropout_' + str(layer + 1)))

    model.add(Dense(units=1, name='Dense_' + str(n_hidden_layers+1),
                    kernel_regularizer=keras.regularizers.l2(l2)))
    model.trainable = trainable
    if compile_model:
        model.compile(loss=rmse_loss_keras, optimizer=keras.optimizers.Adam())

    return model
