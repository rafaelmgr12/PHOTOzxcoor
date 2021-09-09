import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Regression Libraries
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
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


def random_lightGBM(len_dataset):

    b_type = ["gbdt", "dart", "goss"]
    l_rate = 10.0 ** -np.arange(0, 20)
    n_estimator = list(range(200, 1000, 100))
    num_leaves = list(range(31, 200, 10))

    typee = random.choice(b_type)
    rate = random.choice(l_rate)
    nest = random.choice(n_estimator)
    leaves = random.choice(num_leaves)
    alpha = random.choice(l_rate)
    lambd = random.choice(l_rate)

    model = LGBMRegressor(boosting_type=typee, num_leaves=leaves,
                          learning_rate=rate, n_estimators=nest, reg_alpha=alpha, reg_lambda=lambd)
    return model


def random_DTR(len_train_features, alg="Boost"):
    """ Define a Boosted/Bagging  and Gradient Decision tree random
    """
    # DTR parameters
    min_samples_s = list(range(2, 100, 2))
    min_samples_leaf = list(range(2, 100, 2))
    crit = ['mse', 'friedman_mse', 'mae']
    split = ["best", "random"]
    max_feat = ["auto", "sqrt", "log2"]
    # Boost parmeter
    n_estimator = list(range(200, 1000, 100))
    l_rate = 10.0 ** -np.arange(0, 20)
    loss = ['linear', 'square', 'exponential']
    # Bagging parameters
    max_samp = list(range(2, len_train_features, 1))
    max_feat_b = list(range(2, len_train_features, 1))
    boots = [False, True]
    boots_feat = [False, True]
    oob = [False, True]
   # warm = [False, True]
    # Gradient parameters
    loss_g = ['ls', 'lad', 'huber', 'quantile']
    crit_g = ['mse', 'friedman_mse', 'mae']
    max_depth = list(range(3, 40, 1))
    # Hist GradientBoostingRegresso

    loss_h = ["least_squares", "least_absolute_deviation", "poisson"]
    m_iter = list(range(200, 1000, 100))

    mss = random.choice(min_samples_s)
    msl = random.choice(min_samples_leaf)
    crt = random.choice(crit)
    spt = random.choice(split)
    mft = random.choice(max_feat)
    nest = random.choice(n_estimator)
    rate = random.choice(l_rate)
    closs = random.choice(loss)
    msb = random.choice(max_samp)
    mfb = random.choice(max_feat_b)
    bts = random.choice(boots)
    btf = random.choice(boots_feat)
    oobs = random.choice(oob)
    #wrms = random.choice(warm)
    lsg = random.choice(loss_g)
    mdt = random.choice(max_depth)
    l = random.choice(loss_h)
    m = random.choice(m_iter)

    tree = DecisionTreeRegressor(
        criterion=crt, splitter=spt, min_samples_split=mss, min_samples_leaf=msl, max_features=mft)
    if alg == "Boost":
        bdt = AdaBoostRegressor(base_estimator=tree, n_estimators=nest,
                                learning_rate=rate, loss=closs)
        return bdt
    if alg == "Bagging":
        bdt = BaggingRegressor(base_estimator=tree, n_estimators=nest, max_samples=msb,
                               max_features=mfb)  # warm_start=wrms)
        return bdt
    if alg == "Gradient":
        gbt = GradientBoostingRegressor(loss=lsg, learning_rate=rate, n_estimators=nest,
                                        criterion=crt, min_samples_split=mss, min_samples_leaf=msl, max_depth=mdt)
        return gbt
    if alg == "Hist":
        hist = HistGradientBoostingRegressor(
            loss=l, learning_rate=rate, max_iter=m, max_depth=mdt, l2_regularization=1e-5, min_samples_leaf=msl)
        return hist