import pandas as pd
import numpy as np
import sys
import os
import metrics
import ml_algorithims as ml
import matplotlib.pyplot as plt
# Neural Network Libs
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization,Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import confusion_matrix
import keras as ks
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers
from tensorflow.keras import regularizers


from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor

home = os.getenv("HOME")

# user here the path where we download the folder DESzxcoorr
sys.path.append(home+"/Projetos/PHOTOzxcorr/functions/")

vipers = pd.read_csv("/home/rafael/Projetos/PHOTOzxcorr/data/viper_clean.csv")

X, y = ml.get_features_targets_des2(vipers)

X_train, X_test, y_train, y_test = ml.tts_split(X, y, 0.3, 5)

def rmse_ann(y_true, y_pred):
    diff = keras.backend.square((y_pred - y_true) / (keras.backend.abs(y_true) + 1))
    return keras.backend.sqrt(keras.backend.mean(diff))x  

EarlyStop = EarlyStopping(monitor='rmse_ann', mode='min', patience=10)
n_inputs = X_train.shape[1]

BATCH_SIZE = 64
STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False)

ann_model = Sequential([Dense(n_inputs, input_shape=X_train.shape[1:], kernel_initializer=tf.keras.initializers.Ones(), kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),

                        Dense(8, kernel_initializer='normal',  kernel_constraint=max_norm(2.), activation='tanh', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
                        BatchNormalization(),
                        
                        Dense(8, kernel_initializer='normal', kernel_constraint=max_norm(2.), activation='tanh', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)),
                        
                        Dense(1, activation=None, name="output")
                        ])
opt = ks.optimizers.RMSprop(lr_schedule)
#opt = tf.keras.optimizers.RMSprop(0.001)
ann_model.compile(optimizer=opt, loss=rmse_ann, metrics=[
                  'mse', 'mae', 'mape', rmse_ann])

history = ann_model.fit(X_train, y_train, epochs=256,
                        batch_size=64, validation_split=0.2, callbacks=[EarlyStop])

ml.plot_history(history)

loss, mae, mse, mape, rmse_ann = ann_model.evaluate(X_test, y_test)

print("Testing set Mean Abs Error: {:5.3f} ".format(mae))
print("\n")
print("Testing set Root Mean Abs Error: {:5.3f} ".format(rmse_ann))

test_predictions = ann_model.predict(X_test).flatten()

plt.figure(figsize=(16, 8))
plt.scatter(y_test, test_predictions, s=1, c="k")
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100],color = "red")
plt.show()
plt.savefig("plots/scatter_ann.png")
plt.close()

plt.figure(figsize=(16, 8))
plt.hist(test_predictions, bins=50)
plt.xlabel('Predictions')
plt.ylabel('Frequency')
plt.show()
plt.savefig("plots/ann_hist.png")
plt.close()
