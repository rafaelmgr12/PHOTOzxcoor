import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
import matplotlib


# Regression Libraries
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge

# Other Libraries
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error,mean_squared_log_error



from mlxtend.regressor import StackingCVRegressor

home = os.getenv("HOME")
sys.path.append(home+"/Projetos/PHOTOzxcorr/functions/")


import ml_algorithims as ml

models = [KNeighborsRegressor(n_neighbors=9),
 SVR(C=10, epsilon=0.2, gamma=1e-07, kernel='linear'),
 SGDRegressor(alpha=1e-06, penalty='elasticnet'),
 RandomForestRegressor(max_depth=18, min_samples_split=34),
 DecisionTreeRegressor(criterion='mae', max_depth=18, min_samples_split=36,
                       splitter='random'),
 Lasso(max_iter=200),
 KernelRidge(alpha=1e-08, degree=1),
 GradientBoostingRegressor(criterion='mae', loss='lad', max_features='sqrt',
                           n_estimators=500),
AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='mae',
                                                        max_depth=18,
                                                        min_samples_split=36,
                                                        splitter='random'),
                   learning_rate=0.01, loss='exponential', n_estimators=400),
 BaggingRegressor(base_estimator=DecisionTreeRegressor(criterion='mae',
                                                       max_depth=18,
                                                       min_samples_split=36,
                                                       splitter='random'),
                  n_estimators=300, n_jobs=-1)]

# user here the path where we download the folder DESzxcoorr
sys.path.append(home+"/Projetos/PHOTOzxcorr/functions/")

print('\n',"-"*100)
print("\nReading data")

vipers = pd.read_csv("/home/rafael/Projetos/PHOTOzxcorr/data/viper_clean.csv")

print('\n',"-"*100)
print("Models that we will be used is :\n")
print(models)

#ml.clean_tab(vipers,"Z",1.5)


X, y = ml.get_features_targets_des2(vipers)
rob1 = StandardScaler()
rob2 = StandardScaler()

X = rob1.fit_transform(X)
y = rob2.fit_transform(y.reshape(-1,1))


X_train, X_test, y_train, y_test = ml.tts_split(X, y, 0.3, 5)

BATCH_SIZE = 64
STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def build_nn():

    ann_model = Sequential([Dense(n_inputs,input_shape = X_train.shape[1:],kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)),
                       Dense(10, kernel_initializer='normal',  kernel_constraint=max_norm(2.5) ,activation='tanh',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)),
                       BatchNormalization(),
                       Dense(10,kernel_initializer='normal', kernel_constraint=max_norm(2.5) ,activation='tanh',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)) ,
                       Dense(1,activation = None,name = "output")
                       ])
    opt = ks.optimizers.RMSprop(lr_schedule)
    #opt = tf.keras.optimizers.RMSprop(0.001)
    ann_model.compile(optimizer=opt, loss=rmse_ann3, metrics=['mse', 'mae', 'mape',rmse_ann3])
    
    return ann_model

def build_pnn():
    pnn_model = Sequential([Dense(n_inputs,input_shape = X_train.shape[1:],kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)),
                    BatchNormalization(),
                     tfpl.DenseVariational(units=10,
                          make_prior_fn=prior2,
                          make_posterior_fn=posterior2,
                          kl_weight=1/X_train.shape[0],activation =  "tanh"),
                    tfpl.DenseVariational(units=10,
                          make_prior_fn=prior2,
                          make_posterior_fn=posterior2,
                          kl_weight=1/X_train.shape[0],activation =  "tanh"),
                    
                    tfpl.DenseReparameterization( units = tfpl.IndependentNormal.params_size(1), activation=None,                                                
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn,
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn ),
                                                
                    tfpl.IndependentNormal(1)
                    
                      ])
    opt = ks.optimizers.RMSprop(lr_schedule)
    pnn_model.compile(optimizer=opt, loss=nll2, metrics=['mse', 'mae', 'mape',rmse_ann4])
    return pnn_model
    
    


pnn1_clf = tf.keras.wrappers.scikit_learn.KerasRegressor(
                            build_pnn,
                            epochs=20,batch_size=64,validation_split=0.2,
                            verbose=False)

ann1_clf = tf.keras.wrappers.scikit_learn.KerasRegressor(
                            build_nn,
                            epochs=10,batch_size=64,validation_split=0.2,
                            verbose=False)
ann1_clf._estimator_type = 'regressor'

pnn1_clf._estimator_type = 'regressor'


stack = StackingCVRegressor(regressors=(models[8],models[0],models[6],models[7], pnn1_clf,ann1_clf,models[2]),
                            meta_regressor=models[9],
                            use_features_in_secondary=True,cv = 5,n_jobs = -1)

print("\n Starting the fit, may taking a while")
stack.fit(X_train,y_train.ravel())


pred_1 = stack.predict(X_test)
print("\n Stacked base models score: ROOT MAE {:5.2f} MAE ({:5.2f})\n".format(ml.rmsle(pred_1,y_test), mean_squared_error(pred_1,y_test)))


test_predictions = pred_1

plt.figure(figsize=(16, 8))
plt.scatter(y_test, test_predictions, s=1, c="k")
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100],color = "red")
plt.savefig("../plots/stack_ann_vipers.png")
plt.show()
plt.close()

plt.figure(figsize=(16, 8))
plt.hist(test_predictions, bins=50)
plt.xlabel('Predictions')
plt.ylabel('Frequency')
plt.savefig("../plots/stack_hist_vipers.png")
plt.show()
plt.close()
