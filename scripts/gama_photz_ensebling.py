import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import healpy as hp
import os,sys
import matplotlib
import seaborn as sns
import time

# Regression Libraries
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor,StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb

# Other Libraries
from sklearn.decomposition import PCA
import category_encoders as ce
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split,ShuffleSplit
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import norm, skew #for some statistics

from mlxtend.regressor import StackingCVRegressor

home = os.getenv("HOME")

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

vipers = pd.read_csv("/home/rafael/Projetos/PHOTOzxcorr/data/gama_ps1.csv")

#ml.clean_tab(vipers,"Z",1.5)

X, y = ml.get_features_targets_gama1(vipers)
rob1 = StandardScaler()
rob2 = StandardScaler()

X = rob1.fit_transform(X)
y = rob2.fit_transform(y.reshape(-1,1))


X_train, X_test, y_train, y_test = ml.tts_split(X, y, 0.3, 5)

stack = StackingCVRegressor(regressors=(models[0:8]),
                            meta_regressor=models[9],
                            cv = 5 , n_jobs=-1,use_features_in_secondary=True)

stack.fit(X_train,y_train.ravel())

pred_1 = stack.predict(X_test)
print(" Stacked averaged base models score: ROOT MAE {:5.2f} MAE ({:5.2f})\n".format(ml.rmsle(pred_1,y_test), mean_squared_error(pred_1,y_test)))


test_predictions = pred_1

plt.figure(figsize=(16, 8))
plt.scatter(y_test, test_predictions, s=1, c="k")
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100, 100], [-100, 100],color = "red")
plt.savefig("plots/stack_ann_gama.png")
plt.show()
plt.close()

plt.figure(figsize=(16, 8))
plt.hist(test_predictions, bins=50)
plt.xlabel('Predictions')
plt.ylabel('Frequency')
plt.savefig("plots/stack_hist_gama.png")
plt.show()
plt.close()
