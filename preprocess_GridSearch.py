#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:35:33 2018

@author: Jianhua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import random
import math
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, preprocessing

from sklearn import preprocessing

import xgboost as xgb
import seaborn as sns

from scipy import sparse
from scipy.stats import skew, boxcox


def logregobj(labels, preds):
    con = 2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess 

def log_mae(y,yhat):
    return mean_absolute_error(np.exp(y), np.exp(yhat))

def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
    ##Grid Search for the best model
    model = GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring    = log_mae_scorer,
                                     verbose    = 10,
                                     n_jobs  = n_jobs,
                                     iid        = True,
                                     refit    = refit,
                                     cv      = cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("params:\n")
    print(model.cv_results_.__getitem__('params'))
    print("mean test scores:\n")
    print(model.cv_results_.__getitem__('mean_test_score'))
    print("std test scores:\n")
    print(model.cv_results_.__getitem__('std_test_score'))
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("Scores:", model.grid_scores_)
    print("**********************************************")
    
    return model

log_mae_scorer = metrics.make_scorer(log_mae, greater_is_better = False)

## Type your answers here ##
df_train = pd.read_csv("./input/train.csv")
df_test = pd.read_csv("./input/test.csv")

train_size = df_train.shape[0]

full_data = pd.concat([df_train, df_test]).reset_index(drop=True)
print ("Full Data set created.")

data_types = full_data.dtypes

cat_cols = list(data_types[data_types == 'object'].index)
cont_cols = list(data_types[(data_types == 'int64') | (data_types == 'float64')].index)

id_col = 'id'
loss_col = 'loss'
cont_cols.remove(id_col)
cont_cols.remove(loss_col)

LBE = preprocessing.LabelEncoder()
start=time.time()
LE_map=dict()
for cat_col in cat_cols:
    full_data[cat_col] = LBE.fit_transform(full_data[cat_col])
    LE_map[cat_col]=dict(zip(LBE.classes_, LBE.transform(LBE.classes_)))
print ('Label enconding processes in %f seconds' % (time.time() - start))

OHE = preprocessing.OneHotEncoder(sparse=True)
start = time.time()
full_data_sparse = OHE.fit_transform(full_data[cat_cols])
print('One-Hot encoding processes in %f seconds' % (time.time() - start))

skewed_cols = full_data[cont_cols].apply(lambda x: skew(x))
skelist = list(skewed_cols[(skewed_cols > 0.25) | (skewed_cols < -0.25)].index)

for skewed in skelist:
    full_data[skewed], lam = boxcox(full_data[skewed] + 1)
    
scaler = preprocessing.StandardScaler()
full_data[cont_cols] = scaler.fit_transform(full_data[cont_cols])

full_data_sparse = sparse.hstack((full_data_sparse,full_data[cont_cols]), format='csr')
print (full_data_sparse.shape)

shift = 200
full_cols = cat_cols + cont_cols
train_x = full_data_sparse[:train_size]
test_x = full_data_sparse[train_size:]
train_y = np.log(full_data[:train_size].loss.values + shift)
ID = full_data.id[:train_size].values


param_grid = {'objective':[logregobj],
              'learning_rate':[0.001, 0.01, 0.02, 0.05, 0.1],
              'n_estimators':[1500],
              'max_depth': [9],
              'min_child_weight':[50],
              'subsample': [0.78],
              'colsample_bytree':[0.67],
              'gamma':[0.9],
              'nthread': [-1],
              'seed' : [1234]}

model = search_model(train_x,
                     train_y,
                     xgb.XGBRegressor(),
                     param_grid,
                     n_jobs = 1,
                     cv = 4,
                     refit = True)
