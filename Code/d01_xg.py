# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file 
	- trains and validates sklearn model
"""

import numpy as np
np.random.seed(7)
import pandas as pd
import warnings

import xgboost as xgb
from sklearn.metrics import average_precision_score
from utils.log_utils import get_logger
from utils.eval_utils import eval_map7, get_pred_index
import d00_config
import d01_skl

##################################################`
# PARAMETER

# 'sample', 'validate', 'submission'
TRAIN_PHASE = 'sample'
REMOVE_ZERO = True
VERBOSE = False
N_ESTIMATORS = 100

xgb_pars = {
    #'booster': 'gblinear',
    'eta': 0.01,
    'gamma': 0.01,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1,
    'lambda': 1,
    'objective': 'binary:logistic',
    'eval_metrics': 'logloss',
    'nthread': 8,
    'seed': 7,
    'silent': 1
}
##################################################`

LOG = get_logger('d01_xg_{}-byclass.txt'.format(TRAIN_PHASE))
LOG.info('# Training XGBoost for EACH CLASS(Phase : {})...'.format(TRAIN_PHASE))

def fit_model(trn, vld):
  if TRAIN_PHASE == 'sample' or TRAIN_PHASE == 'validate':
    # load data
    trn = pd.read_csv(trn, dtype='int8')

    if REMOVE_ZERO:
      trn = trn[trn.iloc[:,55:].sum(axis=1) != 0]

    X_trn, y_trn = trn.iloc[:,:55], trn.iloc[:,55:].replace(-99,0)
    LOG.info('# Fitting model to trn data - X_trn: {} y_trn: {}'.format(X_trn.shape, y_trn.shape))
    LOG.info('# Memory Usage: X_trn: {}MB y_trn: {}MB' \
              .format(X_trn.memory_usage().sum()/(1024*1024), y_trn.memory_usage().sum()/(1024*1024)))

    vld = pd.read_csv(vld, dtype='int8')
    X_vld, y_vld = vld.iloc[:,:55], vld.iloc[:,55:]
    LOG.info('# Evaluating on vld data - X_vld: {} y_vld: {}'.format(X_vld.shape, y_vld.shape))
    LOG.info('# Memory Usage: X_vld: {}MB y_vld: {}MB' \
              .format(X_vld.memory_usage().sum()/(1024*1024), y_vld.memory_usage().sum()/(1024*1024)))

    models = []
    preds = []
    LOG.info('# Training XGBoost model..')
    for i in range(24):
      dtrain = xgb.DMatrix(X_trn, label=y_trn.iloc[:,i])
      dval = xgb.DMatrix(X_vld, label=y_vld.iloc[:,i])
      watchlist = [(dtrain,'train'),(dval,'val')]

      #### add early stopping
      model = xgb.train(xgb_pars, dtrain, \
                        early_stopping_rounds=5, \
                        num_boost_round=N_ESTIMATORS, evals=watchlist)
      preds.append(model.predict(dval))
    
    preds = np.asarray(preds).T.tolist()
    aps = 0.0
    for ind in range(vld.shape[0]):
      if sum(y_vld.values[ind]) == 0:
        aps += 0
      else:
        aps += average_precision_score(y_vld.values[ind], preds[ind], average='weighted')
    aps /= y_vld.shape[0]
    LOG.info('# Mean Average Precision: {}'.format(aps))

    map7 = eval_map7(y_vld.values, preds)
    LOG.info('# Mean Average Precision @ 7: {}'.format(map7))

  elif TRAIN_PHASE == 'submission':
    trn = pd.read_csv(trn)
    X_trn, y_trn = trn[:55], trn[55:]
    LOG.info('# Fitting model to trn data - X_trn: {} y_trn: {}'.format(X_trn.shape, y_trn.shape))
    model.fit(X_trn, y_trn)

  return model

def main():

  # get path
  trn, vld, tst = d01_skl.get_data_path(TRAIN_PHASE)

  # model
  LOG.info('# Initialize XGBoost model')

  # fit
  models = fit_model(trn, vld)
  a
  # submission
  LOG.info('# Predicting tst data...')
  X_tst = pd.read_csv(tst)
  preds = []
  for i in range(24):
    dtest = xgb.DMatrix(X_tst)
    preds.append(models[i].predict(X_tst))
  preds = np.asarray(preds).T.tolist()

  # making submission
  LOG.info('# Making submission csv...')
  out_df = d01_skl.get_final_preds(trn, tst, preds)
  out_df.to_csv('../Output/Subm/submission_rf_{}_{}.csv' \
                .format(MODEL_VERSION,TRAIN_PHASE), index=False)

if __name__=='__main__':
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    main()
