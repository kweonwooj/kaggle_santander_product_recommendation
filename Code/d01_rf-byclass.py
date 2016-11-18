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

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

##################################################`

LOG = get_logger('d01_rf_{}-byclass.txt'.format(TRAIN_PHASE))
LOG.info('# Training Random Forest for EACH CLASS (Phase : {})...'.format(TRAIN_PHASE))

def fit_model(trn, vld, model):
  if TRAIN_PHASE == 'sample' or TRAIN_PHASE == 'validate':
    # load data
    trn = pd.read_csv(trn, dtype='int8')

    if REMOVE_ZERO:
      trn = trn[trn.iloc[:,55:].sum(axis=1) != 0]

    X_trn, y_trn = trn.iloc[:,:55], trn.iloc[:,55:]
    LOG.info('# Fitting model to trn data - X_trn: {} y_trn: {}'.format(X_trn.shape, y_trn.shape))
    LOG.info('# Memory Usage: X_trn: {}MB y_trn: {}MB' \
              .format(X_trn.memory_usage().sum()/(1024*1024), y_trn.memory_usage().sum()/(1024*1024)))
    LOG.info('# Evaluating vld set...')
    vld = pd.read_csv(vld, dtype='int8')
    X_vld, y_vld = vld.iloc[:,:55], vld.iloc[:,55:]
    LOG.info('# Evaluating on vld data - X_vld: {} y_vld: {}'.format(X_vld.shape, y_vld.shape))
    LOG.info('# Memory Usage: X_vld: {}MB y_vld: {}MB' \
              .format(X_vld.memory_usage().sum()/(1024*1024), y_vld.memory_usage().sum()/(1024*1024)))

    models = []
    preds = []
    for i in range(24):
      # fit 
      model.fit(X_trn, y_trn.iloc[:,i])
      preds.append(model.predict_proba(X_vld)[:,1])
      models.append(model)

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

  return models

def main():

  # get path
  trn, vld, tst = d01_skl.get_data_path(TRAIN_PHASE)

  print trn,vld,tst

  # model
  LOG.info('# Initialize RandomForest model')
  model = RandomForestClassifier(
		n_estimators = 10,
		n_jobs=-1, random_state=7)

  # fit
  models = fit_model(trn, vld, model)
  a
  # submission
  LOG.info('# Predicting tst data...')
  X_tst = pd.read_csv(tst, dtype='int8')
  preds = []
  for i in range(24):
    preds.append(models[i].predict_proba(X_vld)[:,1])
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
