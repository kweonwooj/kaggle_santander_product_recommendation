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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from utils.log_utils import get_logger
from utils.eval_utils import eval_map7
import d00_config

##################################################`
# PARAMETER

# 'sample', 'validate', 'submission'
TRAIN_PHASE = 'sample'

##################################################`

LOG = get_logger('d01_rf_{}.txt'.format(TRAIN_PHASE))
LOG.info('# Training Random Forest (Phase : {})...'.format(TRAIN_PHASE))

def get_data_path():
  if TRAIN_PHASE == 'sample':
    trn = '../Data/Raw/sample_trn.csv'
    vld = '../Data/Raw/sample_vld.csv'
  elif TRAIN_PHASE == 'validate':
    trn = '../Data/Raw/trn.csv'
    vld = '../Data/Raw/vld.csv'
  elif TRAIN_PHASE == 'submission':
    trn = '../Data/Raw/train_ver3.csv'
    vld = ''
  tst = '../Data/Raw/test_ver2.csv'
  return trn, vld, tst

def get_last_instance_df(trn):
  trn = '../Data/Raw/train_ver2.csv'
  last_instance_df = pd.read_csv(trn, usecols=['ncodpers']+target_cols, dtype=dtype_list)
  last_instance_df = last_instance_df.drop_duplicates('ncodpers', keep='last')
  last_instance_df = last_instance_df.fillna(0).astype('int')
  return last_instance_df

def fit_model(trn, vld, model):
  if TRAIN_PHASE == 'sample' or TRAIN_PHASE == 'validate':
    # load data
    trn = pd.read_csv(trn)
    X_trn, y_trn = trn.iloc[:,:55], trn.iloc[:,55:]
    LOG.info('# Fitting model to trn data\n# X_trn: {} y_trn: {}'.format(X_trn.shape, y_trn.shape))

    # fit 
    model.fit(X_trn, y_trn)
    del trn, X_trn, y_trn

    LOG.info('# Evaluating vld set...')
    vld = pd.read_csv(vld)
    X_vld, y_vld = vld.iloc[:,:55], vld.iloc[:,55:]
    LOG.info('# Evaluating on vld data\n# X_vld: {} y_vld: {}'.format(X_vld.shape, y_vld.shape))

    preds = model.predict_proba(X_vld) # need to reshape this
    preds = [1-pred[:,0] for pred in preds]
    preds = np.asarray(preds).T.tolist()
    aps = average_precision_score(y_vld.values, preds, average='weighted')
    LOG.info('# Mean Average Precision: {}'.format(aps))

    map7 = eval_map7(y_vld.values, preds)
    LOG.info('# Mean Average Precision @ 7: {}'.format(map7))

  elif TRAIN_PHASE == 'submission':
    trn = pd.read_csv(trn)
    X_trn, y_trn = trn[:55], trn[55:]
    LOG.info('# Fitting model to trn data\n# X_trn: {} y_trn: {}'.format(X_trn.shape, y_trn.shape))
    model.fit(X_trn, y_trn)

  return model

def get_final_preds(trn, tst, preds):
  last_instance_df = get_last_instance_df(trn)
  cust_dict = dict()
  target_cols = np.array(d00_config.target_cols)
  for ind, row in last_instance_df.iterrows():
    cust = row['ncodpers']
    used_products = set(target_cols[np.array(row[1:])==1])
    cust_dict[cust] = used_products
  del last_instance_df

  preds = np.argsort(preds, axis=1)
  preds = np.fliplr(preds)
  test_id = np.array(pd.read_csv(tst, usecols=['ncodpers'])['ncodpers'])
  final_preds = []
  for ind, pred in enumerate(preds):
    cust = test_id[ind]
    top_products = target_cols[pred]
    used_products = cust_dict.get(cust,[])
    new_top_products = []
    for product in top_products:
      if product not in used_products:
        new_top_products.append(product)
      if len(new_top_products) == 7:
        break
    final_preds.append(' '.join(new_top_products))
  out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
  return out_df

def main():

  # get path
  trn, vld, tst = get_data_path()

  # model
  LOG.info('# Initialize RandomForest model')
  model = RandomForestClassifier(
		n_estimators = 1000,
		n_jobs=-1, random_state=7)

  # fit
  model = fit_model(trn, vld, model)
  a
  # submission
  LOG.info('# Predicting tst data...')
  X_tst = pd.read_csv(tst)
  preds = model.predict_proba(X_tst)

  # making submission
  LOG.info('# Making submission csv...')
  out_df = get_final_preds(trn, tst, preds)
  out_df.to_csv('../Output/Subm/submission_rf_{}_{}.csv' \
                .format(MODEL_VERSION,TRAIN_PHASE), index=False)

if __name__=='__main__':
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    main()
