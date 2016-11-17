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

from utils.eval_utils import eval_map7, get_pred_index
import d00_config

def get_data_path(TRAIN_PHASE):
  if TRAIN_PHASE == 'sample':
    trn = '../Data/Raw/sample_trn.csv'
    vld = '../Data/Raw/sample_vld.csv'
  elif TRAIN_PHASE == 'validate':
    trn = '../Data/Raw/validate_trn.csv'
    vld = '../Data/Raw/validate_vld.csv'
  elif TRAIN_PHASE == 'submission':
    trn = '../Data/Raw/submission_trn.csv'
    vld = ''
  tst = '../Data/Raw/submission_vld.csv'
  return trn, vld, tst

def get_last_instance_df(trn):
  trn = '../Data/Raw/train_ver2.csv'
  last_instance_df = pd.read_csv(trn, usecols=['ncodpers']+target_cols, dtype=dtype_list)
  last_instance_df = last_instance_df.drop_duplicates('ncodpers', keep='last')
  last_instance_df = last_instance_df.fillna(0).astype('int')
  return last_instance_df

def get_final_preds(trn, tst, preds):
  last_instance_df = get_last_instance_df(trn)
  cust_dict = dict()
  target_cols = np.array(d00_config.target_cols)
  for ind, row in last_instance_df.iterrows():
    cust = row['ncodpers']
    used_products = set(target_cols[np.array(row[1:])==1])
    cust_dict[cust] = used_products
  del last_instance_df

  #### add overallbest if confidence is low
  preds = np.argsort(preds, axis=1)
  preds = np.fliplr(preds)
  test_id = np.array(pd.read_csv(tst, usecols=['ncodpers'])['ncodpers'])
  final_preds = []
  for ind, pred in enumerate(preds):
    pred = get_pred_index(pred)
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

