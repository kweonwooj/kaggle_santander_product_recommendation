# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file saves 
    - unique values of each feature column
    - unique values of each feature column per ncodpers (valid and train)
"""

import pandas as pd
import pickle
from utils.log_utils import get_logger

LOG = get_logger('preprocess.txt')

# import data
trn = pd.read_csv('../Data/Raw/train_ver2.csv')
vld_period = '2016-05-28'
vld = trn[trn.fecha_dato != vld_period].copy()
"""
# unique values of each feature column
for col in trn.columns:
  LOG.info('Storing column: {}'.format(col))
  # get unique 

  unique_trn = trn[col].unique()
  unique_vld = vld[col].unique()

  # store in dict
  feat_trn = dict()
  feat_vld = dict()
  # get number of unique elements
  feat_trn['len'] = len(unique_trn)
  feat_trn['elements'] = unique_trn
  feat_vld['len'] = len(unique_vld)
  feat_vld['elements'] = unique_vld

  # store in pkl
  pickle.dump(feat_trn, open('../Data/Clean/unq_{}.pkl'.format(col),'wb'))
  pickle.dump(feat_vld, open('../Data/Clean/unq_{}_vld.pkl'.format(col),'wb'))
"""

# unique values of each feature column per user
users_trn = trn.ncodpers.unique()
users_vld = vld.ncodpers.unique()

do_cols = ['antiguedad','indrel','indrel_1mes','tiprel_1mes','indresi', \
           'indext','canal_entrada','indfall','cod_prov','nomprov', \
           'ind_actividad_cliente','segmentp']

for col in trn.columns:
  if col not in do_cols: continue
  LOG.info('Storing column for per user: {}'.format(col))
  feat = dict()

  LOG.info('users_trn : {}'.format(len(users_trn)))
  for i, user in enumerate(users_trn):
    # get unique elements per user
    feat[user] = {'elements':trn[trn.ncodpers == user][col].unique()}
    if i % 100000 == 0:
      LOG.info('trn_iteration : {}'.format(i))

  # store in pkl
  pickle.dump(feat, open('../Data/Clean/unq_{}_per_user.pkl'.format(col),'wb'))
  
  feat = dict()
  LOG.info('users_vld : {}'.format(len(users_vld)))
  for i, user in enumerate(users_vld):
    # get unique elements per user
    feat[user] = {'elements':vld[vld.ncodpers == user][col].unique()}
    if i % 100000 == 0:
      LOG.info('vld_iteration : {}'.format(i))
    
  # store in pkl
  pickle.dump(feat, open('../Data/Clean/unq_{}_per_user_vld.pkl'.format(col),'wb'))
  
    




