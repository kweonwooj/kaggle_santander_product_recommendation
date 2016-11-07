# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file saves 
    - unique values of each feature column
    - unique values of each feature column per ncodpers (valid and train)
"""

import pandas as pd
import pickle

# import data
trn = pd.read_csv('../Data/Raw/train_ver2.csv')

# unique values of each feature column
for col in trn.columns:
  print 'Storing column: {}'.format(col)
  # get unique 
  vld_period = '2016-05-28'

  unique_trn = trn[col].unique()
  unique_vld = trn[trn.fecha_dato != vld_period][col].unique()

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


# unique values of each feature column per user
users_trn = trn.ncodpers.unique()
users_vld = trn[trn.fecha_dato != 

valid
for user in users:
  # 
