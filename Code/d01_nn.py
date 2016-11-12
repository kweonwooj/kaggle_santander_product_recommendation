# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file 
	- trains and validates sklearn model
"""

import numpy as np
np.random.seed(7)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
import warnings

from utils.log_utils import get_logger
from utils.eval_utils import eval_map7
import d00_config

##################################################`
# PARAMETER

# import from d00_config
cols_to_use = list(d00_config.mapping_dict.keys())
target_cols = d00_config.target_cols
numerical_cols = d00_config.numerical_cols
ohes = d00_config.ohes
mapping_dict = d00_config.mapping_dict
num_min_values = d00_config.num_min_values
num_range_values = d00_config.num_range_values
num_max_values = d00_config.num_max_values
dtype_list = d00_config.dtype_list

FEAT_COUNT = 0
for ohe in ohes:
  FEAT_COUNT += ohe.n_values_[0]
FEAT_COUNT += len(numerical_cols)

# 'sample', 'validate', 'submission'
TRAIN_PHASE = 'sample'
TARGET_COLS = len(target_cols)
BATCH_SIZE = 1024
NB_EPOCH = 1

if TRAIN_PHASE == 'sample':
  TRN_SIZE = 1272204
  VLD_SIZE = 93166
elif TRAIN_PHASE == 'validate':
  TRN_SIZE = 12715856
  VLD_SIZE = 931453
elif TRAIN_PHASE == 'submission':
  TRN_SIZE = 13647309
TST_SIZE = 929615

# Link : https://www.kaggle.com/sudalairajkumar/santander-product-recommendation/keras-starter-script/code
##################################################`

LOG = get_logger('d01_nn.txt')
LOG.info('# Training Neural Network (Phase : {})...'.format(TRAIN_PHASE))

def get_data_path():
  if TRAIN_PHASE == 'sample':
    trn = '../Data/Raw/sample_trn.csv'
    vld = '../Data/Raw/sample_vld.csv'
  elif TRAIN_PAHSE == 'validate':
    trn = '../Data/Raw/trn.csv'
    vld = '../Data/Raw/vld.csv'
  elif TRAIN_PHASE == 'submission':
    trn = '../Data/Raw/train_ver2.csv'
    vld = ''
  tst = '../Data/Raw/test_ver2.csv'
  return trn, vld, tst

def keras_model():
  model = Sequential()
  model.add(Dense(128, input_dim=FEAT_COUNT, init='he_uniform'))
  model.add(Activation('relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(TARGET_COLS, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='rmsprop')
  return model

def get_last_instance_df(trn):
  last_instance_df = pd.read_csv(trn, usecols=['ncodpers']+target_cols, dtype=dtype_list)
  last_instance_df = last_instance_df.drop_duplicates('ncodpers', keep='last')
  last_instance_df = last_instance_df.fillna(0).astype('int')
  return last_instance_df

def batch_generator(file_name, batch_size, shuffle, state, train_input=True):
  while (True):
    if train_input:
      chunked_df = pd.read_csv(file_name, usecols=['ncodpers']+cols_to_use+numerical_cols+target_cols, chunksize=batch_size)
    else:
      chunked_df = pd.read_csv(file_name, usecols=['ncodpers']+cols_to_use+numerical_cols, chunksize=batch_size)

    nrows = 0
    for chunk_df in chunked_df:
      chunk_X = chunk_df[cols_to_use]
      chunk_X = chunk_X.fillna(-99)
      for col_ind, col in enumerate(cols_to_use):
        chunk_X[col] = chunk_X[col].apply(lambda x: mapping_dict[col][x])
        ohe = ohes[col_ind]
        temp_X = ohe.transform( np.array(chunk_X[col]).reshape(-1,1) )
        if col_ind == 0:
          X = temp_X.todense().copy()
        else:
          X = np.hstack((X, temp_X.todense()))

      chunk_X = chunk_df[numerical_cols]
      for ind, col in enumerate(numerical_cols):
        if chunk_X[col].dtype == 'object':
          chunk_X[col] = chunk_X[col].map(str.strip).replace(['NA'], value=-1).fillna(-1).astype('float64')
        else:
          chunk_X[col] = chunk_X[col].fillna(-1).astype('float64')
        chunk_X[col] = (chunk_X[col] - num_min_values[ind]) / num_range_values[ind]
      chunk_X = np.array(chunk_X).astype('float64')
      X = np.hstack((X, chunk_X))

      if train_input:
        y = np.array(chunk_df[target_cols].fillna(0))

      if shuffle:
        shuffle_index = np.random.shuffle(np.arange(X.shape[0]))
        X = X[shuffle_index,:]
        if train_input:
          y = y[shuffle_index,:]

      if train_input:
        yield X, y
      else:
        yield X

      nrows += batch_size
      if train_input:
        if state == 'train' and nrows >= TRN_SIZE:
          break
        if state == 'valid' and nrows >= VLD_SIZE:
          break
      else:
        if state == 'test' and nrows >= TST_SIZE:
          break

def main():

  # get path
  trn, vld, tst = get_data_path()

  # model
  LOG.info('# Initialize Neural Net model')
  model = keras_model()

  # fit
  LOG.info('# Fitting model to trn data with batch {} total {}' \
             .format(BATCH_SIZE,TRN_SIZE))
  if TRAIN_PHASE == 'sample' or TRAIN_PHASE == 'validate':
    # fit 
    fit = model.fit_generator(
      generator = batch_generator(trn, BATCH_SIZE, False, 'train'),
      nb_epoch = NB_EPOCH, 
      samples_per_epoch = TRN_SIZE,
      validation_data = batch_generator(vld, BATCH_SIZE, False, 'valid'),
      nb_val_samples = VLD_SIZE,
      nb_worker = 8,
      pickle_safe = True
    )

    '''
    # get map7 accuracy for validation set
    vld_preds = model.predict_generator(
      generator = batch_generator(vld, BATCH_SIZE, False, 'test', False),
      val_samples = VLD_SIZE,
      nb_worker = 8,
      pickle_safe = True
    )
    vld_trues = pd.read_csv(vld, usecols=target_cols).values
    eval_map7(vld_trues, vld_preds)
    '''

  elif TRAIN_PHASE == 'submission':
    fit = model.fit_generator(
      generator = batch_generator(trn, BATCH_SIZE, False, 'train'),
      nb_epoch = NB_EPOCH, 
      samples_per_epoch = TRN_SIZE,
      nb_worker = 8,
      pickle_safe = True
    )

  # submission
  LOG.info('# Predicting tst data with batch {} total {}' \
           .format(5, TST_SIZE))
  preds = model.predict_generator(
    generator = batch_generator(tst, 5, False, 'test', False),
    val_samples = TST_SIZE,
    nb_worker = 8,
    pickle_safe = True
  )
  
  # making submission
  LOG.info('# Making submission csv...')
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
  out_df.to_csv('../Output/Subm/sub_keras__epoch_{}_{}.csv'.format(NB_EPOCH,TRAIN_PHASE), index=False)

if __name__=='__main__':
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    main()
