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
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import Callback
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
date_cols = d00_config.date_cols
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
TRAIN_PHASE = 'validate'
TARGET_COLS = len(target_cols)
BATCH_SIZE = 1024
NB_EPOCH = 5
MODEL_VERSION = 'v2'

if TRAIN_PHASE == 'sample':
  TRN_SIZE = 1272204
  VLD_SIZE = 93166
  TRN_PRED_BATCH = 35339
  VLD_PRED_BATCH = 1259
elif TRAIN_PHASE == 'validate':
  TRN_SIZE = 12715856
  VLD_SIZE = 931453
  TRN_PRED_BATCH = 794741
  VLD_PRED_BATCH = 1637
elif TRAIN_PHASE == 'submission':
  TRN_SIZE = 13647309
TST_SIZE = 929615
TST_BATCH = 185923

##################################################`

LOG = get_logger('d01_nn_{}.txt'.format(TRAIN_PHASE))
LOG.info('# Training Neural Network (Phase : {})...'.format(TRAIN_PHASE))

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

def keras_model():
  model = Sequential()
  model.add(Dense(128, input_dim=FEAT_COUNT, init='he_uniform'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(TARGET_COLS, activation='softmax'))
  model.compile(loss='binary_crossentropy', optimizer='rmsprop')
  return model

def get_last_instance_df(trn):
  trn = '../Data/Raw/train_ver2.csv'
  last_instance_df = pd.read_csv(trn, usecols=['ncodpers']+target_cols, dtype=dtype_list)
  last_instance_df = last_instance_df.drop_duplicates('ncodpers', keep='last')
  last_instance_df = last_instance_df.fillna(0).astype('int')
  return last_instance_df

def batch_generator(file_name, batch_size, shuffle, state, train_input=True):
  while (True):
    if train_input:
      chunked_df = pd.read_csv(file_name, usecols=['ncodpers']+cols_to_use+numerical_cols+target_cols+date_cols, chunksize=batch_size)
    else:
      chunked_df = pd.read_csv(file_name, usecols=['ncodpers']+cols_to_use+numerical_cols+date_cols, chunksize=batch_size)

    # Feature engineering on spot
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

      chunk_X = chunk_df[date_cols]
      for ind, col in enumerate(date_cols):
        if col == 'fecha_dato':
          chunk_X['fecha_dat_y'] = pd.to_datetime(chunk_X[col]).dt.year # needs to be in ohe
          chunk_X['fecha_dat_m'] = pd.to_datetime(chunk_X[col]).dt.month # needs to be in ohe
        


      if train_input:
        y = np.array(chunk_df[target_cols].fillna(0))
          
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

def fit_model(trn, vld, tst, model):
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
      pickle_safe = True,
    )
    LOG.info('# Evaluating Binary XEntropy score...')
    LOG.info('## Fit History - Binary XEntropy\n    Train: {}\n    Valid: {}'.format(fit.history['loss'][-1], fit.history['val_loss'][-1]))
  elif TRAIN_PHASE == 'submission':
    fit = model.fit_generator(
      generator = batch_generator(trn, BATCH_SIZE, False, 'train'),
      nb_epoch = NB_EPOCH, 
      samples_per_epoch = TRN_SIZE,
      nb_worker = 8,
      pickle_safe = True
    )
    LOG.info('## Fit History - Binary XEntropy\n    Train: {}'.format(fit.history['loss'][0]))

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
  LOG.info('# Initialize Neural Net model')
  model = keras_model()

  # fit
  model = fit_model(trn, vld, tst, model)

  # submission
  LOG.info('# Predicting tst data with batch {} total {}' \
           .format(TST_BATCH, TST_SIZE)) 
  preds = model.predict_generator(
    generator = batch_generator(tst, TST_BATCH, False, 'test', False), 
    val_samples = TST_SIZE,
    nb_worker = 8,
    pickle_safe = True
  )
  
  # making submission
  LOG.info('# Making submission csv...')
  out_df = get_final_preds(trn, tst, preds)
  out_df.to_csv('../Output/Subm/submission_keras_{}_{}_epoch_{}.csv' \
                .format(MODEL_VERSION,TRAIN_PHASE,NB_EPOCH), index=False)

if __name__=='__main__':
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    main()
