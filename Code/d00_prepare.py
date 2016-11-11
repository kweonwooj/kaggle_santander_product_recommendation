# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file generates
	- sample_trn, sample_vld that is 1/10 of ncodpers
	- trn, vld
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from utils.log_utils import get_logger

LOG = get_logger('d00_prepare.txt')
LOG.info('# Generating sample_trn and sample_vld using 1/10 of ncodpers...')

def get_df_idx(df, col, values):
  return np.where(df[col].isin(values))[0]

def main():

  # import data
  path = '../Data/Raw/'
  df = pd.read_csv(path+'train_ver2.csv')

  ## ORIGINAL ##
  LOG.info('# Serializing original into trn, vld')
  # split into trn,vld
  trn = df[df.fecha_dato != '2016-05-28']
  vld = df[df.fecha_dato == '2016-05-28']

  # serialize
  trn.to_csv(path+'trn.csv', index=False)
  vld.to_csv(path+'vld.csv', index=False)
  del trn
  del vld
  LOG.info('# DONE! Serialized original into trn, vld')
  

  ## SAMPLE ##
  LOG.info('# Serializing sample into trn, vld')
  # split ncodeprs into 1/10
  ncodpers = df.ncodpers.unique()
  _, X_sample, _, _ = train_test_split(ncodpers, range(ncodpers.shape[0]), \
                                       test_size=0.1, random_state=7)

  # get sample sets
  index = get_df_idx(df, 'ncodpers', X_sample)
  sample = df.iloc[index]

  # split into trn,vld
  sample_trn = sample[sample.fecha_dato != '2016-05-28']
  sample_vld = sample[sample.fecha_dato == '2016-05-28']

  # serialize
  sample_trn.to_csv(path+'sample_trn.csv', index=False)
  sample_vld.to_csv(path+'sample_vld.csv', index=False)
  LOG.info('# DONE! Serialized sample into trn, vld')

if __name__=='__main__':
  main()
