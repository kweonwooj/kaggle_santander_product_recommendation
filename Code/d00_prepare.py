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

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

def main():

  # import data
  path = '../Data/Raw/'
  df = pd.read_csv(path+'train_ver2.csv')

  ## ORIGINAL ##
  LOG.info('# Serializing original into trn, vld')
  # split into trn,vld
  trn = df[df.fecha_dato != '2016-05-28']
  vld = df[df.fecha_dato == '2016-05-28']

  cust = dict()
  targets = []
  for ind, (run, row) in enumerate(trn.iterrows()):
    ncodper = row['ncodpers']
    if ncodper in cust:
      temp = np.clip(row[24:].values - cust[ncodper], 0, 1)
      cust[ncodper] = row[24:].values
      targets.append(temp)
    else:
      cust[ncodper] = row[24:].values
      targets.append(row[24:].values)

  targets_vld = []
  for ind, (run, row) in enumerate(vld.iterrows()):
    ncodper = row['ncodpers']
    if ncodper in cust:
      temp = np.clip(row[24:].values - cust[ncodper], 0, 1)
      cust[ncodper] = row[24:].values
      targets_vld.append(temp)
    else:
      cust[ncodper] = row[24:].values
      targets_vld.append(row[24:].values)

  trn.reset_index(drop=True, inplace=True)
  trn[target_cols] = pd.DataFrame(targets, columns=target_cols)
  vld.reset_index(drop=True, inplace=True)
  vld[target_cols] = pd.DataFrame(targets_vld, columns=target_cols)

  # serialize
  trn.to_csv(path+'trn.csv', index=False)
  vld.to_csv(path+'vld.csv', index=False)
  del trn
  del vld
  LOG.info('# DONE! Serialized original into trn, vld')
  
  '''
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

  cust = dict()
  targets = []
  for ind, (run, row) in enumerate(sample_trn.iterrows()):
    ncodper = row['ncodpers']
    if ncodper in cust:
      temp = np.clip(row[24:].values - cust[ncodper], 0, 1)
      cust[ncodper] = row[24:].values
      targets.append(temp)
    else:
      cust[ncodper] = row[24:].values
      targets.append(row[24:].values)
        
  targets_vld = []
  for ind, (run, row) in enumerate(sample_vld.iterrows()):
    ncodper = row['ncodpers']
    if ncodper in cust:
      temp = np.clip(row[24:].values - cust[ncodper], 0, 1)
      cust[ncodper] = row[24:].values
      targets_vld.append(temp)
    else:
      cust[ncodper] = row[24:].values
      targets_vld.append(row[24:].values)
  
  sample_trn.reset_index(drop=True, inplace=True)
  sample_trn[target_cols] = pd.DataFrame(targets, columns=target_cols)
  sample_vld.reset_index(drop=True, inplace=True)
  sample_vld[target_cols] = pd.DataFrame(targets_vld, columns=target_cols)

  # serialize
  sample_trn.to_csv(path+'sample_trn.csv', index=False)
  sample_vld.to_csv(path+'sample_vld.csv', index=False)
  LOG.info('# DONE! Serialized sample into trn, vld')
  '''

  LOG.info('# Serializing original into train_ver3.csv')
  cust = dict()
  targets = []
  for ind, (run, row) in enumerate(df.iterrows()):
    ncodper = row['ncodpers']
    if ncodper in cust:
      temp = np.clip(row[24:].values - cust[ncodper], 0, 1)
      cust[ncodper] = row[24:].values
      targets.append(temp)
    else:
      cust[ncodper] = row[24:].values
      targets.append(row[24:].values)

  df.reset_index(drop=True, inplace=True)
  df[target_cols] = pd.DataFrame(targets, columns=target_cols)
  
  df.to_csv(path+'train_ver3.csv', index=False)
  LOG.info('# DONE! Serialized original into train_ver3.csv')

if __name__=='__main__':
  main()
