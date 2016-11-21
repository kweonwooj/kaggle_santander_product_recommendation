# -*- encoding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief:
	input : raw trn, label_sp
	output : only select buyers at May 2016.
		 feature engineer
		 convert into single multiclass classification
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.log_utils import get_logger
import time
import pickle

st = time.time()

LOG = get_logger('preprocess.txt')
LOG.info('# Begin Preprocessing')

## June 2015 only
st_index = 3144384
en_index = 3776493
## define label_cols (same as target_cols)

LOG.info('# Select June 2015 purchase only')
# select ncodpers that purchased on June 2015 only
labels = pd.read_csv('../Data/Raw/labels.csv', header=None)
buy_flag = labels.iloc[st_index:en_index,:].sum(axis=1)
ncodpers = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=['ncodpers'])
buy_ncodpers = ncodpers.iloc[st_index:en_index][buy_flag > 0]
pickle.dump(buy_ncodpers, open('../Data/Raw/buy_ncodpers.pkl','wb'))
index = ncodpers.isin(buy_ncodpers).values
del ncodpers

# use indexed rows
feature_cols = ['ncodpers','age','ind_nuevo','antiguedad']
trn = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=feature_cols)
trn = trn[index]
labels = labels[index]

LOG.info('# Feature engineering..')
# feature engineer
## None for initial 

# convert to multiclass classification
f = open('../Data/Raw/trn_v1.csv','w')
out = ','.join(feature_cols) + ',label\n' 
f.write(out)

check_point = labels.shape[0]/10
for ind, (run, row) in enumerate(labels.iterrows()):
	for i,r in enumerate(row):
		if r == 1:
			## put modularized feature engineerer, reusable at tst time
			row_str = [str(val).strip() for val in trn.ix[run].values]
			out = ','.join(row_str) + ',' + str(i) + '\n'
			f.write(out)
	if ind % check_point == 0:
		LOG.info('# Processing {} lines..'.format(ind))

LOG.info('# Processed total of {} lines.'.format(ind))
f.close()

en = time.time()
el = en-st
unit = 'sec'
if el > 60:
	el /= 60.
	unit = 'min'
if el > 60:
	el /= 60.
	unit = 'hour'
LOG.info('# Elapsed time : {} {}'.format(el, unit))
LOG.info('# DONE')


