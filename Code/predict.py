# -*- encoding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief:
	input: trn_x, tst_x, trn_y
	output: show evaluation score
"""

import numpy as np
import pandas as pd
from config import state, model
from utils.log_utils import get_logger
from utils.eval_utils import eval_map7
import time
from sklearn.cross_validation import StratifiedShuffleSplit

## define target_cols

st = time.time()

LOG = get_logger('predict_{}.txt'.format(state))
LOG.info('# Fitting and Predicting {}'.format(state))

data_path = '../Data/Raw/'
df = pd.read_csv('{}trn_{}.csv'.format(data_path, state))
LOG.info('# df\t{}'.format(df.shape))

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

## convert tst
feature_cols = ['ncodpers','age','ind_nuevo','antiguedad']
tst = pd.read_csv('../Data/Raw/test_ver2.csv', usecols=feature_cols)

f = open('../Data/Raw/tst_v1.csv', 'w')
out = ','.join(feature_cols)+'\n'
f.write(out)

check_point = tst.shape[0]/10
for ind, (run, row) in enumerate(tst.iterrows()):
	row_str = [str(val).strip() for val in tst.ix[run].values]
	out = ','.join(row_str) + '\n'
	f.write(out)
	
	if ind % check_point == 0:
		LOG.info('# Processing {} lines..'.format(ind))
LOG.info('# Processed total of {} lines.'.format(ind))
f.close()

tst = pd.read_csv('../Data/Raw/tst_v1.csv')

model.fit(x,y)
preds = model.predict_proba(tst)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv('../Data/Raw/test_ver2.csv', usecols=['ncodpers'])['ncodpers'])
final_preds = [' '.join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers': test_id, 'added_products':final_preds})
out_df.to_csv('../Output/Subm/kweonwooj_v1.csv', index=False)	

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
LOG.info('# DONE!')


