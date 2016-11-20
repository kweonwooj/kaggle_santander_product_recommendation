# -*- encoding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief:
	input : raw trn, raw tst
	output : preprocessed trn, tst
"""
from config import state, dtype_list, target_cols, initial_run
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.log_utils import get_logger
import time
from config import generate_label_sp

st = time.time()

feature_cols = ["ind_empleado","pais_residencia","sexo","age", "ind_nuevo", "antiguedad", "nomprov", "segmento"]

LOG = get_logger('preprocess_{}.txt'.format(state))
LOG.info('# Begin Preprocessing _ {}'.format(state))

# initialize logger

data_path = '../Data/Raw/'
trn_file = data_path + 'train_ver2.csv'
tst_file = data_path + 'test_ver2.csv'

trn_size = 13647309
nrows =    13647309
start_index = trn_size - nrows
LOG.info('# Preprocessing {} rows / {} : {}'.format(nrows, trn_size, round(1.*nrows/trn_size*100,3)))

# get trn_sp_y

# get ncodpers that has purchased something in May only

colnames = []
for ind, col in enumerate(feature_cols):
        LOG.info('# Preprocessing column: {} | {}/{}'.format(col, ind+1, len(feature_cols)))
	trn = pd.read_csv(trn_file, usecols=[col])
	tst = pd.read_csv(tst_file, usecols=[col])

	trn.fillna(-99,inplace=True)
	tst.fillna(-99,inplace=True)

	## ADD ## 
	# special preprocessing for each columns
	# additional feature engineering possible
	colnames.append(col)

	if trn[col].dtype == 'object':
		le = LabelEncoder()
		le.fit(list(trn[col].values)+list(tst[col].values))
		temp_trn_x = le.transform(list(trn[col].values)).reshape(-1,1)[start_index:,:]
		temp_tst_x = le.transform(list(tst[col].values)).reshape(-1,1)
	else:
		temp_trn_x = np.array(trn[col]).reshape(-1,1)[start_index:,:]
		temp_tst_x = np.array(tst[col]).reshape(-1,1)

	if ind == 0:
		trn_x = temp_trn_x.copy()
		tst_x = temp_tst_x.copy()
	else:
		trn_x = np.hstack([trn_x, temp_trn_x])
		tst_x = np.hstack([tst_x, temp_tst_x])
	del trn, tst

# one-time only
if initial_run:
	LOG.info('# Generate label_sparse')
	generate_label_sp(LOG)
	
	LOG.info('# Save trn_y')
	trn_y = pd.read_csv(trn_file, usecols=['ncodpers']+target_cols, dtype=dtype_list)
	trn_y = np.array(trn_y.fillna(0)).astype('int')[start_index:,1:]
	pd.DataFrame(trn_y, columns=target_cols).to_csv('{}{}_label.csv'.format(data_path, state), index=False)

LOG.info('# Saving preprocessed outputs')
pd.DataFrame(trn_x, columns=colnames).to_csv('{}{}_trn.csv'.format(data_path, state), index=False)
pd.DataFrame(tst_x, columns=colnames).to_csv('{}{}_tst.csv'.format(data_path, state), index=False)

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


