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
import os.path

## define target_cols

target_cols = [	'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1', \
		'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1', \
		'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1', \
		'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1', \
		'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1', \
		'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1', \
		'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1', \
		'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

st = time.time()

LOG = get_logger('predict_{}.txt'.format(state))
LOG.info('# Fitting and Predicting {}'.format(state))

data_path = '../Data/Raw/'
df = pd.read_csv('{}trn_{}.csv'.format(data_path, state))
LOG.info('# df\t{}'.format(df.shape))

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

if not os.path.isfile('../Data/Raw/tst_v1.csv'):
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
LOG.info('# tst shape : {}'.format(tst.shape))

x.fillna(0,inplace=True)
LOG.info('# Fitting model.')
model.fit(x,y)

LOG.info('# Predicting submission.')
preds = model.predict_proba(tst)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv('../Data/Raw/test_ver2.csv', usecols=['ncodpers'])['ncodpers'])
final_preds = []
for pred in preds:
	row_pred = []
	for p in pred:
		row_pred.append(target_cols[p])
	final_preds.append(' '.join(row_pred))
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


