# -*- encoding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief:
	input: tst
	output: final submission
"""

import numpy as np
import pandas as pd
from config import state, model, target_cols, dtype_list
from utils.log_utils import get_logger
import pickle
from datetime import datetime
import time

st = time.time()

use_sp = False

LOG = get_logger('predict_{}.txt'.format(state))
LOG.info('# Predicting and Making Submission _ {}'.format(state))

data_path = '../Data/Raw/'
trn = pd.read_csv('{}{}_trn.csv'.format(data_path, state))
tst = pd.read_csv('{}{}_tst.csv'.format(data_path, state))
# choose label_sp or label according to fit_eval
if use_sp:
	y = pd.read_csv('{}{}_label_sp.csv'.format(data_path, state))
else:
	y = pd.read_csv('{}{}_label.csv'.format(data_path, state))
LOG.info('# trn_x {} | trn_y {} | tst_x {}'.format(trn.shape, y.shape, tst.shape))

preds = []
for ind, col in enumerate(target_cols):
	LOG.info('# Fitting and Predicting tst on {} | {} / {}'.format(col,ind+1,len(target_cols)))
	# fit
	model.fit(trn, y[col])

	# predict
	pred = np.array(model.predict_proba(tst)[:,1])
	preds.append(pred)
del trn, y, tst, model
preds = np.array(preds).T
LOG.info('# preds.shape {} | must be (tst_x, 24)'.format(preds.shape))

LOG.info('# Getting the last instance dict..')
y = pd.read_csv('{}train_ver2.csv'.format(data_path), usecols=['ncodpers']+target_cols, dtype=dtype_list)
last_instance_df = y.drop_duplicates('ncodpers', keep='last')
del y
last_instance_df = last_instance_df.fillna(0).astype('int')
cust_dict = {}
target_cols = np.array(target_cols)
for ind, row in last_instance_df.iterrows():
	cust = row['ncodpers']
	used_products = set(target_cols[np.array(row[1:])==1])
	cust_dict[cust] = used_products
del last_instance_df

LOG.info("# Creating submission..")
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)
test_id = np.array(pd.read_csv('{}test_ver2.csv'.format(data_path), usecols=['ncodpers'])['ncodpers'])
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
	final_preds.append(" ".join(new_top_products))
LOG.info('# final_preds.shape {} | must be (tst_x, 1)'.format(np.array(final_preds).shape))
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
now = str(datetime.now().strftime('%Y-%m-%d-%H-%M'))
out_df.to_csv('../Output/Subm/sub_rf_{}.csv'.format(now), index=False)

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
LOG.info("# DONE!")


