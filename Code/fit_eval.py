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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

np.random.seed(7)

# use xgboost - softprob
# make validation process - MAP@7
def get_index(df, col, values):
    return np.where(df[col].isin(values))[0]

st = time.time()

LOG = get_logger('fit_eval_{}.txt'.format(state))
LOG.info('# Fitting and Evaluating {}'.format(state))

data_path = '../Data/Raw/'
df = pd.read_csv('{}trn_{}.csv'.format(data_path, state))
x = df.iloc[:,:-1]
x.fillna(0, inplace=True) ## treat this in preprocessing.py
y = df.iloc[:,-1]
LOG.info('# df\t{}'.format(df.shape))

buy_ncodpers = pickle.load(open('../Data/Raw/buy_ncodpers.pkl','rb'))

for i in range(5):
	arr = np.random.permutation(buy_ncodpers.shape[0])
	t_ind = arr < (buy_ncodpers.shape[0] * 0.2)
	t_ncodpers = buy_ncodpers[t_ind]
	v_ncodpers = buy_ncodpers[~t_ind]
	t_ind = np.in1d(x.ncodpers,t_ncodpers.values)
	v_ind = np.in1d(x.ncodpers,v_ncodpers.values)

	x_trn = x.iloc[t_ind]
	x_vld = x.iloc[v_ind]
	y_trn = y[t_ind]
	y_vld = y[v_ind]
    
	model.fit(x_trn, y_trn)
    
	preds_trn = model.predict_proba(x_trn)
	preds_trn = [np.argmax(pred) for pred in preds_trn]
	score = accuracy_score(y_trn, preds_trn)
	print 'TRN:', score
    
	preds_vld = model.predict_proba(x_vld)
	preds_vld = [np.argmax(pred) for pred in preds_vld]
	score = accuracy_score(y_vld, preds_vld)
	print 'VLD:', score


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


