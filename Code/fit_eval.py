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

st = time.time()

LOG = get_logger('fit_eval_{}.txt'.format(state))
LOG.info('# Fitting and Evaluating {}'.format(state))

data_path = '../Data/Raw/'
x = pd.read_csv('{}{}_trn.csv'.format(data_path, state))
#sp_y = pd.read_csv('{}{}_label_sp.csv'.format(data_path, state))
y = pd.read_csv('{}{}_label.csv'.format(data_path, state))
LOG.info('# x\t{}\t| y\t{}'.format(x.shape, y.shape))

vld_bound = 11787582

trn_x = x.iloc[:vld_bound,:]
vld_x = x.iloc[vld_bound:,:]
trn_y = y.iloc[:vld_bound,:]
vld_y = y.iloc[vld_bound:,:]
#trn_sp_y = sp_y.iloc[:vld_bound,:]
#vld_sp_y = sp_y.iloc[vld_bound:,:]
LOG.info('# trn_x\t{}\t| trn_y\t{}\t| trn_sp_y\t{}'.format(trn_x.shape, trn_y.shape, 1))#trn_sp_y.shape))
LOG.info('# vld_x\t{}\t| vld_y\t{}\t| vld_sp_y\t{}'.format(vld_x.shape, vld_y.shape, 1))#vld_sp_y.shape))

# train with trn_y
trn_preds = []; vld_preds = []
for ind, col in enumerate(trn_y.columns):
	LOG.info('# Fitting model on {} with trn_y\t| {} / {}'.format(col,ind+1,len(trn_y.columns)))
	model.fit(trn_x, trn_y[col])

	# eval
	preds = np.array(model.predict_proba(trn_x))[:,1]
	trn_preds.append(preds)
	preds = np.array(model.predict_proba(vld_x))[:,1]
	vld_preds.append(preds)

trn_preds = np.array(trn_preds).T
vld_preds = np.array(vld_preds).T
LOG.info('# trn preds {} vld_preds {}'.format(trn_preds.shape, vld_preds.shape))
map7 = eval_map7(trn_y.values.tolist(), (trn_preds).tolist()) # must be sp_y
LOG.info('# MAP@7 score _trn : {}'.format(map7))
map7 = eval_map7(vld_y.values.tolist(), (vld_preds).tolist()) # must be sp_y
LOG.info('# MAP@7 score _vld : {}'.format(map7))

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


