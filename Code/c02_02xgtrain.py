# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file 
	- trains and validates xgboost model
"""

import xgboost as xgb
from utils.log_utils import get_logger

LOG = get_logger('xg_train_vld.txt')
LOG.info('# Training XGBoost and validating...')

LOG.info('# Open vld_trn data...')
# train data
f = open('../Data/Clean/xgb_vld_trn.csv','r')
trn_data = []
trn_labels = []
total = 0
while 1:
    line = f.readline()[:-1]
    total += 1

    if line == '': 
        break    

    line = line.split(',')
    datum = line[:464]
    label = line[464:]
    
    trn_data.append(datum)
    trn_labels.append(label)
    
    if total % 1000000 == 0:
        LOG.info('# Processing {} lines ...'.format(total))
f.close()
LOG.info('# Processed all vld_trn data : {} lines'.format(total))

import numpy as np

# fit tree
LOG.info('# Training XGBoost')

xgb_pars = {
    'eta': 0.05,
    'gamma': 0.01,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1,
    'lambda': 1,
    'objective': 'multi:softmax',
    'num_class': 24,
    'eval_metrics': 'map@7',
    'nthread': 12,
    'seed': 7,
    'silent': 1
}

LOG.info('# Open vld_tst data...')
# valid data
f = open('../Data/Clean/xgb_vld_tst.csv','r')
vld_data = []
vld_labels = []
total = 0
while 1:
    line = f.readline()[:-1]
    total += 1    

    if line == '':
       break

    line = line.split(',')
    datum = line[:464]
    label = line[464:]
    
    vld_data.append(datum)
    vld_labels.append(label)

    if total % 1000000 == 0:
        LOG.info('# Processing {} lines ...'.format(total))
f.close()
LOG.info('# Processed all vld_tst data : {} lines'.format(total))

dtrain = xgb.DMatrix(trn_data, label=trn_labels)
dval = xgb.DMatrix(vld_data, label=vld_labels)
watchlist = [(dtrain, 'train'), (dval, 'val')]

LOG.info('# Training XGBoost model')
model = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators, verbose_eval=10, evals=watchlist)

LOG.info('# Making predictions on vld_tst_data')
y_pred = model.predict(dval)

def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def get_true_values(label):
    true = []
    for i, l in enumerate(label):
        if l == '1':
            true.append(i)
    return true

def get_pred_values(pred):
    all = []
    overallbest = [23,22,21,2,18,4,12,11,17,6,19,7,13,9,8,15,5,10,16,3,14,20,1,0]
    for j in range(len(pred[0])):
        l = []
        for i in range(24):
            try:
                l.append(pred[j][i][1])
            except:
                l = overallbest[:7]
        if sum(l) <= 0.3:
            l = overallbest[:7]
        if len(l) == 24:
            l = sorted(range(len(l)), key=lambda k: l[k], reverse=True)[:7]
        all.append(l)
    return all

LOG.info('# Validating MAP@7 score...')
map7 = 0
for i in range(len(vld_labels)):
    map7 += apk(get_true_values(vld_labels[i]),get_pred_values(preds)[i])
map7 /= len(vld_labels)

LOG.info('# MAP@7 score : {}'.format(map7))
