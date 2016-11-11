# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file 
	- trains and validates neural net model
"""
from __future__ import division
from utils.log_utils import get_logger
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(7)

def get_model():
    # define neural net
    # FC 512 x 512 as default
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(464,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

## define custom map7 loss function for keras
#def keras_map7(y_true, y_pred):
#  return map7

# input: y_true, y_pred
# output: map7 score
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

# input: y_true ohe label (24,1) 
# output: index of purchases (n,1)
def get_true_values(label):
    true = []
    for i, l in enumerate(label):
        if l == '1':
            true.append(i)
    return true

# input: y_pred (24,1)
# output: top 7 indices (7,1)
def get_pred_values(pred):
    pred = pred[0]
    result = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)[:7]
    return result

# input: list of y_true, list of y_pred
# output: total map7 score
def get_map7_score(y_true, y_pred):
    y_true = get_true_values(y_true)
    y_pred = get_pred_values(y_pred)
       
    map7 = apk(y_true, y_pred)
    return map7

def get_vld_data(model, LOG):
    g = open('../Data/Clean/xgb_vld_tst.csv','r')
    total = 0
    vld_map7 = 0.0

    LOG.info('# Getting vld_data...')
    while 1:
        line = g.readline()[:-1]
        total += 1

        if line == '':
            break

        line = line.split(',')
        datum = line[:464]
        label = line[464:]

        datum = np.expand_dims(np.array(datum), axis=0)
        pred = model.predict(datum)    

        vld_map7 += get_map7_score(label, pred)

        if total % 1000000 == 0:
            LOG.info('# Processing {} lines ...'.format(total))
    g.close()
    vld_map7 /= (1. * total)
    return vld_map7

def main():
    LOG = get_logger('nn_train_vld.txt')
    LOG.info('# Training Neural network and validating...')

    LOG.info('# Open vld_trn data...')
    # train data
    f = open('../Data/Clean/xgb_vld_trn.csv','r')
    trn_data = []
    trn_labels = []
    total = 0

    model = get_model()

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
            LOG.info('# Processing {} lines and Training Neural Net...'.format(total))
        
            # shuffle data
            # train on neural net - keras
            model.fit(trn_data, trn_labels, nb_epoch=1, batch_size=512)
            # IDEA # can use memory based random selection to reduce variance
        
            # show trn, vld map7 error
            LOG.info('# Computing Train MAP@7 score...')
            trn_map7 = 0.0
            for i,feat in enumerate(trn_data):
                feat = np.expand_dims(np.array(feat), axis=0)
                trn_pred = model.predict(feat)
                trn_map7 += get_map7_score(trn_labels[i], trn_pred)
                if i % 100000 == 0:
                    LOG.info('# Processing {} / 1000000 lines...'.format(i+1))
            trn_map7 /= (1. * len(trn_data))
            LOG.info('# Train MAP@7 score : {}'.format(trn_map7))

            LOG.info('# Computing Valid MAP@7 score...')
            vld_map7 = get_vld_data(model, LOG)
            LOG.info('# Valid MAP@7 score : {}'.format(vld_map7))
            
            trn_data = []
            trn_labels = []
    f.close()
    LOG.info('# Processed all vld_trn data : {} lines'.format(total))

    LOG.info('# Saving model weights')
    model.save_weights('../Model/nn_01.h5')

if  __name__=='__main__':
    main()
