"""
    This file has a eval function for MAP@7 score
"""
from __future__ import division
import numpy as np

def get_pred_index(y_pred):
  ## Option A
  # if confidence is low, return overallbest instead
  y_pred = list(y_pred)
  real = sorted(range(len(y_pred)), key=lambda k: y_pred[k], reverse=True)[:7]
  return real

def apk(actual, predicted, k=7):
    if len(actual) == 0:
      return 0.0

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


def eval_map7(y_trues, y_preds):

  map7 = 0.0

  # iterate over each row
  for row_ind in range(y_preds.shape[0]):

    # y_true is in the form [7,23,1,2] index
    y_true = y_trues[row_ind]

    # convert y_pred from [0.1, 0.2,...] to [7,23,1,2] index
    y_pred = get_pred_index(y_preds[row_ind])

    # get score and add to map7
    map7 += apk(y_true, y_pred)

  # divide by number of rows
  map7 /= (1. * y_preds.shape[0])

  return map7 
