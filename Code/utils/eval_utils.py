"""
    This file has a eval function for MAP@7 score
"""

def get_true_index(y_true):
  real = []
  


def get_pred_index(y_pred):


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


def eval_map7(y_trues, y_preds):

  map7 = 0.0

  # check two dimension match
  assert y_trues.shape == y_preds.shape

  # iterate over each row
  for row_ind in range(y_preds.shape[0]):

    # convert y_true from [0,0,0,1,1,0] to [7,23,1,2] index
    y_true = get_true_index(y_trues[row_ind])    

    # convert y_pred from [0.1, 0.2,...] to [7,23,1,2] index
    y_pred = get_pred_index(y_preds[row_ind])

    # get score and add to map7
    map7 += apk(y_true, y_pred)

  # divide by number of rows
  map7 /= (1. * y_preds.shape[0])

  return map7 
