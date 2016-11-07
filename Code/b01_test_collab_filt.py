# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: runs best hyperparameter for collaborative filtering approach
"""

import numpy as np
import config
from b01_collab_filt import run_solution
from utils.log_utils import get_logger

seed = 7
np.random.seed(seed)

def main():

  i = 4

  seeds = [7,777,123,14,2016]
  arrs = [ [11, 14, 13, 1, 2, 15, 21, 23],
           [20, 11, 10, 1, 19],
           [3, 23, 10, 17, 16, 22],
           [1],
           [1, 18, 21, 15, 7, 16, 4] ]
  ult_fec = [1,1,1,0,0]
  age = [1,0,0,0,0]
  indfall = [1,1,0,0,1]
  renta = [0,0,1,1,0]
  p_count = [0,1,0,1,0]

  for i in range(5):
    fname = 'log_seed_{}_best.txt'.format(seeds[i])
    arr = arrs[i]
    bools = dict()
    bools['ult_fec'] = ult_fec[i]
    bools['age'] = age[i]
    bools['indfall'] = indfall[i]
    bools['renta'] = renta[i]
    bools['p_count'] = p_count[i]
    
    run(fname, arr, bools)

def run(fname, arr, bools):
  LOG = get_logger(fname)
  LOG.info('Training begin.. seed: {}'.format(seed))

  # run collaborative filtering
  n_group, map7 = run_solution(arr, bools, LOG, train=True)
  LOG.info('Submission created!')
    
if __name__=='__main__':
  main()
