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

  seeds = [7,777,123,14,2016]
  arrs = [ [19,17,12,1,4],
           [1,16,20,9],
           [1,22,9,20,19,12,15,3,4,2,10,11],
           [2,12,1,4,18],
           [13,4,16,1,18] ]
  ult_fec = [0,1,0,0,1]
  age = [0,1,0,0,1]
  indfall = [1,0,1,0,1]
  renta = [1,0,0,1,0]
  p_count = [1,1,1,1,1]

  for i in range(5):
    if i < 2: continue
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
