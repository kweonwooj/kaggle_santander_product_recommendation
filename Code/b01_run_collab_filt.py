# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: runs hyperparameter optimization for collaborative filtering approach
"""

import numpy as np
import config
import time 
from b01_collab_filt import run_solution
from utils.log_utils import get_logger

seed = 14
np.random.seed(seed)

def main():
  start = time.time()
  LOG = get_logger('log_seed_{}.txt'.format(seed))
  LOG.info('Iteration begin.. seed: {}'.format(seed))
  count = 0

  best_map7 = 0.0
  best_param = dict()

  while True:
    # randomly select length between 1 ~ 24
    n = np.random.randint(24) + 1

    # randomly select int in (0~23) for each element of an index
    arr = np.arange(24)
    np.random.shuffle(arr)
    arr = arr[:n]

    # list of boolean options
    bools = dict()
    
    bools['indfall'] = np.random.randint(2)
    bools['age'] = np.random.randint(2)
    bools['p_count'] = np.random.randint(2)
    bools['ult_fec'] = np.random.randint(2)
    bools['renta'] = np.random.randint(2)

    LOG.info('-'*50)
    count += 1
    LOG.info('iteration : {}'.format(count))
    LOG.info('arr: {}'.format([config.FEATURES[i] for i in arr]))
    for k,v in bools.items():
      LOG.info('{} : {}'.format(k,v))

    # run collaborative filtering
    n_group, map7 = run_solution(arr, bools, LOG)
    LOG.info('    MAP@7 : {}'.format(map7))
    LOG.info('    N_GROUP : {}'.format(n_group))
    
    if best_map7 < map7:
      LOG.info('## Performance Improved!')
      best_map7 = map7
      best_param['arr'] = arr
      best_param['bools'] = bools

    end = time.time()
    elapsed_secs = end - start
    elapsed_mins = elapsed_secs / 60.
    elapsed_hours = elapsed_mins / 60.
    if elapsed_hours > 8:
      break

  LOG.info('='*50)
  LOG.info('Optimization Finished.')
  LOG.info('    Best Score:')
  LOG.info('        MAP@7 : {}'.format(best_map7))
  LOG.info('    Best Param:')
  LOG.info('        ARR : {}'.format(best_param['arr']))
  LOG.info('        BOOLS : {}'.format(best_param['bools']))

if __name__=='__main__':
  main()
