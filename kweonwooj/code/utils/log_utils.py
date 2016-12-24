"""
    This file has a utility function for logging.
"""

import logging
import os

def get_logger(fname='log.txt'):
    logger = logging.getLogger('my_logger')

    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG)

        path = './log'
        if not os.path.exists(path):
            os.makedirs(path)
        fh = logging.FileHandler('./log/{}'.format(fname))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)-15s %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
    
    return logger
