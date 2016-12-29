'''
    This file is a python-implementation of Jack (Japan)'s 3rd place solution in Kaggle \
    Santander Product Recommendation competition.

'''

from utils.log_utils import get_logger
from make_data_v3 import make_data
from train_predict import train_predict


LOG = get_logger('3rd_place_solution.log')


if __name__ == "__main__":
    LOG.info('=' * 50)
    make_data()

    LOG.info('=' * 50)
    train_predict()

    LOG.info('=' * 50)
    #make_submission()