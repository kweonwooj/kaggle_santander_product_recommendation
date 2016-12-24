'''
    This file is the kweonwooj's Santander Product Recommendation pipeline script.
    Feature Engineering is mainly lag-5 feature with few aggregation.
    Model is XGBoost with complexity tuned.
    Cross-validation scheme uses 95:05 split stratifiedshufflesplit x 2 times

    This script has result as below:
        TRN logloss : 0.8548372
        VLD logloss : 0.9492806
        Best Iters  : 199.5
        Private LB  : 0.0302238
'''

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb
from utils.log_utils import get_logger
from prepare_data import prepare_data
from preprocess_data import preprocess_data
import os
import time


LOG = get_logger('kweonwooj_solution.log')


def main():

    ##################################################################################################################
    # Prepare data
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Prepare data..')
    prepare_data(LOG)

    ##################################################################################################################
    # Preprocessing
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Preprocessing data..')
    preprocess_data(LOG)

    ##################################################################################################################
    # Feature Engineering
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Feature Engineering..')
    trn_path = './input/trn.csv'
    tst_path = './input/tst.csv'
    trg_path = './input/target.csv'

    # load data
    trn = pd.read_csv(trn_path)
    tst = pd.read_csv(tst_path)
    trg = pd.read_csv(trg_path)

    target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                   'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                   'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                   'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                   'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                   'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    lags = ['_lag_one', '_lag_two', '_lag_thr', '_lag_fou', '_lag_fiv']
    diffs = [['fiv', 'fou'], ['fou', 'thr'], ['thr', 'two'], ['two', 'one']]

    LOG.info('# na_count')
    # null count per row
    trn['na_count'] = trn.isnull().sum(axis=1)
    tst['na_count'] = tst.isnull().sum(axis=1)

    LOG.info('# target_sum_lag')
    # total count of purchases per month
    for lag in lags:
        trn['target_sum' + lag] = (trn[[col + lag for col in target_cols]].sum(axis=1))
        tst['target_sum' + lag] = (tst[[col + lag for col in target_cols]].sum(axis=1))

    LOG.info('# avg of cols')
    # average of cols over past 5 months
    cols = ['ind_actividad_cliente', 'ult_fec_cli_1t']
    for col in cols:
        trn[col + lag + '_avg'] = (trn[[col + lag for lag in lags]]).mean(axis=1)
        tst[col + lag + '_avg'] = (tst[[col + lag for lag in lags]]).mean(axis=1)

    LOG.info('# target_sum over lag-5')
    # cumulative sum of target cols over past 5 months
    for col in target_cols:
        trn[col + '_sum'] = (trn[[col + lag for lag in lags]].sum(axis=1))
        tst[col + '_sum'] = (tst[[col + lag for lag in lags]].sum(axis=1))

    LOG.info('# target_sum_diff for each months')
    # change in count of purchases per month compared to its last month
    for diff in diffs:
        pre = diff[0]
        post = diff[1]
        trn['target_diff_' + post + '-' + pre] = trn['target_sum_lag_' + post] - trn['target_sum_lag_' + pre]
        tst['target_diff_' + post + '-' + pre] = tst['target_sum_lag_' + post] - tst['target_sum_lag_' + pre]

    LOG.info('# target_diff for each months')
    # change in individual purchases for each month compared to its last month
    for col in target_cols:
        for diff in diffs:
            pre = diff[0]
            post = diff[1]
            trn[col + '_label_lag_' + post] = trn[col + '_lag_' + post] - trn[col + '_lag_' + pre]
            tst[col + '_label_lag_' + post] = tst[col + '_lag_' + post] - tst[col + '_lag_' + pre]

    LOG.info('# unique target count')
    # unique count of purchased targets over 5 months
    trn['unique_target_count'] = (trn[[col + '_sum' for col in target_cols]] > 0).astype(int).sum(axis=1)
    tst['unique_target_count'] = (tst[[col + '_sum' for col in target_cols]] > 0).astype(int).sum(axis=1)

    LOG.info('# Drop infrequent targets..')
    rem_targets = [2, 23, 22, 21, 18, 17, 4, 12, 11, 9, 6, 13, 7, 19, 8]
    trn = trn[trg['0'].isin(rem_targets)]
    trg = trg[trg['0'].isin(rem_targets)]
    trg = LabelEncoder().fit_transform(trg)

    LOG.info('# trn : {} | trg : {} | tst : {}'.format(trn.shape, trg.shape, tst.shape))

    # cache
    LOG.info('# Caching data as trn.csv / tst.csv ..')
    trn.to_csv('./input/trn_cache.csv', index=False)
    tst.to_csv('./input/tst_cache.csv', index=False)
    pd.DataFrame(trg).to_csv('./input/trg_cache.csv', index=False)

    ##################################################################################################################
    # CV Evaluation
    ##################################################################################################################

    # from cache
    trn = pd.read_csv('./input/trn_cache.csv')
    tst = pd.read_csv('./input/tst_cache.csv')
    trg = pd.read_csv('./input/trg_cache.csv')

    LOG.info('=' * 50)
    LOG.info('# Cross validation..')

    # XGB Model Param
    num_round = 500
    early_stop = 50
    xgb_params = {
        'booster': 'gbtree',
        'gamma': 1,
        'learning_rate': 0.1,
        'max_depth': 4,
        'min_child_weight': 3,
        'nthread': 4,
        'num_class': 15,
        'objective': 'multi:softprob',
        'silent': 1,
        'eval_metric': 'mlogloss',
        'seed': 777,
        }

    trn_scores = []
    vld_scores = []
    best_iters = []
    n_splits = 2
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.05, random_state=777)
    for i, (t_ind, v_ind) in enumerate(sss.split(trn, trg)):
        LOG.info('# Iter {} / {}'.format(i+1, n_splits))
        x_trn = np.asarray(trn)[t_ind]
        x_vld = np.asarray(trn)[v_ind]
        y_trn = np.asarray(trg)[t_ind]
        y_vld = np.asarray(trg)[v_ind]

        dtrn = xgb.DMatrix(x_trn, label=y_trn)
        dvld = xgb.DMatrix(x_vld, label=y_vld)
        watch_list = [(dtrn, 'train'), (dvld, 'eval')]

        # fit xgb
        bst = xgb.train(xgb_params, dtrn, num_round, watch_list,
                        early_stopping_rounds=early_stop, verbose_eval=True)

        # eval _ trn
        score = log_loss(y_trn, bst.predict(dtrn))
        trn_scores.append(score)

        # eval _ vld
        score = log_loss(y_vld, bst.predict(dvld))
        vld_scores.append(score)

        # best iters
        best_iters.append(bst.best_iteration)

    LOG.info('# TRN logloss: {}'.format(np.mean(trn_scores)))
    LOG.info('# VLD logloss: {}'.format(np.mean(vld_scores)))
    LOG.info('# Best Iters : {}'.format(np.mean(best_iters)))

    ##################################################################################################################
    # Model Fit
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Refit and predict on test data..')
    dtrn = xgb.DMatrix(trn, label=trg)
    num_round = int(np.mean(best_iters) / 0.9)
    bst = xgb.train(xgb_params, dtrn, num_round, verbose_eval=False)

    dtst = xgb.DMatrix(tst)
    preds = bst.predict(dtst)
    preds = np.fliplr(np.argsort(preds, axis=1))

    ##################################################################################################################
    # Submission
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Generating a submission..')
    submit_cols = [target_cols[i] for i, col in enumerate(target_cols) if i in rem_targets]

    final_preds = []
    for pred in preds:
        top_products = []
        for i, product in enumerate(pred):
            top_products.append(submit_cols[product])
            if i == 6:
                break
        final_preds.append(' '.join(top_products))

    t_index = pd.read_csv('../root_input/test_ver2.csv', usecols=['ncodpers'])
    test_id = t_index['ncodpers']
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    file_name = datetime.now().strftime("result_%Y%m%d%H%M%S") + '.csv'
    path = './output'
    if not os.path.exists(path):
        os.makedirs(path)
    out_df.to_csv(os.path.join(path, file_name), index=False)

    LOG.info('# Clean files')
    cmd = 'rm -rf ./input'
    os.system(cmd)

    LOG.info('=' * 50)
    LOG.info('# Finished!')
    LOG.info('=' * 50)


if __name__ == '__main__':
    start = time.time()
    main()
    LOG.info('finished ({:.2f} sec elapsed)'.format(time.time() - start))
