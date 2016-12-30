'''
    This file concatenates all individual submission files into single kaggle-uploadable submission file
'''

import numpy as np
import pandas as pd
import math

def make_submission(LOG):
    mode = 'submission'
    product_list = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    date_list = ['2016-05-28']

    submission = pd.read_csv('../root_input/sample_submission.csv', usecols=['ncodpers'])

    result = pd.DataFrame()

    for i in range(len(date_list)):
        print(date_list[i])

        result_temp = pd.DataFrame()

        for j in range(len(product_list)):
            product = product_list[j]

            if j in [0, 1, 9, 10]:
                temp = submission
                temp['pr'] = 1e-10
            else:
                if product == 'ind_cco_fin_ult1':
                    train_date = '2015-12-28'
                elif product == 'ind_reca_fin_ult1':
                    train_date = '2015-06-28'
                else:
                    train_date = '2016-05-28'

                temp = pd.read_csv('./input/submission_{}_{}.csv'.format(product, train_date))
                temp.columns = ['ncodpers', 'pr']

            temp[product] = product
            result_temp = pd.concat([result_temp, temp], axis=0)

        # normalize
        pred_sum = result_temp.drop(['ind_cco_fin_ult1', 'ind_reca_fin_ult1'], axis=1).sum(axis=1)
        result_temp['pred_sum'] = pred_sum
        result_temp['log_pr'] = result_temp[['pr', 'pred_sum']].apply(lambda x: math.log(x['pr'] / x['pred_sum']))
        result = pd.concat([result, result_temp['ncodpers', 'product', 'log_pr']], axis=0)
        result['N'] = 1
        result = result.groupby(['ncodpers', product])['log_pr', 'N'].sum()

    # log-average
    result['log_pr'] = result.apply(lambda x: x['log_pr'] / x['N'])

    # elect top 7 products
    # order by ncodpers

    for i in range(7):
        print(i + 1)
        temp = result
    # replicate 3_make_submission.R line 50 ~ 55

    submission[submission.isnull()] = ''
    submission['added_products'] = submission['p{}'.format(1)]
    for i in range(2, 8):
        submission['added_products'] = submission['added_products'].map(str) + submission['p{}'.format(i)].map(str)

    # order by ncodpers
    file_name = '{}.csv'.format(mode)
    submission['ncodpers', 'added_products'].to_csv(file_name, index=False)
