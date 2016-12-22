'''
    This file is the initial Santander Product Recommendation pipeline script that serves as a starter.
    It has all the components necessary for a ml pipeline(data, model, eval(trn,vld), submission) with no tuning.
    Cross-validation scheme uses 9:1 split stratifiedshufflesplit x 5 times

    This script has result as below:
        TRN logloss : 1.80879
        VLD logloss : 1.84295
        PLB :
'''

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb
import os

def main():
    # file paths
    TRN = '../root_input/train_ver2.csv'
    TST = '../root_input/test_ver2.csv'

    print('# Loading data..')
    target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                   'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                   'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                   'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                   'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

    # use 2015-06 data only
    trn = pd.read_csv(TRN, usecols=target_cols + ['fecha_dato', 'ncodpers'])
    trn_may = trn[trn['fecha_dato'] == '2015-05-28'].drop(['fecha_dato'], axis=1).set_index('ncodpers')
    trn_jun = trn[trn['fecha_dato'] == '2015-06-28'].drop(['fecha_dato'], axis=1).set_index('ncodpers')
    trn_may.columns = [col + '_may' for col in target_cols]
    trn_jun.columns = [col + '_jun' for col in target_cols]

    # calculating 'purchase' from 'possessions' by subtracting May target values from June, this is naive method
    labels = trn_jun.join(trn_may)
    labels.fillna(0, inplace=True)
    for col in target_cols:
        labels[col] = labels[col + '_jun'] - labels[col + '_may']
    labels = labels[target_cols]
    index = labels[labels.sum(axis=1) > 0].index
    labels = labels.loc[index]

    trn = pd.read_csv(TRN)
    trn = trn[trn['fecha_dato'] == '2015-06-28'].set_index('ncodpers')
    trn = trn.loc[index]
    trn[target_cols] = labels

    del labels, index, trn_may, trn_jun

    # melt 24 classes into single multi-class
    trn_melt = []
    for ind, (run, row) in enumerate(trn.iterrows()):
        for i in range(24):
            if row[23 + i] == 1:
                temp = row[:23].values.tolist()
                temp.append(i)
                trn_melt.append(temp)
    trn = pd.DataFrame(np.asarray(trn_melt), columns=trn.columns.tolist()[:23] + ['target'])
    y = trn['target'].astype(int)

    # drop targets with too few frequency
    rem_targets = [23, 22, 2, 21, 18, 17, 4, 11, 12, 9, 6, 13, 7, 19, 8]
    trn = trn[y.isin(rem_targets)]
    y = y[y.isin(rem_targets)]
    y = LabelEncoder().fit_transform(y)

    tst = pd.read_csv(TST)
    tst.drop(['ncodpers'], axis=1, inplace=True)
    print('# trn shape : {} | y shape : {} | tst shape : {}'.format(trn.shape, y.shape, tst.shape))


    ##################################################################################################################
    ### Feature Engineering
    ##################################################################################################################

    print('# Processing data..')
    cols_to_use = ['pais_residencia', 'sexo', 'age', 'antiguedad', 'canal_entrada', 'cod_prov', 'renta', 'segmento']
    trn = trn[cols_to_use]
    tst = tst[cols_to_use]

    # factorize = LabelEncode categorical features
    categoricals = ['pais_residencia', 'sexo', 'canal_entrada', 'segmento']
    for col in categoricals:
        temp, _ = pd.concat([trn[col], tst[col]], axis=0).factorize()
        trn[col] = temp[:trn.shape[0]]
        tst[col] = temp[trn.shape[0]:]

    # preprocessing
    trn.fillna(-99, inplace=True)
    tst.fillna(-99, inplace=True)

    trn['age'].replace(' NA', -99, inplace=True)
    trn['age'] = trn['age'].astype(int)

    trn['antiguedad'].replace('     NA', -99, inplace=True)
    trn['antiguedad'] = trn['antiguedad'].astype(int)

    trn['cod_prov'] = trn['cod_prov'].replace('nan', -99).astype(float).astype(int)
    tst['cod_prov'] = tst['cod_prov'].replace('nan', -99).astype(float).astype(int)

    trn['renta'] = trn['renta'].replace('nan', -99).astype(float).astype(int)
    tst['renta'] = tst['renta'].replace('         NA', -99).astype(float).astype(int)

    # leave nan as is
    trn.replace(-99, np.nan, inplace=True)
    tst.replace(-99, np.nan, inplace=True)

    ##################################################################################################################
    ### CV Evaluation
    ##################################################################################################################

    print('# Cross validation..')

    # XGB Model Param
    num_round = 500
    early_stop = 10
    xgb_params = {
        'booster': 'gbtree',

        # model complexity
        'max_depth': 2,  # higher, more complex
        # 'gamma': 3,    # lower, more complex
        # 'min_child_weight': 5, # lower, more complex

        # regularization via random
        # 'colsample_bylevel': 0.7,
        # 'colsample_bytree': 1,
        # 'subsample': 0.8,

        # regulization
        # 'reg_alpha': 2,
        # 'reg_lambda': 3,

        # 'learning_rate': 0.03,

        # basic
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
    n_splits = 5
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=777)
    for i, (t_ind, v_ind) in enumerate(sss.split(trn, y)):
        print('# Iter {} / {}'.format(i+1, n_splits))
        x_trn = np.asarray(trn)[t_ind]
        x_vld = np.asarray(trn)[v_ind]
        y_trn = np.asarray(y)[t_ind]
        y_vld = np.asarray(y)[v_ind]

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

    print('# TRN logloss: {}'.format(np.mean(trn_scores)))
    print('# VLD logloss: {}'.format(np.mean(vld_scores)))
    print('# Best Iters : {}'.format(np.mean(best_iters)))
    # TRN logloss : 1.80879
    # VLD logloss : 1.84295
    # Best Iters  : 56

    ##################################################################################################################
    ### Model Fit
    ##################################################################################################################

    print('# Refit and predict on test data..')
    dtrn = xgb.DMatrix(trn, label=y)
    num_round = 56
    bst = xgb.train(xgb_params, dtrn, num_round, verbose_eval=False)

    dtst = xgb.DMatrix(tst)
    preds = bst.predict(dtst)
    preds = np.fliplr(np.argsort(preds, axis=1))

    ##################################################################################################################
    ### Submission
    ##################################################################################################################

    print('# Generating a submission..')
    submit_cols = [target_cols[i] for i, col in enumerate(target_cols) if i in rem_targets]

    final_preds = []
    for pred in preds:
        top_products = []
        for i, product in enumerate(pred):
            top_products.append(submit_cols[product])
            if i == 6:
                break
        final_preds.append(' '.join(top_products))

    t_index = pd.read_csv(TST, usecols=['ncodpers'])
    test_id = t_index['ncodpers']
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    file_name = datetime.now().strftime("result_%Y%m%d%H%M%S") + '.csv'
    out_df.to_csv(os.path.join('./output', file_name), index=False)


if __name__=='__main__':
    main()
