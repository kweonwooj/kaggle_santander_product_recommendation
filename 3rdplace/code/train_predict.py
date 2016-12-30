'''
    This file trains and predicts each product with xgboost

    2 of the 20 features are trained and predicted on specific months
    18 of the 20 features are trained and predicted on 2016-05-28 data

    Feature engineering are done simultaneously in this file
'''

import pandas as pd
import numpy as np
import xgboost as xgb


def train_predict(LOG):

    # drop ind_ahor_fin_ult1, ind_aval_fin_ult1, ind_deco_fin_ult1 and ind_deme_fin_ult1
    product_list = ["ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
                    "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_dela_fin_ult1",
                    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
                    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
                    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1"]

    # train xgboost per product as binary classifier
    for i, product in enumerate(product_list):

        LOG.info('# Product # {} : {}'.format(i+1, product))

        # choose train_date according to product
        if product == 'ind_cco_fin_ult1':
            train_date = '2015-12-28'
        elif product == 'ind_reca_fin_ult1':
            train_date = '2015-06-28'
        else:
            train_date = '2016-05-28'

        LOG.info('# Load and append train/count data')
        # load and append train/count data
        data_1 = pd.read_csv('./input/train_{}.csv'.format(train_date), dtype={'indrel_1mes': str,
                                                                               'conyuemp': str})
        data_2 = pd.read_csv('./input/count_{}.csv'.format(train_date))
        data_train = pd.concat([data_1, data_2], axis=1)

        LOG.info('# Load and append validation data')
        # load validation data
        if train_date == '2016-05-28':
            data_1 = pd.read_csv('./input/train_2016-04-28.csv', dtype={'indrel_1mes': str, 'conyuemp': str})
            dat1_2 = pd.read_csv('./input/count_2016-04-28.csv')
        else:
            data_1 = pd.read_csv('./input/train_2016-05-28.csv', dtype={'indrel_1mes': str, 'conyuemp': str})
            data_2 = pd.read_csv('./input/count_2016-05-28.csv')
        data_valid = pd.concat([data_1, data_2], axis=1)

        LOG.info('# Load and append test data')
        # load test data
        data_1 = pd.read_csv('./input/test_2016-06-28.csv', dtype={'indrel_1mes': str, 'conyuemp': str})
        data_2 = pd.read_csv('./input/count_2016-06-28.csv')
        data_test = pd.concat([data_1, data_2], axis=1)

        # memory efficiency
        del data_1, data_2

        LOG.info('# Select rows with {}_last == 0'.format(product))
        # select rows with [product]_last == 0
        data_train = data_train[data_train['{}_last'.format(product)] == 0]
        data_valid = data_valid[data_valid['{}_last'.format(product)] == 0]
        data_test = data_test[data_test['{}_last'.format(product)] == 0]

        LOG.info('# Add ind_actividad_cliente_from_to')
        # add ind_actividad_cliente_from_to, simple string concat
        data_train['ind_actividad_cliente_from_to'] = data_train['ind_actividad_cliente_last'].astype(int).astype(str).str.cat(
            data_train['ind_actividad_cliente'].astype(int).astype(str), sep=' ')
        data_valid['ind_actividad_cliente_from_to'] = data_valid['ind_actividad_cliente_last'].astype(int).astype(str).str.cat(
            data_valid['ind_actividad_cliente'].astype(int).astype(str), sep=' ')
        data_test['ind_actividad_cliente_from_to'] = data_test['ind_actividad_cliente_last'].astype(int).astype(str).str.cat(
            data_test['ind_actividad_cliente'].astype(int).astype(str), sep=' ')

        LOG.info('# Add tiprel_1mes_from_to')
        # add tiprel_1mes_from_to, simple string concat
        data_train['tiprel_1mes_from_to'] = data_train['tiprel_1mes_last'].astype(str).str.cat(
            data_train['tiprel_1mes'].astype(str), sep=' ')
        data_valid['tiprel_1mes_from_to'] = data_valid['tiprel_1mes_last'].astype(str).str.cat(
            data_valid['tiprel_1mes'].astype(str), sep=' ')
        data_test['tiprel_1mes_from_to'] = data_test['tiprel_1mes_last'].astype(str).str.cat(
            data_test['tiprel_1mes'].astype(str), sep=' ')

        LOG.info('# Drop ind_actividad_cliente_last')
        # drop ind_actividad_cliente_last
        data_train.drop(['ind_actividad_cliente_last'], axis=1, inplace=True)
        data_valid.drop(['ind_actividad_cliente_last'], axis=1, inplace=True)
        data_test.drop(['ind_actividad_cliente_last'], axis=1, inplace=True)

        LOG.info('# Drop tiprel_1mes_last')
        # drop tiprel_1mes_last
        data_train.drop(['tiprel_1mes_last'], axis=1, inplace=True)
        data_valid.drop(['tiprel_1mes_last'], axis=1, inplace=True)
        data_test.drop(['tiprel_1mes_last'], axis=1, inplace=True)

        LOG.info('# Add sum of last months product possessions')
        # add sum of last months' product possession
        data_train['n_products_last'] = data_train[data_train.columns[40:60].tolist()].sum(axis=1)
        data_valid['n_products_last'] = data_valid[data_valid.columns[40:60].tolist()].sum(axis=1)
        data_test['n_products_last'] = data_test[data_test.columns[40:60].tolist()].sum(axis=1)

        LOG.info('# Add string concat of last months product possessions')
        # add string concatenation of last months' product possessions
        data_train['products_last'] = ''
        data_valid['products_last'] = ''
        data_test['products_last'] = ''

        for j in range(20):
            data_train['products_last'] = data_train['products_last'].map(str) + data_train[
                data_train.columns[40 + j]].map(str)
            data_valid['products_last'] = data_valid['products_last'].map(str) + data_valid[
                data_valid.columns[40 + j]].map(str)
            data_test['products_last'] = data_test['products_last'].map(str) + data_test[data_test.columns[20 + j]].map(
                str)

        LOG.info('# Iterate over columns, convert categoricals')
        exp_var = data_test.columns[2:].tolist()
        for var in exp_var:
            LOG.info('# var {}'.format(var))
            if data_train[var].iloc[0].__class__ == str:
                levels = data_train[var].unique()

                if len(levels) == 2:
                    LOG.info('# len(levels) == 2')
                    # concatenate all train/valid/test
                    temp = pd.concat([data_train[var], data_valid[var], data_test[var]])

                    # map the category to int
                    map_dict = {}
                    for i in range(len(levels)):
                        map_dict[levels[i]] = np.argsort(levels)[i]
                    temp = temp.apply(lambda x: map_dict[x])
                    temp = temp.reset_index(drop=True)

                    # redistribute into train/valid/test
                    data_train[var] = temp[:data_train.shape[0]]
                    data_valid[var] = temp[data_train.shape[0]:data_train.shape[0] + data_valid.shape[0]]
                    data_test[var] = temp[-data_test.shape[0]:]
                else:
                    LOG.info('# len(levels) > 2')
                    # get var/product dataframe only
                    data_train[var].fillna('-99', inplace=True)
                    data_valid[var].fillna('-99', inplace=True)
                    data_test[var].fillna('-99', inplace=True)
                    data_temp = data_train[[var, product]].fillna('-99')
                    data_temp.columns = [var, 'target']

                    # get average of product per category label
                    target_mean = data_temp.groupby(var).mean().reset_index()

                    # fill in target_mean with 0 if item not found in target_mean
                    for item in data_valid[var].unique().tolist():
                        if item not in target_mean[var].tolist():
                            add = pd.DataFrame({var: item, 'target': 0},
                                               index=[len(target_mean[var].unique().tolist())])
                            target_mean = target_mean.append(add)
                    for item in data_test[var].unique().tolist():
                        if item not in target_mean[var].tolist():
                            add = pd.DataFrame({var: item, 'target': 0},
                                               index=[len(target_mean[var].unique().tolist())])
                            target_mean = target_mean.append(add)

                    # map the average to valid and test
                    data_valid[var] = data_valid[var].map(
                        lambda x: target_mean.loc[target_mean[var] == x, 'target'].values[0])
                    data_test[var] = data_test[var].map(
                        lambda x: target_mean.loc[target_mean[var] == x, 'target'].values[0])

                    ### What is this calculating?
                    temp = pd.Series([np.nan] * data_train.shape[0])
                    for j in range(4):
                        # ids_1 : index to drop per 4
                        ids_1 = np.arange(j, data_train.shape[0], 4).tolist()
                        ids_1 = np.delete(data_temp.index.tolist(), ids_1, None)

                        # ids_2 : index to select per 4
                        ids_2 = np.arange(j, data_train.shape[0], 4)

                        # average of values groupby var
                        target_mean = data_temp.ix[ids_1].groupby(var).mean().reset_index()
                        try:
                            temp.loc[ids_2] = data_train.reset_index(drop=True).loc[ids_2, var].map(
                                lambda x: target_mean.loc[target_mean[var] == x, 'target'].values[0])
                        except:
                            for item in data_temp[var].unique().tolist():
                                if item not in target_mean[var].tolist():
                                    add = pd.DataFrame({var: item, 'target': 0},
                                                       index=[len(data_temp[var].unique().tolist())])
                                    target_mean = target_mean.append(add)
                            temp.loc[ids_2] = data_train.reset_index(drop=True).loc[ids_2, var].map(
                                lambda x: target_mean.loc[target_mean[var] == x, 'target'].values[0])
                    data_train[var] = temp
        del data_temp

        LOG.info('# Prepare data for xgb')
        # prepare data
        x_train = data_train[exp_var]
        y_train = data_train[product]
        dtrain = xgb.DMatrix(x_train.as_matrix(), label=y_train)
        del data_train, x_train

        x_valid = data_valid[exp_var]
        y_valid = data_valid[product]
        dvalid = xgb.DMatrix(x_valid.as_matrix(), label=y_valid)
        data_valid = data_valid['ncodpers']
        del x_valid

        x_test = data_test[exp_var]
        dtest = xgb.DMatrix(x_test.as_matrix())
        data_test = data_test['ncodpers']
        del x_test

        LOG.info('# XGB params initialize')
        # xgb param
        nrounds = 1000
        early_stop = 50
        params = {
            'eta': 0.05,
            'max_depth': 4,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': 0,
        }

        LOG.info('# Fit XGB')
        # fit
        model_xgb = xgb.train(params=params,
                              dtrain=dtrain,
                              num_boost_round=nrounds,
                              evals=[(dtrain, 'train'), (dvalid, 'eval')],
                              early_stopping_rounds=early_stop,
                              verbose_eval=10,
                              )
        best_ntree_limit = model_xgb.best_ntree_limit

        LOG.info('# Predict and save valid/test')
        # predict valid / test
        result = pd.DataFrame({'ncodpers': data_valid.values,
                               product: model_xgb.predict(dvalid, ntree_limit=best_ntree_limit)})
        result.to_csv('./input/valid_{}_{}.csv'.format(product, train_date), index=False)

        result = pd.DataFrame({'ncodpers': data_test.values,
                               product: model_xgb.predict(dtest, ntree_limit=best_ntree_limit)})
        result.to_csv('./input/submission_{}_{}.csv'.format(product, train_date), index=False)

        del dtrain, dvalid, dtest, result

        LOG.info('# Save model')
        # save model
        model_xgb.save_model('./input/xgboost_{}_{}.model'.format(product, train_date))