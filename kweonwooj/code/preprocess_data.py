'''
    This file preprocess and append lag-5 to train and test data.

    Preprocess all values(date, categorical, numeric) into numeric.
    Append lag-5 features to trn and tst.

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

# preprocess data into numeric and save it
def preprocess_data(LOG):

    trn_path = './input/trn_lag.csv'
    tst_path = './input/tst_lag.csv'

    # load data
    trn = pd.read_csv(trn_path)
    tst = pd.read_csv(tst_path)

    # clean data
    skip_cols = ['fecha_dato', 'ncodpers']
    target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                   'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                   'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                   'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                   'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                   'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

    for col in trn.columns:
        if col in skip_cols:
            continue
        LOG.info('# column : {}'.format(col))

        if col == 'ind_empleado':
            trn[col].fillna('S', inplace=True)
        elif col == 'age':
            trn[col].replace(' NA', 0, inplace=True)
            trn[col] = trn[col].astype(str).astype(int)
            trn[col] = trn[col].astype(str).astype(int)
            continue
        elif col == 'fecha_alta':
            trn[col] = ((pd.to_datetime(trn['fecha_dato']) - pd.to_datetime(trn[col].fillna('2015-07-01')))
                        / np.timedelta64(1,'D')).astype(int)
            tst[col] = ((pd.to_datetime(tst['fecha_dato']) - pd.to_datetime(tst[col]))
                        / np.timedelta64(1, 'D')).astype(int)
            continue
        elif col == 'antiguedad':
            trn[col].replace('     NA', -1, inplace=True)
            trn[col] = trn[col].astype(str).astype(int)
            tst[col] = tst[col].astype(str).astype(int)
            continue
        elif col == 'ult_fec_cli_1t':
            trn[col] = ((pd.to_datetime(trn['fecha_dato']) - pd.to_datetime(trn[col].fillna('2015-06-30')))
                        / np.timedelta64(1,'D')).astype(int)
            tst[col] = ((pd.to_datetime(tst['fecha_dato']) - pd.to_datetime(tst[col].fillna('2016-01-03')))
                        / np.timedelta64(1,'D')).astype(int)
            continue
        elif col == 'indrel_1mes':
            tst[col].replace('1', '1.0', inplace=True)
            tst[col].replace('2', '1.0', inplace=True)
            tst[col].replace('2.0', '1.0', inplace=True)
            tst[col].replace(2.0, '1.0', inplace=True)
            tst[col].replace('3', '3.0', inplace=True)
            tst[col].replace('4', '3.0', inplace=True)
            tst[col].replace(4.0, '3.0', inplace=True)
            tst[col].replace('4.0', '3.0', inplace=True)
            tst[col].replace('P', '3.0', inplace=True)
        elif col == 'tiprel_1mes':
            tst[col].replace('N', 'I', inplace=True)
            tst[col].replace('R', 'P', inplace=True)
        elif col == 'indresi':
            trn[col].fillna('N', inplace=True)
        elif col == 'indext':
            trn[col].fillna('S', inplace=True)
        elif col == 'indfall':
            trn[col].fillna('N', inplace=True)
        elif col == 'tipodom':
            trn.drop([col], axis=1, inplace=True)
            tst.drop([col], axis=1, inplace=True)
            continue
        elif col == 'ind_actividad_cliente':
            trn[col].fillna(0.0, inplace=True)
        elif col == 'renta':
            tst[col].replace('         NA', 0, inplace=True)
            trn[col].fillna(-1, inplace=True)
            tst[col].fillna(-1, inplace=True)
            trn[col] = trn[col].astype(str).astype(float).astype(int)
            tst[col] = tst[col].astype(str).astype(float).astype(int)
            continue
        elif col in target_cols:
            trn[col].fillna(0, inplace=True)
            trn[col] = trn[col].astype(int)
            tst[col].fillna(0, inplace=True)
            tst[col] = tst[col].astype(int)

        lb = LabelEncoder()
        lb.fit(pd.concat([trn[col].astype(str), tst[col].astype(str)], axis=0))
        trn[col] = lb.transform(trn[col].astype(str))
        tst[col] = lb.transform(tst[col].astype(str))

    LOG.info('# Appending lag-5..')
    trn_june = trn[trn['fecha_dato'] == '2015-06-28'].drop(target_cols, axis=1)
    trn_othr = trn[trn['fecha_dato'] != '2015-06-28']
    tst_june = tst[tst['fecha_dato'] == '2016-06-28'].drop(target_cols, axis=1)
    tst_othr = tst[tst['fecha_dato'] != '2016-06-28']

    drop_cols = ['fecha_dato', 'ncodpers']

    LOG.info('# Appending trn data.. {} rows'.format(trn_june.shape[0]))
    st = time.time()
    trn_append = []
    for i, ncodper in enumerate(trn_june['ncodpers']):
        temp = trn_othr[trn_othr['ncodpers'] == ncodper].drop(drop_cols, axis=1)
        if temp.shape[0] == 0:
            row = ['NA'] * 225
        else:
            row = np.hstack([temp.shift(periods=i).iloc[-1, :] for i in range(temp.shape[0])]).tolist()
        trn_append.append(trn_june.iloc[i].drop(drop_cols).values.tolist() + row)

        if i % int(trn_june.shape[0] / 10) == 0:
            LOG.info('# {} rows.. {} secs..'.format(i, round(time.time() - st), 2))

    LOG.info('# Appending tst data.. {} rows'.format(tst_june.shape[0]))
    st = time.time()
    tst_append = []
    for i, ncodper in enumerate(tst_june['ncodpers']):
        temp = tst_othr[tst_othr['ncodpers'] == ncodper].drop(drop_cols, axis=1)
        if temp.shape[0] == 0:
            row = ['NA'] * 225
        else:
            row = np.hstack([temp.shift(periods=i).iloc[-1, :] for i in range(temp.shape[0])]).tolist()
        tst_append.append(tst_june.iloc[i].drop(drop_cols).values.tolist() + row)

        if i % int(tst_june.shape[0] / 10) == 0:
            LOG.info('# {} rows.. {} secs..'.format(i, round(time.time() - st), 2))

    # change colnames
    colnames = trn_june.drop(drop_cols, axis=1).columns.values.tolist()
    suffixes = ['_lag_one', '_lag_two', '_lag_thr', '_lag_fou', '_lag_fiv']
    for suffix in suffixes:
        for col in trn_othr.drop(drop_cols, axis=1).columns.values.tolist():
            colnames.append(col + suffix)

    trn = pd.DataFrame(trn_append, columns=colnames)
    tst = pd.DataFrame(tst_append, columns=colnames)
    LOG.info('# trn : {} | tst : {}'.format(trn.shape, tst.shape))

    LOG.info('# Saving data as trn.csv / tst.csv ..')
    trn.to_csv('./input/trn.csv', index=False)
    tst.to_csv('./input/tst.csv', index=False)