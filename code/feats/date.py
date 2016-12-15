"""
input : train, test_ver2
output: date feature enginnering
"""

import pandas as pd
import numpy as np

def main():
    # date columns
    date_cols = ['fecha_dato','fecha_alta','ult_fec_cli_1t']

    # load data
    trn = pd.read_csv('../input/feats/train_base.csv', usecols=date_cols)
    tst = pd.read_csv('../input/feats/test_base.csv', usecols=date_cols)

    ## dates values
    trn['dato-alta'] = (pd.to_datetime(trn['fecha_dato']) - pd.to_datetime(trn['fecha_alta'])).dt.days
    trn['ult-dato'] = (pd.to_datetime(trn['ult_fec_cli_1t']) - pd.to_datetime(trn['fecha_dato'])).dt.days

    tst['dato-alta'] = (pd.to_datetime(tst['fecha_dato']) - pd.to_datetime(tst['fecha_alta'])).dt.days
    tst['ult-dato'] = (pd.to_datetime(tst['ult_fec_cli_1t']) - pd.to_datetime(tst['fecha_dato'])).dt.days + 31

    trn['alta_year'] = pd.to_datetime(trn['fecha_alta']).dt.year
    trn['alta_month'] = pd.to_datetime(trn['fecha_alta']).dt.month
    trn['alta_week'] = pd.to_datetime(trn['fecha_alta']).dt.week
    trn['alta_day'] = pd.to_datetime(trn['fecha_alta']).dt.day
    trn['alta_dayofweek'] = pd.to_datetime(trn['fecha_alta']).dt.dayofweek

    tst['alta_year'] = pd.to_datetime(tst['fecha_alta']).dt.year
    tst['alta_month'] = pd.to_datetime(tst['fecha_alta']).dt.month
    tst['alta_week'] = pd.to_datetime(tst['fecha_alta']).dt.week
    tst['alta_day'] = pd.to_datetime(tst['fecha_alta']).dt.day
    tst['alta_dayofweek'] = pd.to_datetime(tst['fecha_alta']).dt.dayofweek

    trn['ult_year'] = pd.to_datetime(trn['ult_fec_cli_1t']).dt.year
    trn['ult_month'] = pd.to_datetime(trn['ult_fec_cli_1t']).dt.month
    trn['ult_week'] = pd.to_datetime(trn['ult_fec_cli_1t']).dt.week
    trn['ult_day'] = pd.to_datetime(trn['ult_fec_cli_1t']).dt.day
    trn['ult_dayofweek'] = pd.to_datetime(trn['ult_fec_cli_1t']).dt.dayofweek

    tst['ult_year'] = pd.to_datetime(tst['ult_fec_cli_1t']).dt.year
    tst['ult_month'] = pd.to_datetime(tst['ult_fec_cli_1t']).dt.month
    tst['ult_week'] = pd.to_datetime(tst['ult_fec_cli_1t']).dt.week
    tst['ult_day'] = pd.to_datetime(tst['ult_fec_cli_1t']).dt.day
    tst['ult_dayofweek'] = pd.to_datetime(tst['ult_fec_cli_1t']).dt.dayofweek

    #### feature ideas ####
    # dato - alta
    # ult - dato (add +31 days in tst)
    # year, month, week, day, dayofweek (weak)

    # drop origin cols
    date_cols = ['fecha_dato','fecha_alta','ult_fec_cli_1t']
    trn.drop(['fecha_dato','fecha_alta','ult_fec_cli_1t'], axis=1, inplace=True)
    tst.drop(['fecha_dato','fecha_alta','ult_fec_cli_1t'], axis=1, inplace=True)

    # save
    trn.to_csv('../input/feats/train_date.csv',index=False)
    tst.to_csv('../input/feats/test_date.csv', index=False)
