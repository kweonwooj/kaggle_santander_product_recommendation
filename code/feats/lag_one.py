"""
input : train, test_ver2
output: lag-1 features
"""

import pandas as pd
import numpy as np

def main():

    trn = pd.read_csv('../input/feats/train_base.csv', usecols=['ncodpers'])
    df = pd.read_csv('../input/train_ver2.csv')
    labels = pd.read_csv('../input/labels.csv')
    ncodpers = pd.read_csv('../input/train.csv', usecols=['ncodpers'])
    trn_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28']

    df_trn = df[df['fecha_dato'].isin(trn_dates)]
    lb_trn = labels[df['fecha_dato'].isin(trn_dates)]
    lb_trn = lb_trn[df_trn['ncodpers'].isin(ncodpers.values)]
    df_trn = df_trn[df_trn['ncodpers'].isin(ncodpers.values)]
    df_trn = clean(df_trn)

    ncodpers = pd.read_csv('../input/train_clean.csv', usecols=['ncodpers'])

    cols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
            'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
            'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
            'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
            'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
            'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
            'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
            'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

    labl_temp = lb_trn[df_trn['fecha_dato'] == '2015-05-28']
    trgt_temp = df_trn[df_trn['fecha_dato'] == '2015-05-28']
    trgt_temp = trgt_temp[cols].fillna(0).astype(int)

    lag_one_trgt = np.zeros((trn.shape[0], 24))
    lag_one_labl = np.zeros((trn.shape[0], 24))
    for i, (run, row) in enumerate(trn.iterrows()):
        ncodper = ncodpers.iloc[i][0]
        trgt_lag = trgt_temp[trgt_temp['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values
        labl_lag = labl_temp[trgt_temp['ncodpers'] == ncodper].values
        if len(trgt_lag) != 0:
            lag_one_trgt[i, :] = trgt_lag[0]
        if len(labl_lag) != 0:
            lag_one_labl[i, :] = labl_lag[0]



    # save
    trn_data.to_csv('../input/feats/train_lags.csv',index=False)
    tst_data.to_csv('../input/feats/test_lags.csv', index=False)



def clean(df_trn):
    # clean NAs
    col = 'ind_empleado'
    df_trn[col].fillna('NN', inplace=True)

    col = 'pais_residencia'
    df_trn[col].fillna('NN', inplace=True)

    col = 'sexo'
    df_trn[col].fillna('NN', inplace=True)

    col = 'age'
    df_trn[col].replace(' NA', 0, inplace=True)
    df_trn[col] = df_trn[col].astype(np.int64)

    col = 'fecha_alta'
    df_trn[col].fillna('2015-06-30', inplace=True)

    col = 'ind_nuevo'
    df_trn[col].fillna(-1, inplace=True)
    df_trn[col] = df_trn[col].astype(int)

    col = 'antiguedad'
    df_trn[col].replace('     NA', 0, inplace=True)
    df_trn[col] = df_trn[col].astype(np.int64)
    df_trn[col].replace(-999999, 0, inplace=True)

    col = 'indrel'
    df_trn[col].fillna(0, inplace=True)
    df_trn[col].replace(99, 2, inplace=True)
    df_trn[col] = df_trn[col].astype(int)

    col = 'ult_fec_cli_1t'
    df_trn[col].fillna('2015-06-30', inplace=True)

    col = 'indrel_1mes'
    df_trn[col].fillna(2, inplace=True)
    df_trn[col] = df_trn[col].astype(int)

    col = 'tiprel_1mes'
    df_trn[col].fillna('NN', inplace=True)

    col = 'indresi'
    df_trn[col].fillna('NN', inplace=True)

    col = 'indext'
    df_trn[col].fillna('NN', inplace=True)

    col = 'conyuemp'
    df_trn[col].fillna('NN', inplace=True)

    col = 'canal_entrada'
    df_trn[col].fillna('NN', inplace=True)

    col = 'indfall'
    df_trn[col].fillna('NN', inplace=True)

    col = 'tipodom'
    # drop tipodom, tst has only one unique value

    col = 'cod_prov'
    df_trn[col].fillna(0, inplace=True)
    df_trn[col] = df_trn[col].astype(int)

    col = 'nomprov'
    df_trn[col].fillna('NN', inplace=True)
    df_trn[col].replace('CORU\xc3\x91A, A', 'CORU', inplace=True)

    col = 'ind_actividad_cliente'
    df_trn[col].fillna(-1, inplace=True)
    df_trn[col] = df_trn[col].astype(int)

    col = 'renta'
    df_trn[col].fillna(0, inplace=True)

    col = 'segmento'
    df_trn[col].fillna('NN', inplace=True)

    df_trn.fillna(0, inplace=True)

    return df_trn

