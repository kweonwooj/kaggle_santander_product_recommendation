"""
input : train, test_ver2
output: clean (remove NAs) dataset > train_clean, test_clean
"""

import pandas as pd
import numpy as np

def main():
    # load data
    trn = pd.read_csv('../input/train.csv').drop(['target', 'tipodom'], axis=1)
    tst = pd.read_csv('../input/test_ver2.csv').drop(['tipodom'], axis=1)

    # clean NAs
    col = 'ind_empleado'
    trn[col].fillna('NN',inplace=True)

    col = 'pais_residencia'
    trn[col].fillna('NN',inplace=True)

    col = 'sexo'
    trn[col].fillna('NN',inplace=True)
    tst[col].fillna('NN',inplace=True)

    col = 'age'
    trn[col].replace(' NA', 0, inplace=True)
    trn[col] = trn[col].astype(np.int64)

    col = 'fecha_alta'
    trn[col].fillna('2015-06-30',inplace=True)

    col = 'ind_nuevo'
    trn[col].fillna(-1,inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'antiguedad'
    trn[col].replace('     NA', 0, inplace=True)
    trn[col] = trn[col].astype(np.int64)
    tst[col].replace(-999999, 0, inplace=True)

    col = 'indrel'
    trn[col].fillna(0, inplace=True)
    trn[col].replace(99,2, inplace=True)
    trn[col] = trn[col].astype(int)
    tst[col].replace(99,2, inplace=True)

    col = 'ult_fec_cli_1t'
    trn[col].fillna('2015-06-30', inplace=True)
    tst[col].fillna('2016-05-30', inplace=True)

    col = 'indrel_1mes'
    trn[col].fillna(2, inplace=True)
    tst[col].fillna(2, inplace=True)
    trn[col] = trn[col].astype(int)
    tst[col] = tst[col].astype(int)

    col = 'tiprel_1mes'
    trn[col].fillna('NN', inplace=True)
    tst[col].fillna('NN', inplace=True)

    col = 'indresi'
    trn[col].fillna('NN', inplace=True)

    col = 'indext'
    trn[col].fillna('NN', inplace=True)

    col = 'conyuemp'
    trn[col].fillna('NN', inplace=True)
    tst[col].fillna('NN', inplace=True)

    col = 'canal_entrada'
    trn[col].fillna('NN', inplace=True)
    tst[col].fillna('NN', inplace=True)

    col = 'indfall'
    trn[col].fillna('NN', inplace=True)

    col = 'tipodom'
    # drop tipodom, tst has only one unique value

    col = 'cod_prov'
    trn[col].fillna(0, inplace=True)
    tst[col].fillna(0, inplace=True)
    trn[col] = trn[col].astype(int)
    tst[col] = tst[col].astype(int)

    col = 'nomprov'
    trn[col].fillna('NN', inplace=True)
    tst[col].fillna('NN', inplace=True)
    trn[col].replace('CORU\xc3\x91A, A','CORU',inplace=True)
    tst[col].replace('CORU\xc3\x91A, A','CORU',inplace=True)

    col = 'ind_actividad_cliente'
    trn[col].fillna(-1, inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'renta'
    trn[col].fillna(0, inplace=True)
    tst[col].replace('         NA', 0, inplace=True)
    tst[col] = tst[col].astype(np.float64)

    col = 'segmento'
    trn[col].fillna('NN', inplace=True)
    tst[col].fillna('NN', inplace=True)

    # save
    trn.to_csv('../input/train_clean.csv',index=False)
    tst.to_csv('../input/test_clean.csv', index=False)


