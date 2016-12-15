"""
input : train, test_ver2
output: category feature enginnering
"""

import pandas as pd
import numpy as np

def main():
    # category columns
    category_cols = ['ind_empleado','pais_residencia','sexo','ind_nuevo','indrel','indrel_1mes','tiprel_1mes','indresi',
                     'indext','conyuemp','canal_entrada','indfall','cod_prov','nomprov','ind_actividad_cliente','segmento']

    # load data
    trn = pd.read_csv('../input/feats/train_base.csv', usecols=category_cols)
    tst = pd.read_csv('../input/feats/test_base.csv', usecols=category_cols)

    #### feature ideas ####
    #

    #### for all ####
    #

    # save
    trn.to_csv('../input/feats/train_category.csv',index=False)
    tst.to_csv('../input/feats/test_category.csv', index=False)
