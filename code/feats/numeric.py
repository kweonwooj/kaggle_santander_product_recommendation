"""
input : train, test_ver2
output: numerical feature enginnering
"""

import pandas as pd
import numpy as np

def main():
    # numerical columns
    num_cols = ['age','antiguedad','renta']

    # load data
    trn = pd.read_csv('../input/feats/train_base.csv', usecols=num_cols)
    tst = pd.read_csv('../input/feats/test_base.csv', usecols=num_cols)

    # age
    col = 'age'
    trn['age_log'] = np.log(trn[col]+1)
    trn['age_bin5'] = (trn[col]/5).astype(np.int64)
    trn['age_bin10'] = (trn[col]/10).astype(np.int64)
    trn['age_q1'] = (trn[col] <= trn[col].quantile(0.25)).astype(int)
    trn['age_q2'] = (trn[col] <= trn[col].quantile(0.5)).astype(int) - (trn[col] <= trn[col].quantile(0.25)).astype(int)
    trn['age_q3'] = (trn[col] <= trn[col].quantile(0.75)).astype(int) - (trn[col] <= trn[col].quantile(0.5)).astype(int)
    trn['age_q4'] = (trn[col] > trn[col].quantile(0.75)).astype(int)

    tst['age_log'] = np.log(tst[col]+1)
    tst['age_bin5'] = (tst[col]/5).astype(np.int64)
    tst['age_bin10'] = (tst[col]/10).astype(np.int64)
    tst['age_q1'] = (tst[col] <= tst[col].quantile(0.25)).astype(int)
    tst['age_q2'] = (tst[col] <= tst[col].quantile(0.5)).astype(int) - (tst[col] <= tst[col].quantile(0.25)).astype(int)
    tst['age_q3'] = (tst[col] <= tst[col].quantile(0.75)).astype(int) - (tst[col] <= tst[col].quantile(0.5)).astype(int)
    tst['age_q4'] = (tst[col] > tst[col].quantile(0.75)).astype(int)


    # antiguedad
    col = 'antiguedad'
    trn['antiguedad_log'] = np.log(trn[col]+1)
    trn['antiguedad_bin5'] = (trn[col]/5).astype(np.int64)
    trn['antiguedad_bin10'] = (trn[col]/10).astype(np.int64)
    trn['antiguedad_q1'] = (trn[col] <= trn[col].quantile(0.25)).astype(int)
    trn['antiguedad_q2'] = (trn[col] <= trn[col].quantile(0.5)).astype(int) - (trn[col] <= trn[col].quantile(0.25)).astype(int)
    trn['antiguedad_q3'] = (trn[col] <= trn[col].quantile(0.75)).astype(int) - (trn[col] <= trn[col].quantile(0.5)).astype(int)
    trn['antiguedad_q4'] = (trn[col] > trn[col].quantile(0.75)).astype(int)

    tst['antiguedad_log'] = np.log(tst[col]+1)
    tst['antiguedad_bin5'] = (tst[col]/5).astype(np.int64)
    tst['antiguedad_bin10'] = (tst[col]/10).astype(np.int64)
    tst['antiguedad_q1'] = (tst[col] <= tst[col].quantile(0.25)).astype(int)
    tst['antiguedad_q2'] = (tst[col] <= tst[col].quantile(0.5)).astype(int) - (tst[col] <= tst[col].quantile(0.25)).astype(int)
    tst['antiguedad_q3'] = (tst[col] <= tst[col].quantile(0.75)).astype(int) - (tst[col] <= tst[col].quantile(0.5)).astype(int)
    tst['antiguedad_q4'] = (tst[col] > tst[col].quantile(0.75)).astype(int)


    # renta
    col = 'renta'
    trn['renta_log'] = np.log(trn[col]+1)
    trn['renta_bin10k'] = (trn[col]/10000).astype(np.int64)
    trn['renta_bin20k'] = (trn[col]/20000).astype(np.int64)
    trn['renta_q1'] = (trn[col] <= trn[col].quantile(0.25)).astype(int)
    trn['renta_q2'] = (trn[col] <= trn[col].quantile(0.5)).astype(int) - (trn[col] <= trn[col].quantile(0.25)).astype(int)
    trn['renta_q3'] = (trn[col] <= trn[col].quantile(0.75)).astype(int) - (trn[col] <= trn[col].quantile(0.5)).astype(int)
    trn['renta_q4'] = (trn[col] > trn[col].quantile(0.75)).astype(int)

    tst['renta_log'] = np.log(tst[col]+1)
    tst['renta_bin10k'] = (tst[col]/10000).astype(np.int64)
    tst['renta_bin20k'] = (tst[col]/20000).astype(np.int64)
    tst['renta_q1'] = (tst[col] <= tst[col].quantile(0.25)).astype(int)
    tst['renta_q2'] = (tst[col] <= tst[col].quantile(0.5)).astype(int) - (tst[col] <= tst[col].quantile(0.25)).astype(int)
    tst['renta_q3'] = (tst[col] <= tst[col].quantile(0.75)).astype(int) - (tst[col] <= tst[col].quantile(0.5)).astype(int)
    tst['renta_q4'] = (tst[col] > tst[col].quantile(0.75)).astype(int)


    #### feature ideas ####
    # bin
    # quantile (25,50,75)
    # log(x+1)
    ####

    # save
    trn.to_csv('../input/feats/train_num.csv',index=False)
    tst.to_csv('../input/feats/test_num.csv', index=False)
