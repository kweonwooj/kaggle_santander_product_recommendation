

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
def get_df_idx(df, col, values):
    return np.where(df[col].isin(values))[0]

def gen_sp_y():
    df = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=target_cols, dtype=dtype_list)
    ncodper = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=['ncodpers'], dtype='object')
    df['ncodpers'] = ncodper

    ncodpers = df.ncodpers.unique()
    _, sample, _, _ = train_test_split(ncodpers, range(ncodpers.shape[0]), \
                                       test_size=0.0001, random_state=7)
    index = get_df_idx(df, 'ncodpers', sample)
    df_sample = df.iloc[index]

    print 'unique ncodpers:', sample.shape
    print 'df_sample shape:', df_sample.shape

    cust = dict()
    for ind, (run, row) in enumerate(df_sample.iterrows()):
            ncodper = row['ncodpers']
            labels = row[:-1].fillna(0).astype(int).values
            temp_sp_y = np.zeros(24).astype(int)
            if ncodper in cust:
                    for i in range(24):
                            if labels[i] == 1 and cust[ncodper][i] == 0:
                                    temp_sp_y[i] = 1
            else:
                    temp_sp_y = labels

            # print
            print '-'*50
            if ncodper in cust:
                print 'Prev    :', cust[ncodper]
            else:
                print 'First occurance'
            print 'original:', labels
            print 'sp_y    :', temp_sp_y

            cust[ncodper] = temp_sp_y

            if ind == 0:
                    sp_y = temp_sp_y.copy()
            else:
                    sp_y = np.vstack([sp_y, temp_sp_y])

if __name__=='__main__':
	gen_sp_y()
