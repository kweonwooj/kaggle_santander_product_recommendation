"""
input : train, test_ver2
output: advanced feature enginnering
"""

import pandas as pd
import numpy as np

def main():

    df = pd.read_csv('../input/train_ver2.csv')
    labels = pd.read_csv('../input/labels.csv')
    ncodpers = pd.read_csv('../input/train.csv', usecols=['ncodpers'])
    trn_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28']
    tst_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

    df_trn = df[df['fecha_dato'].isin(trn_dates)]
    df_trn = df_trn[df_trn['ncodpers'].isin(ncodpers.values)]
    df_tst = df[df['fecha_dato'].isin(tst_dates)]
    df_tst = df_tst[df_tst['ncodpers'].isin(ncodpers.values)]

    df_trn, df_tst = clean(df_trn, df_tst)

    lb_trn = labels.iloc[df_trn.index.tolist(), :]
    lb_trn['ncodpers'] = df_trn['ncodpers']
    lb_tst = labels.iloc[df_tst.index.tolist(), :]
    lb_tst['ncodpers'] = df_tst['ncodpers']


    cols = ['ncodpers','fecha_dato']
    # ncodpers_count : count of ncodpers in past 6 months
    ncodpers_count_trn = df_trn[cols].groupby(['ncodpers']).agg('size')
    ncodpers_count_tst = df_tst[cols].groupby(['ncodpers']).agg('size')


    # first appearance : distance from June to first appeared month
    first_app_trn = df_trn[cols].drop_duplicates(['ncodpers'], keep='first')
    first_app_tst = df_tst[cols].drop_duplicates(['ncodpers'], keep='first')


    # last appearance : distance from June to last appearance
    last_app_trn = df_trn[cols].drop_duplicates(['ncodpers'], keep='last')
    last_app_tst = df_tst[cols].drop_duplicates(['ncodpers'], keep='last')


    cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
            'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
            'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
            'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
            'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1','ncodpers']
    # prev target : target values in prev appearance
    prev_target_trn = df_trn[cols].drop_duplicates(['ncodpers'], keep='last')
    prev_target_tst = df_tst[cols].drop_duplicates(['ncodpers'], keep='last')
    # prev_target_sum : sum of prev_targets
    prev_target_trn['prev_target_sum'] = prev_target_trn.iloc[:, :-1].sum(axis=1)
    prev_target_tst['prev_target_sum'] = prev_target_tst.iloc[:, :-1].sum(axis=1)


    # prev bought : new purchase labels in prev appearance
    prev_bought_trn = lb_trn.drop_duplicates(['ncodpers'], keep='last')
    prev_bought_tst = lb_tst.drop_duplicates(['ncodpers'], keep='last')
    # prev_bought_sum : sum of prev_bought
    prev_bought_trn['prev_bought_sum'] = prev_bought_trn.iloc[:, :-1].sum(axis=1)
    prev_bought_tst['prev_bought_sum'] = prev_bought_tst.iloc[:, :-1].sum(axis=1)


    # cum target : cumulative sum of target values in prev months
    cum_targets_trn = df_trn.groupby(['ncodpers'])[cols].sum()
    cum_targets_tst = df_tst.groupby(['ncodpers'])[cols].sum()
    cum_targets_trn['ncodpers'] = cum_targets_trn.index.tolist()
    cum_targets_tst['ncodpers'] = cum_targets_tst.index.tolist()
    # cum_target_sum : cum sum of prev_targets
    cum_targets_trn['cum_target_sum'] = cum_targets_trn.iloc[:, :-1].sum(axis=1)
    cum_targets_tst['cum_target_sum'] = cum_targets_tst.iloc[:, :-1].sum(axis=1)


    # cum bought : cumulative sum of new purchase labels in prev months
    cum_bought_trn = lb_trn.groupby(['ncodpers'])[cols].sum()
    cum_bought_tst = lb_tst.groupby(['ncodpers'])[cols].sum()
    cum_bought_trn['ncodpers'] = cum_bought_trn.index.tolist()
    cum_bought_tst['ncodpers'] = cum_bought_tst.index.tolist()
    # cum_bought_sum : cum sum of prev_bought
    cum_bought_trn['cum_bought_sum'] = cum_bought_trn.iloc[:, :-1].sum(axis=1)
    cum_bought_tst['cum_bought_sum'] = cum_bought_tst.iloc[:, :-1].sum(axis=1)


    #### capture change in status which will lead to new purchases
    # unique count of each column over last months : show of variance >> no change in base columns / only for target values
    target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                   'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                   'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                   'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                   'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    target_unique_counts_trn = dict()
    target_unique_counts_tst = dict()
    extra = 'tipodom'
    for col in target_cols:
        count_trn = df_trn[['ncodpers', col, extra]].groupby(['ncodpers', col]).count()
        index_trn = count_trn.index
        count_tst = df_tst[['ncodpers', col, extra]].groupby(['ncodpers', col]).count()
        index_tst = count_tst.index

        unique_counts_trn = index_trn.get_level_values(0).value_counts()
        target_unique_counts_trn[col + '_unique'] = unique_counts_trn
        unique_counts_tst = index_tst.get_level_values(0).value_counts()
        target_unique_counts_tst[col + '_unique'] = unique_counts_tst

    # amount of change in numerical columns (renta, antiguedad) : Jan~June avg, June val - Jan~May avg, std, max, min >> no change
    # M-1 change : number of changes in categorical values >> no change
    # cum change : total number of changes in categorical values >> no change
    # [col]_changed : boolean showing change compared to prev month (M-1,2,3) >> no change


    #### feature ideas ####

    # Note
    # According to xgboost feature importance, below are top features
    ## age
    ## antiguedad
    ## fecha_alta
    ## segmento
    ## renta


    #### without an advanced features, mlogloss stays around 1.8~1.9
    #### try adding new features to reduce the mlogloss
    trn = pd.read_csv('../input/feats/train_base.csv', usecols=['ncodpers'])
    trn_data = []
    for run, ncodper in trn.iterrows():
        row = []
        ncodper = ncodper.values[0]


        if ncodper in ncodpers_count_trn:
            row.append(ncodpers_count_trn[ncodper])
        else:
            row.append(0)


        if ncodper in first_app_trn['ncodpers'].values:
            row.append(int(first_app_trn[first_app_trn['ncodpers'] == ncodper]['fecha_dato'].values[0].split('-')[1]))
        else:
            row.append(6)


        if ncodper in last_app_trn['ncodpers'].values:
            row.append(int(last_app_trn[last_app_trn['ncodpers'] == ncodper]['fecha_dato'].values[0].split('-')[1]))
        else:
            row.append(6)


        if ncodper in prev_target_trn['ncodpers'].values:
            temp = prev_target_trn[prev_target_trn['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        if ncodper in prev_bought_trn['ncodpers'].values:
            temp = prev_bought_trn[prev_bought_trn['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        if ncodper in cum_targets_trn['ncodpers'].values:
            temp = cum_targets_trn[cum_targets_trn['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        if ncodper in cum_bought_trn['ncodpers'].values:
            temp = cum_targets_trn[cum_targets_trn['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        for k in target_unique_counts_trn.keys():
            if ncodper in target_unique_counts_trn[k]:
                row.append(target_unique_counts_trn[k][ncodper])
            else:
                row.append(0)

        trn_data.append(row)


    tst = pd.read_csv('../input/feats/test_base.csv', usecols=['ncodpers'])
    tst_data = []
    for run, ncodper in tst.iterrows():
        row = []
        ncodper = ncodper.values[0]


        if ncodper in ncodpers_count_tst:
            row.append(ncodpers_count_tst[ncodper])
        else:
            row.append(0)


        if ncodper in first_app_tst['ncodpers'].values:
            row.append(int(first_app_tst[first_app_tst['ncodpers'] == ncodper]['fecha_dato'].values[0].split('-')[1]))
        else:
            row.append(6)


        if ncodper in last_app_tst['ncodpers'].values:
            row.append(int(last_app_tst[last_app_tst['ncodpers'] == ncodper]['fecha_dato'].values[0].split('-')[1]))
        else:
            row.append(6)


        if ncodper in prev_target_tst['ncodpers'].values:
            temp = prev_target_tst[prev_target_tst['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        if ncodper in prev_bought_tst['ncodpers'].values:
            temp = prev_bought_tst[prev_bought_tst['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        if ncodper in cum_targets_tst['ncodpers'].values:
            temp = cum_targets_tst[cum_targets_tst['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        if ncodper in cum_bought_tst['ncodpers'].values:
            temp = cum_bought_tst[cum_bought_tst['ncodpers'] == ncodper].drop(['ncodpers'], axis=1).values.tolist()[0]
            for i in temp:
                row.append(i)
        else:
            for i in range(25):
                row.append(0)


        for k in target_unique_counts_tst.keys():
            if ncodper in target_unique_counts_tst[k]:
                row.append(target_unique_counts_tst[k][ncodper])
            else:
                row.append(0)

        tst_data.append(row)

    # colnames
    colnames = []
    colnames.append('ncodpers_count')
    colnames.append('first_app')
    colnames.append('last_app')
    for i in range(24):
        colnames.append('prev_target_{}'.format(i + 1))
    colnames.append('prev_target_sum')
    for i in range(24):
        colnames.append('prev_bought_{}'.format(i + 1))
    colnames.append('prev_bought_sum')
    for i in range(24):
        colnames.append('cum_target_{}'.format(i + 1))
    colnames.append('cum_target_sum')
    for i in range(24):
        colnames.append('cum_bought_{}'.format(i + 1))
    colnames.append('cum_bought_sum')
    for k in target_unique_counts_trn.keys():
        colnames.append(k)

    trn_data = pd.DataFrame(trn_data, columns=colnames)
    tst_data = pd.DataFrame(tst_data, columns=colnames)
    # merge all features into ncodpers + features and save
    trn_data.to_csv('../input/feats/train_advanced.csv',index=False)
    tst_data.to_csv('../input/feats/test_advanced.csv', index=False)


def clean(df_trn, df_tst):
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
    df_tst[col] = df_tst[col].astype(int)

    col = 'antiguedad'
    df_trn[col].replace('     NA', 0, inplace=True)
    df_trn[col] = df_trn[col].astype(np.int64)
    df_trn[col].replace(-999999, 0, inplace=True)

    col = 'indrel'
    df_trn[col].fillna(0, inplace=True)
    df_trn[col].replace(99, 2, inplace=True)
    df_trn[col] = df_trn[col].astype(int)
    df_tst[col].replace(99, 2, inplace=True)
    df_tst[col] = df_tst[col].astype(int)

    col = 'ult_fec_cli_1t'
    df_trn[col].fillna('2015-06-30', inplace=True)
    df_tst[col].fillna('2016-01-04', inplace=True)

    col = 'indrel_1mes'
    df_trn[col].fillna(2, inplace=True)
    df_trn[col] = df_trn[col].astype(int)
    df_tst[col] = df_tst[col].astype(float).astype(int)

    col = 'tiprel_1mes'
    df_trn[col].fillna('NN', inplace=True)

    col = 'indresi'
    df_trn[col].fillna('NN', inplace=True)

    col = 'indext'
    df_trn[col].fillna('NN', inplace=True)

    col = 'conyuemp'
    df_trn[col].fillna('NN', inplace=True)
    df_tst[col].fillna('NN', inplace=True)

    col = 'canal_entrada'
    df_trn[col].fillna('NN', inplace=True)
    df_tst[col].fillna('NN', inplace=True)

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
    df_tst[col].fillna('NN', inplace=True)
    df_tst[col].replace('CORU\xc3\x91A, A', 'CORU', inplace=True)

    col = 'ind_actividad_cliente'
    df_trn[col].fillna(-1, inplace=True)
    df_trn[col] = df_trn[col].astype(int)

    col = 'renta'
    df_trn[col].fillna(0, inplace=True)

    col = 'segmento'
    df_trn[col].fillna('NN', inplace=True)
    df_tst[col].fillna('NN', inplace=True)

    df_trn.fillna(0, inplace=True)
    df_tst.fillna(0, inplace=True)

    return df_trn, df_tst

