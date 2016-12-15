import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns


def main():

    # load data
    trn = pd.read_csv('../input/train_ver2.csv')


    #### BASIC ####

    # check column names
    print trn.columns

    # check head
    print trn.head()

    # check overall stat
    print trn.info()

    # observe numerical columns
    num_cols = ['ncodpers', 'ind_nuevo', 'indrel', 'tipodom', \
                'cod_prov', 'ind_actividad_cliente', 'renta', \
                'age', 'antiguedad']
    print trn[num_cols].describe()

    # observe categorical columns
    cat_cols = ['fecha_dato', 'ind_empleado', 'pais_residencia', \
                'sexo', 'fecha_alta', 'ult_fec_cli_1t', 'indrel_1mes', \
                'tiprel_1mes', 'indresi', 'indext', 'conyuemp', \
                'canal_entrada', 'indfall', 'nomprov', 'segmento']
    # num_unique, unique set
    for col in cat_cols:
        uniq = np.unique(trn[col])
        print '-' * 50
        print '# col {}\t n_uniq {}\t unique {}'.format(col, len(uniq), uniq)


    #### HISTOGRAM ####

    ## Plot histogram for each features
    for col in trn.columns:
        print '='*50
        print 'Col : ', col
        f, ax = plt.subplots(figsize=(20, 15))
        sns.countplot(trn[col], alpha=0.5)
        plt.show()
    ## you will encounter errors or dirty visualizations

    # observe unique items
    for col in trn.columns:
        print '='*50
        print 'Col : ', col
        print 'TRN : ', np.unique(trn[col])

    # clean the data first
    trn = clean(trn)
    ## Plot histogram for each features
    for col in trn.columns:
        print '='*50
        print 'Col : ', col
        f, ax = plt.subplots(figsize=(20, 15))
        sns.countplot(trn[col], alpha=0.5)
        plt.show()


    #### STACKED HISTOGRAM ####

    #  prepare for visualization
    months = np.unique(trn['fecha_dato']).tolist()
    label_cols = trn.columns[24:].tolist()
    label_over_time = []
    for i in range(len(label_cols)):
        label_over_time.append(trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum').tolist())
    label_sum_over_time = []
    for i in range(len(label_cols)):
        label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
    color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE', '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']

    # plot stacked barplot of month vs labels
    f, ax = plt.subplots(figsize=(30, 15))
    for i in range(len(label_cols)):
        sns.barplot(x=months, y=label_sum_over_time[i], color=color_list[i % 8], alpha=0.7)

    plt.legend([plt.Rectangle((0, 0), 1, 1, fc=color_list[i % 8], edgecolor='none') for i in range(len(label_cols))], \
               label_cols, loc=1, ncol=2, prop={'size': 16})

    # plot stacked barplot of month vs labels in percentage
    label_sum_percent = (label_sum_over_time / (1. * np.asarray(label_sum_over_time).max(axis=0))) * 100

    f, ax = plt.subplots(figsize=(30, 15))
    for i in range(len(label_cols)):
        sns.barplot(x=months, y=label_sum_percent[i], color=color_list[i % 8], alpha=0.7)

    plt.legend([plt.Rectangle((0, 0), 1, 1, fc=color_list[i % 8], edgecolor='none') for i in range(len(label_cols))], \
               label_cols, loc=1, ncol=2, prop={'size': 16})



    #### VISUALIZATION ON JUNE 2015 DATA ONLY ####
    trn = pd.read_csv('../input/train.csv')
    # check column names
    trn.columns
    # check head
    trn.head()
    # check overall stat
    trn.info()

    # clean data
    for col in trn.columns:
        print col, np.unique(trn[col])

    trn['ind_actividad_cliente'].fillna(2, inplace=True)
    trn['ind_nuevo'].fillna(2, inplace=True)
    trn['fecha_alta'].fillna('2015-06-30', inplace=True)
    trn.fillna(0, inplace=True)
    trn['age'] = trn['age'].replace(' NA',0).astype(int)
    trn['nomprov'].replace('CORU\xc3\x91A, A','CORU', inplace=True)
    trn['renta'] = (trn['renta']/4000).astype(int)

    # check cleaned data
    for col in trn.columns:
        print col, np.unique(trn[col])

    ## Plot histogram for each features
    for col in trn.columns:
        print '='*50
        print 'Col : ', col
        f, ax = plt.subplots(figsize=(20, 15))
        sns.countplot(trn[col], alpha=0.5)
        plt.show()


    # scatterplot against target
    sns.jointplot('age', "target", data=trn, kind="reg")
    sns.jointplot('age', "target", data=trn, kind="hex")
    sns.jointplot('age', "target", data=trn, kind="kde")

    lb = LabelEncoder()
    ## Plot histogram for each features
    for col in trn.columns:
        print '='*50
        print 'Col : ', col
        f, ax = plt.subplots(figsize=(20, 15))
        x = lb.fit_transform(trn[col])
        sns.countplot(x, trn['target'], kind='kde')
        plt.show()


    # feature wise within trn
    x = trn['age']
    y = trn['renta']
    sns.jointplot(x, y, kind="kde")

    x = trn['age']
    y = lb.fit_transform(trn['indfall'])
    sns.jointplot(x, y, kind="kde")



    # with final data at hand, run correlation
    trn = pd.read_csv('../input/final_train_data.csv')
    f = {'antiguedad':['mean'],
         'sexo':['first']...}
    col_order = ['antiguedad',
                 'sexo',...]
    agg = trn.groupby('ncodpers').agg(f)[col_order]
    corr_mat = agg.corr(method='pearson')
    f, ax = plt.subplots(figsize=(20, 15))
    plt.matshow(corr_mat)


def clean(trn):
    col = 'ind_empleado'
    trn[col].fillna('NN', inplace=True)

    col = 'pais_residencia'
    trn[col].fillna('NN', inplace=True)

    col = 'sexo'
    trn[col].fillna('NN', inplace=True)

    col = 'age'
    trn[col].replace(' NA', 0, inplace=True)
    trn[col] = trn[col].astype(np.int64)

    col = 'fecha_alta'
    trn[col].fillna('2015-06-30', inplace=True)

    col = 'ind_nuevo'
    trn[col].fillna(-1, inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'antiguedad'
    trn[col].replace('     NA', 0, inplace=True)
    trn[col] = trn[col].astype(np.int64)
    trn[col].replace(-999999, 0, inplace=True)

    col = 'indrel'
    trn[col].fillna(0, inplace=True)
    trn[col].replace(99, 2, inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'ult_fec_cli_1t'
    trn[col].fillna('2015-06-30', inplace=True)

    col = 'indrel_1mes'
    trn[col].fillna(2, inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'tiprel_1mes'
    trn[col].fillna('NN', inplace=True)

    col = 'indresi'
    trn[col].fillna('NN', inplace=True)

    col = 'indext'
    trn[col].fillna('NN', inplace=True)

    col = 'conyuemp'
    trn[col].fillna('NN', inplace=True)

    col = 'canal_entrada'
    trn[col].fillna('NN', inplace=True)

    col = 'indfall'
    trn[col].fillna('NN', inplace=True)

    col = 'tipodom'
    # drop tipodom, tst has only one unique value

    col = 'cod_prov'
    trn[col].fillna(0, inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'nomprov'
    trn[col].fillna('NN', inplace=True)
    trn[col].replace('CORU\xc3\x91A, A', 'CORU', inplace=True)

    col = 'ind_actividad_cliente'
    trn[col].fillna(-1, inplace=True)
    trn[col] = trn[col].astype(int)

    col = 'renta'
    trn[col].fillna(0, inplace=True)

    col = 'segmento'
    trn[col].fillna('NN', inplace=True)

    trn.fillna(0, inplace=True)


if __name__=='__main__':
    main()



# notes
## trn vs tst joinplot
# some feature balances are different between trn and test! record them ALL here
# be aware that most of tst data is insignificant

# sexo : spread
# segmento : spread
# renta_lb : spread
# fecha_alta_dayofweek : spread

# ind_empleado : both in 4
# pais_residencia : both in 40
# ind_nuevo : both in 0
# indrel : both in 0
# indresi: both in 2
# indext : both in 1
# renta : both in 0
# indfall : both in 1

# age : trn widepsread, tst focused on 20s
# fecha_alta : trn widespread, tst focused in 6000
# antiguedad : trn widespread, tst focused in 40
# fecha_alta_year : trn spread, tst in 12.5
# fecha_alta_month : trn spread, tst in 8~10
# fecha_alta_week : trn spread, tst in 30~40
# fecha_alta_day : trn spread, tst in 5~10
# canal_entrada : trn three peak points, tst one peak point

# tiprel_1mes : trn focused in 1, tst spread in 1, 2
# cod_prov : trn one peak, tst three peaks
# ind_actividad : trn focused in 1, tst spread in 0 and 1
# na_count : trn in 0, tst in 0 and 1



## feature brief

# fecha_alta : use year, month, day, dayofweek
# age: main around 40
# sexo : V is 30% more
# renta : bell curve, na is lot
# segmento : distributed
# canal_entrada : long tail
# cod_prov : long tail
# nomprov : long tail
# tiprel_1mes : A is 90%
# indext : N is 95%
# ind_actividad_cliente : 1.0 is 90%
# ind_nuevo : 85% 0.0
# indfall : skewed
# tipodom : skewed
# indresi : too skewed
# pais_residencia : too skewed
# ind_empleado : too skewed
# indrel : too skewed
# indrel_1mes : too skewed

## feature ideas
# renta_le : label encoded renta [done]
# fecha_alta : year, month, day, week, dayofweek [done]
##### FROM HERE

# MAKE FEATURE ENGINEERING HERE
# ohe of targets in prev month
# didBuy last month
# did unbuy last month
# number of purchased targets (int)
# naCount [DONE]
#



## DROP FEATURES
# ncodpers
# fecha_dato
# ult_fec_cli_1t
# conyuemp
# nomprov

## TRIM TARGET VALUES
# Take TOP9 as predictor [0.0342363 as my goal]
# drop target  / / / due to their insignificance
'''
[2]  cco 0.0096681
[23] recibo 0.0086845
[18] tjcr 0.0041178
[17] reca 0.0032092
[22] nom 0.0021801
[21] nomina 0.0021478
[12] ecue 0.0019961
[4]  cno  0.0017839
[6]  ctma 0.0004488
-
valo 0.000278
ctop 0.0001949
ctpp 0.0001142
fond 0.000104
ctju 0.0000502
hip 0.0000161
plan 0.0000126
pres 0.0000054
cder 0.000009
viv 0
deco 0
deme 0
-
ahor 0
aval 0
dela 0
'''