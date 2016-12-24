'''
    This file selects subset of raw train and test data to be used for pipeline.

    2015-06 data with lag-5 is used as train.
    2016-06 data with lag-5 is used as test.
'''

import pandas as pd
import numpy as np
import os


# iterate over train target cols, store the customer's last possession in dictionary, and ultimately
# get new purchase data.
def generate_labels(LOG):

    target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                   'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                   'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                   'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                   'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

    # read train data line by line
    f = open('../root_input/train_ver2.csv', 'r', encoding='utf8')
    f.readline()

    # write labels.csv line by line
    if not os.path.isdir('./input'):
        os.mkdir('./input')
    g = open('./input/labels.csv', 'w')

    # write header
    out = ','.join(target_cols) + '\n'
    g.write(out)

    possession_log = dict()
    count = 0
    # iterate over train data
    while True:
        line = f.readline()[:-1]
        out = ""

        # break when all lines are read
        if line == '':
            break

        # split all values
        vals = line.split('"')
        vals = vals[0][:-1].split(',') + [vals[1]] + vals[2][1:].split(',')
        vals = [a.strip() for a in vals]

        # store current possession in labels
        possession = np.zeros(24).astype(int)
        for i, a in enumerate(vals[24:]):
            try:
                possession[i] = int(float(a))
            except:
                possession[i] = 0
        ncodper = vals[1]

        # extract new purchases only
        purchases = np.zeros(24).astype(int)
        if ncodper in possession_log:
            for i in range(24):
                if possession[i] == 1 and possession_log[ncodper][i] == 0:
                    purchases[i] = 1
        else:
            purchases = possession

        # store current possession of user in dictionary
        possession_log[ncodper] = possession

        for i in range(24):
            out += str(purchases[i]) + ','
        out = out[:-1] + '\n'
        g.write(out)

        count += 1

        if count % 1000000 == 0:
            LOG.info('# Processing {} lines..'.format(count))

    LOG.info('# Generated labels.csv..')
    # close file
    f.close()
    g.close()


# selects subset of raw data (lag-5 related data) from train and test
def prepare(LOG):

    # file paths
    trn_path = '../root_input/train_ver2.csv'
    tst_path = '../root_input/test_ver2.csv'
    labels_path = './input/labels.csv'

    LOG.info('# Loading data to generate lag..')
    # load full data
    trn = pd.read_csv(trn_path)
    tst = pd.read_csv(tst_path)
    labels = pd.read_csv(labels_path).astype(int)

    # set lag dates
    trn_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28']
    tst_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

    LOG.info('# Selecting ncodpers..')
    # select ncodpers
    temp = trn[trn['fecha_dato'] == '2015-06-28']['ncodpers']
    trn_ncodpers = temp[(labels[trn['fecha_dato'] == '2015-06-28'].sum(axis=1) > 0).values].values.tolist()
    tst_ncodpers = np.unique(tst['ncodpers']).tolist()

    LOG.info('# Trimming data..')
    # trim lag data with given date, given ncodpers
    trn_trim = trn[trn['fecha_dato'].isin(trn_dates)]
    trn_trim = trn_trim[trn_trim['ncodpers'].isin(trn_ncodpers)]
    tst_trim = trn[trn['fecha_dato'].isin(tst_dates)]
    tst_trim = tst_trim[tst_trim['ncodpers'].isin(tst_ncodpers)]

    LOG.info('# Melting data into 24-class..')
    # melt target into single 24-class
    fecha_dato = trn['fecha_dato']
    train_index = (labels[fecha_dato == '2015-06-28'].sum(axis=1) > 0)
    train_index = train_index[train_index == True]
    train = trn.ix[train_index.index]
    train.iloc[:, 24:] = labels.ix[train_index.index]

    trn_june = []
    for ind, (run, row) in enumerate(train.iterrows()):
        for i in range(24):
            if row[24 + i] == 1:
                temp = row[:24].values.tolist()
                temp.append(i)
                trn_june.append(temp)

    # define and save target separately
    target = pd.DataFrame(trn_june)[24].values.tolist()
    target = pd.DataFrame(target)

    # make full data set
    trn_june = pd.DataFrame(trn_june, columns=trn.columns[:25]).iloc[:, :-1]
    trn = pd.concat([trn_trim, trn_june], axis=0)
    tst = pd.concat([tst_trim, tst], axis=0)
    LOG.info('# intermediate : selected necessary rows only..')
    LOG.info('# trn : {} | target : {} | tst : {}'.format(trn.shape, target.shape, tst.shape))

    LOG.info('# Saving trn_lag, tst_lag, target in csv..')
    # save data
    trn.to_csv('./input/trn_lag.csv', index=False)
    tst.to_csv('./input/tst_lag.csv', index=False)
    pd.DataFrame(target).to_csv('./input/target.csv', index=False)


def prepare_data(LOG):

    LOG.info('# Generate new purchase data : labels.csv')
    generate_labels(LOG)

    LOG.info('# Prepare lag-5 data..')
    prepare(LOG)
