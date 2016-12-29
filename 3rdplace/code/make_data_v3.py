'''
    This file generates train, count data from raw_data.

    train_{date}.csv includes lag-1 products(20) + non_product ('tiprel_1mes', 'ind_actividad_cliente')
    count_{date].csv includes sum of product status change, 0len for all past dates

    correct output shape of each files is as below:
        count_2015-06-28.csv : shape : (628603, 100)
        count_2015-12-28.csv : shape : (904294, 100)
        count_2016-01-28.csv : shape : (908930, 100)
        count_2016-02-28.csv : shape : (914329, 100)
        count_2016-03-28.csv : shape : (919070, 100)
        count_2016-04-28.csv : shape : (923414, 100)
        count_2016-05-28.csv : shape : (926663, 100)
        count_2016-06-28.csv : shape : (929615, 100)

        train_2015-06-28.csv : shape : (628603, 62)
        train_2015-12-28.csv : shape : (904294, 62)
        train_2016-01-28.csv : shape : (908930, 62)
        train_2016-02-28.csv : shape : (914329, 62)
        train_2016-03-28.csv : shape : (919070, 62)
        train_2016-04-28.csv : shape : (923414, 62)
        train_2016-05-28.csv : shape : (926663, 62)
        test_2016-06-28.csv : shape : (929615, 42)
'''


import pandas as pd
import numpy as np
import os

def make_data(LOG):

    LOG.info('# Creating ../input directory')
    # create input directory
    if not os.path.exists('../input'):
        os.mkdir('../input')

    LOG.info('# Loading train data')
    # drop fecha_alta, ult_fec_cli_1t, tipodom, cod_prov, ind_ahor_fin_ult1, ind_aval_fin_ult1, ind_deco_fin_ult1 and ind_deme_fin_ult1
    cols = ["fecha_dato","ncodpers","ind_empleado","pais_residencia","sexo","age","fecha_alta","ind_nuevo","antiguedad","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","renta","segmento","ind_ahor_fin_ult1","ind_aval_fin_ult1","ind_cco_fin_ult1","ind_cder_fin_ult1","ind_cno_fin_ult1","ind_ctju_fin_ult1","ind_ctma_fin_ult1","ind_ctop_fin_ult1","ind_ctpp_fin_ult1","ind_deco_fin_ult1","ind_deme_fin_ult1","ind_dela_fin_ult1","ind_ecue_fin_ult1","ind_fond_fin_ult1","ind_hip_fin_ult1","ind_plan_fin_ult1","ind_pres_fin_ult1","ind_reca_fin_ult1","ind_tjcr_fin_ult1","ind_valo_fin_ult1","ind_viv_fin_ult1","ind_nomina_ult1","ind_nom_pens_ult1","ind_recibo_ult1"]
    cols_to_remove = ['fecha_alta','ult_fec_cli_1t','tipodom','cod_prov','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1']
    cols_to_use = [col for col in cols if col not in cols_to_remove]
    data = pd.read_csv('../../root_input/train_ver2.csv',usecols=cols_to_use, dtype={'indrel_1mes':str, 'conyuemp':str})

    LOG.info('# Fetching date_list, product_list[20]')
    # save all date_list, product_list
    date_list = np.unique(data['fecha_dato']).tolist()
    date_list.append('2016-06-28')
    product_list = data.columns[data.shape[1]-20:].tolist()

    # 2015-06, 2015-12 ~ 2016-06
    dates = [i for i in range(11,18)]
    dates.append(5)


    LOG.info('# Inner join with last month')
    ### data 1 : inner join with last month ###
    for i in dates:
        LOG.info('# Month : {}'.format(date_list[i]))
        if date_list[i] != '2016-06-28':
            # select current month
            out = data[data.fecha_dato == date_list[i]].reset_index(drop=True)

            # select last month
            temp = data[data.fecha_dato == date_list[i - 1]][
                product_list + ['ncodpers', 'tiprel_1mes', 'ind_actividad_cliente']].reset_index(drop=True)

            # join
            out = out.merge(temp, on='ncodpers', suffixes=('', '_last'))

            # save
            out.to_csv('../input/train_{}.csv'.format(date_list[i]), index=False)
        else:
            # import test (2016-06)
            cols = ["fecha_dato", "ncodpers", "ind_empleado", "pais_residencia", "sexo", "age", "fecha_alta", "ind_nuevo",
                    "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes", "indresi", "indext", "conyuemp",
                    "canal_entrada", "indfall", "tipodom", "cod_prov", "nomprov", "ind_actividad_cliente", "renta",
                    "segmento"]
            cols_to_remove = ['fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'cod_prov']
            cols_to_use = [col for col in cols if col not in cols_to_remove]
            out = pd.read_csv('../root_input/test_ver2.csv', usecols=cols_to_use,
                              dtype={'indrel_1mes': str, 'conyuemp': str}).reset_index(drop=True)

            # select last month (2016-05)
            temp = data[data.fecha_dato == date_list[i - 1]][
                product_list + ['ncodpers', 'tiprel_1mes', 'ind_actividad_cliente']].reset_index(drop=True)

            # join
            out = out.merge(temp, on='ncodpers', suffixes=('', '_last'))

            # save
            out.to_csv('../input/test_{}.csv'.format(date_list[i]), index=False)


    LOG.info('# Count the change of index for all past')
    ### data 2 : count the change of index ###
    for i in dates:
        LOG.info('# Month : {}'.format(date_list[i]))
        if date_list[i] != '2016-06-28':
            # current month
            out = data[data.fecha_dato == date_list[i]]['ncodpers']
            out = pd.DataFrame({'ncodpers': out, 'a': 1})

            # last month
            temp = data[data.fecha_dato == date_list[i - 1]]['ncodpers']
            temp = pd.DataFrame({'ncodpers': temp, 'a': 1})

            # only use ncodpers that appeared in both months
            out = out.merge(temp, on='ncodpers')[['ncodpers']]
        else:
            # get test data
            out = pd.read_csv('../../root_input/test_ver2.csv', usecols=['ncodpers'])

        for j, product in enumerate(product_list):
            LOG.info('# Product # {} : {}'.format(j, product))
            # fetch all data from the past
            temp = data[data.fecha_dato.isin(date_list[:i])][['fecha_dato', 'ncodpers', product]]
            temp = temp.sort(['ncodpers', 'fecha_dato'])

            # calculate lag-1 feature (change in product status)
            t1 = temp.ncodpers == temp.shift().ncodpers
            t2 = temp.shift()[product] == 0
            t3 = temp[product] == 0
            t4 = temp[product] == 1
            t5 = temp.shift()[product] == 1

            temp['n00'] = t1 & t2 & t3
            temp['n01'] = t1 & t2 & t4
            temp['n10'] = t1 & t5 & t3
            temp['n11'] = t1 & t5 & t4

            # sum product status per ncodpers
            count = temp.groupby('ncodpers')['n00', 'n01', 'n10', 'n11'].sum().reset_index()
            count.columns = ['ncodpers', product + '_00', product + '_01', product + '_10', product + '_11']
            count[product + '_0len'] = 0

            for date in date_list[:i]:
                temp2 = temp[temp.fecha_dato == date]
                temp2 = count.set_index('ncodpers').join(temp2.set_index('ncodpers'))[
                    ['fecha_dato', product, 'n00', 'n01', 'n10', 'n11']].reset_index()
                flag = temp2[product] == 0
                count[product + '_0len'] = (count[product + '_0len'] + 1) * flag
            out = out.set_index('ncodpers').join(count.set_index('ncodpers')).reset_index()
        out.drop(['ncodpers'], axis=1, inplace=True)
        out.to_csv('../input/count_{}.csv'.format(date_list[i]), index=False)