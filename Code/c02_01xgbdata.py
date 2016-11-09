# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file generates
	- validation data for xgboost
"""

from __future__ import division
import pandas as pd
import pickle
from utils.log_utils import get_logger

LOG = get_logger('xgb_data_vld.txt')

LOG.info('='*50)
LOG.info('Loading supplementary data ...')
# import supplementary data
ind_empleado_unq_vld = pickle.load(open('../Data/Clean/unq_ind_empleado_vld.pkl','rb'))
pais_residencia_unq_vld = pickle.load(open('../Data/Clean/unq_pais_residencia_vld.pkl','rb'))
conyuemp_unq_vld = pickle.load(open('../Data/Clean/unq_conyuemp_vld.pkl','rb'))
sexo_unq_vld = pickle.load(open('../Data/Clean/unq_sexo_vld.pkl','rb'))
indrel_unq_vld = pickle.load(open('../Data/Clean/unq_indrel_vld.pkl','rb'))
indrel_1mes_unq_vld = pickle.load(open('../Data/Clean/unq_indrel_1mes_vld.pkl','rb'))
tiprel_1mes_unq_vld = pickle.load(open('../Data/Clean/unq_tiprel_1mes_vld.pkl','rb'))
indresi_unq_vld = pickle.load(open('../Data/Clean/unq_indresi_vld.pkl','rb'))
indext_unq_vld = pickle.load(open('../Data/Clean/unq_indext_vld.pkl','rb'))
canal_entrada_unq_vld = pickle.load(open('../Data/Clean/unq_canal_entrada_vld.pkl','rb'))
indfall_unq_vld = pickle.load(open('../Data/Clean/unq_indfall_vld.pkl','rb'))
cod_prov_unq_vld = pickle.load(open('../Data/Clean/unq_cod_prov_vld.pkl','rb'))
nomprov_unq_vld = pickle.load(open('../Data/Clean/unq_nomprov_vld.pkl','rb'))
ind_actividad_cliente_unq_vld = pickle.load(open('../Data/Clean/unq_ind_actividad_cliente_vld.pkl','rb'))
segmento_unq_vld = pickle.load(open('../Data/Clean/unq_segmento_vld.pkl','rb'))

antiguedad_unq_vld_user = pickle.load(open('../Data/Clean/unq_antiguedad_per_user_vld.pkl','rb'))
indrel_unq_vld_user = pickle.load(open('../Data/Clean/unq_indrel_per_user_vld.pkl','rb'))
indrel_1mes_unq_vld_user = pickle.load(open('../Data/Clean/unq_indrel_1mes_per_user_vld.pkl','rb'))
tiprel_1mes_unq_vld_user = pickle.load(open('../Data/Clean/unq_tiprel_1mes_per_user_vld.pkl','rb'))
indresi_unq_vld_user = pickle.load(open('../Data/Clean/unq_indresi_per_user_vld.pkl','rb'))
indext_unq_vld_user = pickle.load(open('../Data/Clean/unq_indext_per_user_vld.pkl','rb'))
canal_entrada_unq_vld_user = pickle.load(open('../Data/Clean/unq_canal_entrada_per_user_vld.pkl','rb'))
indfall_unq_vld_user = pickle.load(open('../Data/Clean/unq_indfall_per_user_vld.pkl','rb'))
cod_prov_unq_vld_user = pickle.load(open('../Data/Clean/unq_cod_prov_per_user_vld.pkl','rb'))
nomprov_unq_vld_user = pickle.load(open('../Data/Clean/unq_nomprov_per_user_vld.pkl','rb'))
ind_actividad_cliente_unq_vld_user = pickle.load(open('../Data/Clean/unq_ind_actividad_cliente_per_user_vld.pkl','rb'))
segmento_unq_vld_user = pickle.load(open('../Data/Clean/unq_segmento_per_user_vld.pkl','rb'))

# import train data
path = '../Data/Raw/'
f = open('../Data/Raw/train_ver2.csv','r')
first_line = f.readline().strip()
first_line = first_line.replace('"','')
map_names = first_line.split(',')[24:]

# write to file
out_vld_trn = open('../Data/Clean/xgb_vld_trn.csv','w')
out_vld_tst = open('../Data/Clean/xgb_vld_tst.csv','w')
out_trn_count = 0
out_tst_count = 0

LOG.info('Iterating over train data..., splitting into 66:33')
prev_target = dict()
threshold = 9098206

total = 0
while 1:
    line = f.readline()[:-1]
    total += 1

    if line == '':
        break

    tmp1 = line.split('"')
    arr = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')
    arr = [a.strip() for a in arr]

    # feature engineer
    (fecha_dato, ncodpers, ind_empleado, pais_residencia, \
    sexo, age, fecha_alta, ind_nuevo, antiguedad, indrel, \
    ult_fec_cli_1t, indrel_1mes, tiprel_1mes, indresi, \
    indext, conyuemp, canal_entrada, indfall, tipodom, \
    cod_prov, nomprov, ind_actividad_cliente, renta, segmento, \
    ahor, aval, cco, cder, cno, tju, ctma, ctop, ctpp, deco, \
    deme, dela, ecue, fond, hip, plan, pres, reca, tjcr, valo, \
    viv, nomina, nom_pens, recibo) = arr

    ##########################################
    # row-wise computation 

    # new user in vld is skipped
    if total >= threshold and ncodpers not in prev_target:
        continue 

    ## fecha_alto
    # extract year and month
    fecha_dato_y, fecha_dato_m, _ = fecha_dato.split('-')
    fecha_dato_y = int(fecha_dato_y)
    fecha_dato_m = int(fecha_dato_m)

    ## ncodpers

    ## ind_empleado
    # one hot encoding of ind_empleado
    ind_empleado_ohe = [0]*ind_empleado_unq_vld['len']
    try:
        ind_empleado_ohe[list(ind_empleado_unq_vld['elements']) \
        .index(ind_empleado)] = 1
    except:
        ind_empleado_ohe[1] = 1
    
    ## pais_residencia
    # one hot encoding of pais_residencia
    pais_residencia_ohe = [0]*pais_residencia_unq_vld['len']
    try:
        pais_residencia_ohe[list(pais_residencia_unq_vld['elements']) \
        .index(pais_residencia)] = 1
    except:
        pais_residencia_ohe[1] = 1
        
    ## sexo
    # one hot encoding of sexo
    sexo_ohe = [0]*sexo_unq_vld['len']
    try:
        sexo_ohe[list(sexo_unq_vld['elements']) \
        .index(sexo)] = 1
    except:
        sexo_ohe[2] = 1
        
    ## age
    try:
        age = int(np.clip(int(float(age)),0,80)/5.)
    except:
        age = -1

    ## fecha_alta
    # fecha_alta month
    try:
        _, fecha_alta_m, _ = fecha_alta.split('-')
        fecha_alta_m = int(fecha_alta_m)
    except:
        fecha_alta_m = 0
    # fecha_dato - fecha_alta
    try:
        fecha_alta_d = datetime.strptime(fecha_alta, '%Y-%m-%d')
        fecha_dato = datetime.strptime(fecha_dato, '%Y-%m-%d')
        fecha_alta_d = (fecha_dato - fecha_alta_d).days / 30
    except:
        fecha_alta_d = -1

    ## ind_nuevo
    # as-is
    if '1' in ind_nuevo:
        ind_nuevo = 1
    else:
        ind_nuevo = 0

    ## antiguedad
    # n_unq
    n_antiguedad = len(antiguedad_unq_vld_user[ncodpers])
    # as-is
    try:
        antiguedad = int(antiguedad)
    except:
        antiguedad = -1

    ## indrel
    # n_unq
    n_indrel = len(indrel_unq_vld_user[ncodpers])
    # one hot encoding of indrel
    if indrel == '1' or '1.0': indrel = 1.0
    elif indrel == '99' or '99.0': indrel = 99.0
    else: indrel = np.nan
    indrel_ohe = [0]*indrel_unq_vld['len']
    try:
        indrel_ohe[list(indrel_unq_vld['elements']) \
        .index(indrel)] = 1
    except:
        indrel_ohe[1] = 1

    ## ult_fec_cli_1t
    # fecha_dato - ult_fec_cli_1t
    try:
        ult_fec_cli_1t = datetime.strptime(ult_fec_cli_1t, '%Y-%m-%d')
        fecha_dato = datetime.strptime(fecha_dato, '%Y-%m-%d')
        ult_fec = (fecha_dato - ult_fec_cli_1t).days / 30
    except:
        ult_fec = 0

    ## indrel_1mes
    # n_unq
    n_indrel_1mes = len(indrel_1mes_unq_vld_user[ncodpers])
    # one hot encoding of indrel_1mes
    indrel_1mes_ohe = [0]*indrel_1mes_unq_vld['len']
    try:
        indrel_1mes_ohe[list(indrel_1mes_unq_vld['elements']) \
        .index(indrel_1mes)] = 1
    except:
        indrel_1mes_ohe[1] = 1

    ## tiprel_1mes
    # n_unq
    n_tiprel_1mes = len(tiprel_1mes_unq_vld_user[ncodpers])
    # one hot encoding of tiprel_1mes
    tiprel_1mes_ohe = [0]*tiprel_1mes_unq_vld['len']
    try:
        tiprel_1mes_ohe[list(tiprel_1mes_unq_vld['elements']) \
        .index(tiprel_1mes)] = 1
    except:
        tiprel_1mes_ohe[2] = 1

    ## indresi
    # n_indresi
    n_indresi = len(indresi_unq_vld_user[ncodpers])
    # one hot encoding of indresi
    indresi_ohe = [0]*indresi_unq_vld['len']
    try:
        indresi_ohe[list(indresi_unq_vld['elements']) \
        .index(indresi)] = 1
    except:
        indresi_ohe[1] = 1

    ## indext
    # n_indext
    n_indext = len(indext_unq_vld_user[ncodpers])
    # one hot encoding of indext
    indext_ohe = [0]*indext_unq_vld['len']
    try:
        indext_ohe[list(indext_unq_vld['elements']) \
        .index(indext)] = 1
    except:
        indext_ohe[2] = 1

    ## conyuemp
    # one hot encoding of conyuemp
    conyuemp_ohe = [0]*conyuemp_unq_vld['len']
    if conyuemp == '': conyuemp_ohe[0] = 1
    else: conyuemp_ohe[list(conyuemp_unq_vld['elements']) \
    .index(conyuemp)] = 1

    ## canal_entrada
    # n_canal_entrada
    n_canal_entrada = len(canal_entrada_unq_vld_user[ncodpers])
    # one hot encoding of canal_entrada
    canal_entrada_ohe = [0]*canal_entrada_unq_vld['len']
    try:
        canal_entrada_ohe[list(canal_entrada_unq_vld['elements']) \
        .index(canal_entrada)] = 1
    except:
        canal_entrada_ohe[6] = 1

    ## indfall
    # n_indfall
    n_indfall = len(indfall_unq_vld_user[ncodpers])
    # one hot encoding of indfall
    indfall_ohe = [0]*indfall_unq_vld['len']
    try:
        indfall_ohe[list(indfall_unq_vld['elements']) \
        .index(indfall)] = 1
    except:
        indfall_ohe[1] = 1

    ## tipodom
    try:
        tipodom = int(tipodom)
    except:
        tipodom = 0

    ## cod_prov
    # n_cod_prov
    n_cod_prov = len(cod_prov_unq_vld_user[ncodpers])
    # one hot encoding of cod_prov
    cod_prov_ohe = [0]*cod_prov_unq_vld['len']
    try:
        cod_prov = float(cod_prov)
        cod_prov_ohe[list(cod_prov_unq_vld['elements']) \
        .index(cod_prov)] = 1
    except:
        cod_prov_ohe[39] = 1

    ## nomprov
    # n_nomprov
    n_nomprov = len(nomprov_unq_vld_user[ncodpers])
    # one hot encoding of nomprov
    nomprov_ohe = [0]*nomprov_unq_vld['len']
    try:
        nomprov_ohe[list(nomprov_unq_vld['elements']) \
        .index(nomprov)] = 1
    except:
        nomprov_ohe[39] = 1

    ## ind_actividad_cliente
    # n_ind_actividad_cliente
    n_ind_actividad_cliente = len(ind_actividad_cliente_unq_vld_user[ncodpers])
    # one hot encoding of ind_actividad_cliente
    ind_actividad_cliente_ohe = [0]*ind_actividad_cliente_unq_vld['len']
    try:
        ind_actividad_cliente = float(ind_actividad_cliente)
        ind_actividad_cliente_ohe[list(ind_actividad_cliente_unq_vld['elements']) \
        .index(ind_actividad_cliente)] = 1
    except:
        ind_actividad_cliente_ohe[2] = 1

    ## renta
    try:
        renta = int(float(renta))
        if renta < 4158:
            renta = '25%'
        elif renta < 8364:
            renta = '50%'
        elif renta < 13745:
            renta = '75%'
        else:
            renta = '100%'
    except:
        renta = '-1'
    renta_var = [0]*5
    renta_var[['25%','50%','75%','100%','-1'].index(renta)] = 1


    ## segmento
    # n_segmento
    n_segmento = len(segmento_unq_vld_user[ncodpers])
    # one hot encoding of ind_actividad_cliente
    segmento_ohe = [0]*segmento_unq_vld['len']
    try:
        segmento_ohe[list(segmento_unq_vld['elements']) \
        .index(segmento)] = 1
    except:
        segmento_ohe[2] = 1

    ## count of na
    na_count = arr.count('') + arr.count('nan') + arr.count('NA')

    ## count of products on previous timeframe
    product_count = 0
    if ncodpers in prev_target:
        for i in prev_target[ncodpers]:
            try:
                product_count += int(i)
            except:
                pass

    target = [0]*24
    ## target
    for i in range(24):
        if ncodpers in prev_target:
            if arr[24+i] == '1' and prev_target[ncodpers][i] == '0':
                target[i] = 1
        else:
            if arr[24+i] == '1':
                target[i] = 1
    prev_target[ncodpers] = arr[24:]

    # make feature
    feature = (fecha_dato_m, ind_empleado_ohe, pais_residencia_ohe, sexo_ohe, age, \
    fecha_alta_m, fecha_alta_d, ind_nuevo, n_antiguedad, antiguedad, n_indrel, \
    indrel_ohe, ult_fec, n_indrel_1mes, indrel_1mes_ohe, n_tiprel_1mes, \
    tiprel_1mes_ohe, n_indresi, indresi_ohe, n_indext, indext_ohe, conyuemp_ohe, \
    n_canal_entrada, canal_entrada_ohe, n_indfall, indfall_ohe, tipodom, n_cod_prov, \
    cod_prov_ohe, n_nomprov, nomprov_ohe, n_ind_actividad_cliente, \
    ind_actividad_cliente_ohe, renta_var, n_segmento, segmento_ohe, na_count, \
    product_count)

    # convert to single list
    feat_set = []
    for feat in feature:
        try: 
            n = len(feat)
            for i in range(n):
                feat_set.append(feat[i])
        except:
            feat_set.append(feat)

    # to write
    out = ''
    for i,j in enumerate(feat_set):
        if i == 0:
            out = str(j)
        else:
            out += ',' + str(j)
    for i in target:
        out += ',' + str(i)
    out += '\n'

    # write to trn or tst according to fecha_dato
    if total >= threshold:
        out_vld_tst.write(out)
        out_tst_count += 1
    else:
        out_vld_trn.write(out)
        out_trn_count += 1

    if total % 1000000 == 0:
        LOG.info('Processing {} lines ...'.format(total))
        LOG.info('trn_count {} vld_count {}'.format(out_trn_count, out_tst_count))

LOG.info('Final Count: trn_count {} vld_count {}'.format(out_trn_count, out_tst_count))

out_vld_trn.close()
out_vld_tst.close() 
