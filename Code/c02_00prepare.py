# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file saves 
    - unique values of each feature column
    - unique values of each feature column per ncodpers (valid and train)
"""

import pandas as pd
import pickle
from utils.log_utils import get_logger

LOG = get_logger('quick_preprocess.txt')
"""
# import data
trn = pd.read_csv('../Data/Raw/train_ver2.csv')
vld_period = '2016-05-28'
vld = trn[trn.fecha_dato != vld_period].copy()

# unique values of each feature column
for col in trn.columns:
  LOG.info('Storing column: {}'.format(col))
  # get unique 

  unique_trn = trn[col].unique()
  unique_vld = vld[col].unique()

  # store in dict
  feat_trn = dict()
  feat_vld = dict()
  # get number of unique elements
  feat_trn['len'] = len(unique_trn)
  feat_trn['elements'] = unique_trn
  feat_vld['len'] = len(unique_vld)
  feat_vld['elements'] = unique_vld

  # store in pkl
  pickle.dump(feat_trn, open('../Data/Clean/unq_{}.pkl'.format(col),'wb'))
  pickle.dump(feat_vld, open('../Data/Clean/unq_{}_vld.pkl'.format(col),'wb'))
"""

#del trn
#del vld

vld_period = '2016-05-28'

do_cols = ['antiguedad','indrel','indrel_1mes','tiprel_1mes','indresi', \
           'indext','canal_entrada','indfall','cod_prov','nomprov', \
           'ind_actividad_cliente','segmento']

for col in do_cols: 
  LOG.info('Processing {}'.format(col))

  f = open('../Data/Raw/train_ver2.csv','r')
  first_line = f.readline().strip()
  first_line = first_line.replace('"','')
  map_names = first_line.split(',')[24:]

  total = 0
  feat_trn = dict()
  feat_vld = dict()
  while 1:
    line = f.readline()[:-1]
    total += 1

    if line == '':
      break

    # clean arr
    tmp1 = line.split('"')
    arr = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')
    arr = [a.strip() for a in arr]

    # assign names
    (fecha_dato, ncodpers, ind_empleado, pais_residencia, \
    sexo, age, fecha_alta, ind_nuevo, antiguedad, indrel, \
    ult_fec_cli_1t, indrel_1mes, tiprel_1mes, indresi, \
    indext, conyuemp, canal_entrada, indfall, tipodom, \
    cod_prov, nomprov, ind_actividad_cliente, renta, segmento, \
    ahor, aval, cco, cder, cno, tju, ctma, ctop, ctpp, deco, \
    deme, dela, ecue, fond, hip, plan, pres, reca, tjcr, valo, \
    viv, nomina, nom_pens, recibo) = arr

    if col == 'antiguedad': var = antiguedad
    elif col == 'indrel': var = indrel
    elif col == 'indrel_1mes': var = indrel_1mes
    elif col == 'tiprel_1mes': var = tiprel_1mes
    elif col == 'indresi': var = indresi
    elif col == 'indext': var = indext
    elif col == 'canal_entrada': var = canal_entrada
    elif col == 'indfall': var = indfall
    elif col == 'cod_prov': var = cod_prov
    elif col == 'nomprov': var = nomprov
    elif col == 'ind_actividad_cliente': var = ind_actividad_cliente 
    elif col == 'segmento': var = segmento
  
    # store unique users
    if fecha_dato != vld_period:
      # trn
      if ncodpers in feat_trn:
        x = feat_trn[ncodpers]
        if var not in x:
          x.append(var)
          feat_trn[ncodpers] = x
      else:
        feat_trn[ncodpers] = [var]
      # vld
      if ncodpers in feat_vld:
        x = feat_vld[ncodpers]
        if var not in x:
          x.append(var)
          feat_vld[ncodpers] = x
      else:
        feat_vld[ncodpers] = [var]
    else:
      # trn
      if ncodpers in feat_trn:
        x = feat_trn[ncodpers]
        if var not in x:
          x.append(var)
          feat_trn[ncodpers] = x
      else:
        feat_trn[ncodpers] = [var]

    if total % 1000000 == 0:
      LOG.info('Process {} lines ...'.format(total))

  # store in pkl
  pickle.dump(feat_trn, open('../Data/Clean/unq_{}_per_user.pkl'.format(col),'wb'))
  pickle.dump(feat_vld, open('../Data/Clean/unq_{}_per_user_vld.pkl'.format(col),'wb'))

