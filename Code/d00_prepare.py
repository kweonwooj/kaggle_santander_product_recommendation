# -*- coding:utf-8 -*-
"""
@author: Kweonwoo Jung
@brief: this file generates
	- sample_trn, sample_vld that is 1/10 of ncodpers
	- trn, vld
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from utils.log_utils import get_logger
import d00_config
import warnings

LOG = get_logger('d00_prepare.txt')
LOG.info('# Preprocessing : Sample (1/10)\t Validate(trn/vld)\t Submission (full train/test)')

class PREV():
    def __init__(self):
        self.prev_target = dict()
        self.prev_cust = dict()  

    def reset(self):
        self.prev_target = dict()
        self.prev_cust = dict()  

def get_df_idx(df, col, values):
    return np.where(df[col].isin(values))[0]

target_cols = d00_config.target_cols
mapping_dict = d00_config.mapping_dict

def get_d_val(row,col,prev_cust,ncodper):
    try:
        return 1 - int(row[col] == prev_cust[ncodper][col])
    except:
        return 0

def process(df, prefix, state, prev_target, prev_cust):

    LOG.info('# Processing {}_{}.csv'.format(prefix,state))
    f = open('../Data/Raw/{}_{}.csv'.format(prefix,state),'w')
    f.write('product_count,n_change,na_count,fd_y,fd_m,fd_m_int,fa_y,fa_m,fa_m_int,fd_fa,uf_y,uf_m,uf_m_int,fd_uf,d_u_f,d_age,age,antiguedad,d_a,renta,d_r,cod_prov,d_c_p,ind_empleado,d_i_e,pais_residencia,d_p_r,sexo,ind_nuevo,d_i_n,indrel,d_il,indrel_1mes,d_i_1,tiprel_1mes,d_t_1,indresi,d_ii,indext,d_it,conyuemp,d_cp,canal_entrada,d_c_e,indfall,d_fall,tipodom,d_tm,nomprov,d_nv,ind_actividad_cliente,d_i_a_c,segmento,d_so,ncodper,ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1\n')

    for ind, (run, row) in enumerate(df.iterrows()):
        # feature engineer
        out = ''
        ncodper = row['ncodpers']    

        ## SPECIAL ## 

        # product_count
        product_count = 0
        if ncodper in prev_target:
            for i in prev_target[ncodper]:
                product_count += int(i)
        else:
            product_count = -1

        # number of changes from its previous state
        n_change = 0
        if ncodper in prev_cust:
            for i,item in enumerate(prev_cust[ncodper]):
                try:
                    if row[i] != item:
                        n_change += 1
                except:
                    pass
        else:
            n_change = -1

        # na_count
        na_count = row.isnull().sum()

        ## DATE ##

        # fecha_dato
        col = 'fecha_dato'
        fd_y, fd_m, _ = row[col].split('-')
        fd_y = int(fd_y)
        fd_m = int(fd_m)
        fd_m_int = fd_y * 12 + fd_m

        # fecha_alta
        col = 'fecha_alta'
        try:
            fa_y, fa_m, _ = row[col].split('-')
            fa_y = int(fa_y)
            fa_m = int(fa_m)
            fa_m_int = fa_y * 12 + fa_m
        except:
            fa_y, fa_m, fa_m_int = 0, 0, 0

        # fd - fa
        fd_fa = fd_m_int - fa_m_int

        # ult_fec_cli_1t
        col = 'ult_fec_cli_1t'
        try:
            uf_y, uf_m, _ = row[col].split('-')
            uf_y = int(uf_y)
            uf_m = int(uf_m)
            uf_m_int = uf_y * 12 + uf_m
        except:
            uf_y, uf_m, uf_m_int = 0, 0, 0

        # fd - uf
        fd_uf = fd_m_int - uf_m_int

        d_u_f = get_d_val(row,col,prev_cust,ncodper)


        ## NUMERICAL ##
        row.fillna(-99, inplace=True)

        # age
        col = 'age'
        d_age = get_d_val(row,col,prev_cust,ncodper)
        try:
            age = int(np.clip(int(float(row[col])),0,80)/5.)
        except:
            age = -1

        # antiguedad
        col = 'antiguedad'
        try:
            antiguedad = int(float(row[col]))
        except:
            antiguedad = -99
        d_a = get_d_val(row,col,prev_cust,ncodper)

        # renta
        col = 'renta'
        renta = int(float(row[col]))
        d_r = get_d_val(row,col,prev_cust,ncodper)

        # cod_prov
        col = 'cod_prov'
        cod_prov = int(float(row[col]))
        d_c_p = get_d_val(row,col,prev_cust,ncodper)


        ## CATEGORICAL ##

        # ind_empleado
        col = 'ind_empleado'
        ind_empleado = mapping_dict[col][row[col]]
        d_i_e = get_d_val(row,col,prev_cust,ncodper)

        # pais_residencia
        col = 'pais_residencia'
        pais_residencia = mapping_dict[col][row[col]]
        d_p_r = get_d_val(row,col,prev_cust,ncodper)

        # sexo
        col = 'sexo'
        sexo = mapping_dict[col][row[col]]

        # ind_nuevo
        col = 'ind_nuevo'
        ind_nuevo = mapping_dict[col][row[col]]
        d_i_n = get_d_val(row,col,prev_cust,ncodper)

        # indrel
        col = 'indrel'
        indrel = mapping_dict[col][row[col]]
        d_il = get_d_val(row,col,prev_cust,ncodper)

        # indrel_1mes
        col = 'indrel_1mes'
        indrel_1mes = mapping_dict[col][row[col]]
        d_i_1 = get_d_val(row,col,prev_cust,ncodper)

        # tiprel_1mes
        col = 'tiprel_1mes'
        tiprel_1mes = mapping_dict[col][row[col]]
        d_t_1 = get_d_val(row,col,prev_cust,ncodper)

        # indresi
        col = 'indresi'
        indresi = mapping_dict[col][row[col]]
        d_ii = get_d_val(row,col,prev_cust,ncodper)

        # indext
        col = 'indext'
        indext = mapping_dict[col][row[col]]
        d_it = get_d_val(row,col,prev_cust,ncodper)

        # conyuemp
        col = 'conyuemp'
        conyuemp = mapping_dict[col][row[col]]
        d_cp = get_d_val(row,col,prev_cust,ncodper)

        # canal_entrada
        col = 'canal_entrada'
        canal_entrada = mapping_dict[col][row[col]]
        d_c_e = get_d_val(row,col,prev_cust,ncodper)

        # indfall
        col = 'indfall'
        indfall = mapping_dict[col][row[col]]
        d_fall = get_d_val(row,col,prev_cust,ncodper)

        # tipodom
        col = 'tipodom'
        tipodom = mapping_dict[col][row[col]]
        d_tm = get_d_val(row,col,prev_cust,ncodper)

        # nomprov
        col = 'nomprov'
        nomprov = mapping_dict[col][row[col]]
        d_nv = get_d_val(row,col,prev_cust,ncodper)

        # ind_actividad_cliente
        col = 'ind_actividad_cliente'
        ind_actividad_cliente = mapping_dict[col][row[col]]
        d_i_a_c = get_d_val(row,col,prev_cust,ncodper)

        # segmento
        col = 'segmento'
        segmento = mapping_dict[col][row[col]]
        d_so = get_d_val(row,col,prev_cust,ncodper)

        # adjust target so that newly purchase is recorded
        targets = np.zeros(24).astype(np.int8)
        if ncodper in prev_target and (prefix!='submission' or state!='vld'):
            for i in range(24):
                if row[24:][i] == 1 and prev_target[ncodper][i] == 0:
                    targets[i] = 1
        else:
            targets = row[24:].values.astype(np.int8)

        prev_target[ncodper] = np.nan_to_num(row[24:]).astype(np.int8)
        prev_cust[ncodper] = row[:24]

        # ncopder
        ncodper = int(float(row['ncodpers']))


        #### Additional Feature Engineering ####
        ## number of unique historical values ##

        feat = (product_count,n_change,na_count,fd_y,fd_m,fd_m_int,fa_y,fa_m, \
         fa_m_int,fd_fa,uf_y,uf_m,uf_m_int,fd_uf,d_u_f,d_age,age,antiguedad, \
         d_a,renta,d_r,cod_prov,d_c_p,ind_empleado,d_i_e,pais_residencia, \
         d_p_r,sexo,ind_nuevo,d_i_n,indrel,d_il,indrel_1mes,d_i_1,tiprel_1mes, \
         d_t_1,indresi,d_ii,indext,d_it,conyuemp,d_cp,canal_entrada,d_c_e, \
         indfall,d_fall,tipodom,d_tm,nomprov,d_nv,ind_actividad_cliente, \
         d_i_a_c,segmento,d_so,ncodper)

        if ind % 100000 == 0:
            LOG.info('#    Processing {} lines...'.format(ind))
            #break

        out += str([i for i in np.hstack((np.array(feat),targets))]).strip('[]')

        out += '\n'
        f.write(out)
    f.close()  
    LOG.info('# Processed {}_{}.csv'.format(prefix,state))

def preprocess(trn, vld, prefix):
  prev = PREV()
  prev_target = prev.prev_target
  prev_cust = prev.prev_cust

  process(trn, prefix, 'trn', prev_target, prev_cust)
  process(vld, prefix, 'vld', prev_target, prev_cust)

def sample(df):
  ## SAMPLE ##
  LOG.info('# Serializing SAMPLE')
  # split ncodeprs into 1/10
  ncodpers = df.ncodpers.unique()
  _, X_sample, _, _ = train_test_split(ncodpers, range(ncodpers.shape[0]), \
                                       test_size=0.1, random_state=7)
  # get sample sets
  index = get_df_idx(df, 'ncodpers', X_sample)
  sample = df.iloc[index]
  # split into trn,vld
  trn = sample[sample.fecha_dato != '2016-05-28']
  vld = sample[sample.fecha_dato == '2016-05-28']

  LOG.info('# trn.shape {} vld.shape {}'.format(trn.shape, vld.shape))

  # feature engineering
  prefix = 'sample'
  preprocess(trn,vld,prefix)
  LOG.info('# DONE! Serialized SAMPLE')
  
def validate(df):
  ## SAMPLE ##
  LOG.info('# Serializing VALIDATE')
  
  # split into trn,vld
  trn = df[df.fecha_dato != '2016-05-28']
  vld = df[df.fecha_dato == '2016-05-28']

  LOG.info('# trn.shape {} vld.shape {}'.format(trn.shape, vld.shape))

  # feature engineering
  prefix = 'validate'
  preprocess(trn,vld,prefix)
  LOG.info('# DONE! Serialized VALIDATE')

def submission(df):
  ## SAMPLE ##
  LOG.info('# Serializing SUBMISSION')
  vld = pd.read_csv('../Data/Raw/test_ver2.csv')

  LOG.info('# trn.shape {} vld.shape {}'.format(df.shape, vld.shape))

  # feature engineering
  prefix = 'submission'
  preprocess(df,vld,prefix)
  LOG.info('# DONE! Serialized SUBMISSION')

def main():

  # import data
  path = '../Data/Raw/'
  df = pd.read_csv(path+'train_ver2.csv')

  sample(df)
  validate(df)
  submission(df)

if __name__=='__main__':
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    main()
