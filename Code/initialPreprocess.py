# -*- encoding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief:
	input : trn
	output : label_sp for trn

ADD FIRST LINE AS TARGET COLS

"""
import numpy as np
import pandas as pd
from utils.log_utils import get_logger
import time

st = time.time()

LOG = get_logger('initialPreprocess.txt')
LOG.info('# Generating label_sp as initial preprocessing..')

dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']


df = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=target_cols, dtype=dtype_list)
ncodper = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=['ncodpers'], dtype='object')
df['ncodpers'] = ncodper

f = open('../Data/Raw/train_ver2.csv', 'r')
f.readline()
g = open('../Data/Raw/labels.csv', 'w')

cust = dict()
count = 0
while 1:
	line = f.readline()[:-1]
	out = ""

	if line == '':
		break

	tmp1 = line.split('"')
	tmp1 = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')
	tmp1 = [a.strip() for a in tmp1]

	labels = np.zeros(24).astype(int)
	for i, a in enumerate(tmp1[24:]):
		try:
			labels[i] = int(float(a))
		except:
			labels[i] = 0
	ncodper = tmp1[0]

	temp_sp_y = np.zeros(24).astype(int)
	if ncodper in cust:
		for i in range(24):
			if labels[i] == 1 and cust[ncodper][i] == 0:
				temp_sp_y[i] = 1
	else:
		temp_sp_y = labels
	cust[ncodper] = temp_sp_y

	for i in range(24):
		out += str(temp_sp_y[i]) + ','
	out = out[:-1] + '\n'
	g.write(out)

	count += 1

	if count % (df.shape[0]/10) == 1:
		LOG.info('# Processing {} lines..'.format(count))

f.close()
g.close()

en = time.time()
el = en-st
unit = 'sec'
if el > 60:
	el /= 60.
	unit = 'min'
if el > 60:
	el /= 60.
	unit = 'hour'
LOG.info('# Elapsed time : {} {}'.format(el, unit))
LOG.info('# DONE')


