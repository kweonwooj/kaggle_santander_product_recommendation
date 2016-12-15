# -*- encoding: utf-8 -*-
"""
@author: Kweonwoo Jung
@brief:
	input : trn
	output : label_sp for trn
# takes 22 min
"""
import numpy as np
import pandas as pd
from utils.log_utils import get_logger
import time

st = time.time()

LOG = get_logger('Preprocess-generate_labels.txt')
LOG.info('# Generating label_sp as initial preprocessing..')

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

f = open('../Data/Raw/train_ver2.csv', 'r')
f.readline()
g = open('../Data/Raw/labels.csv', 'w')
out = ','.join(target_cols) + '\n'
g.write(out)

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
	ncodper = tmp1[1]

	temp_sp_y = np.zeros(24).astype(int)
	if ncodper in cust:
		for i in range(24):
			if labels[i] == 1 and cust[ncodper][i] == 0:
				temp_sp_y[i] = 1
	else:
		temp_sp_y = labels
	cust[ncodper] = labels

	for i in range(24):
		out += str(temp_sp_y[i]) + ','
	out = out[:-1] + '\n'
	g.write(out)

	count += 1

	if count % 1000000 == 1:
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
