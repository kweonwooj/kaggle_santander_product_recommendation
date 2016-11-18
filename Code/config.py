# -*- coding: utf -*-
"""
@author Kweonwoo Jung
@brief: config for kaggle_santander
"""

import os
import platform
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import os_utils
"""
to run above code,
1. add directory to $PYTHONPATH by adding
   export PYTHONPATH=~/kaggle_santander/Code:${PYTHONPATH}
   to ~/.bashrc
2. source ~/.bashrc
3. create blank __init__.py
"""

state = 'sample'
model = RandomForestClassifier(n_jobs=-1, random_state=7)

feature_cols = ["ind_empleado","pais_residencia","sexo","age", "ind_nuevo", "antiguedad", "nomprov", "segmento"]
dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

def generate_label_sp(LOG):
	df = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=target_cols, dtype=dtype_list)
	ncodper = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=['ncodpers'], dtype='object')
	df['ncodpers'] = ncodper
	f = open('../Data/Raw/sample_label_sp.csv','w')
	f.write('ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,ind_pres_fin_ult1,ind_reca_fin_ult1,ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1\n')

	cust = dict()	
	total = 0
	for ind, (run, row) in enumerate(df.iterrows()):

		ncodpers = row['ncodpers']
		t_row = row[target_cols].fillna(0).values.astype(int)

		sp_y = np.zeros(24).astype(int)
		if ncodpers in cust:
			for i in range(24):
				if t_row[i] == 1 and cust[ncodpers][i] == 0:
					sp_y[i] = 1
		else:
			sp_y = t_row
		cust[ncodpers] = sp_y

		sp_y = np.char.mod('%d', sp_y)
		out = ','.join(sp_y) + '\n'
		f.write(out)

		total += 1
		if total % 100000 == 0:
			LOG.info('# Processing {} lines..'.format(total))
	f.close()


'''

# ----- Overall ----- 
TASK = "all"
# for testing data processing and feature engineering
# TASK = "sample"
# SAMPLE_SIZE = 1000

# ----- Path -----
ROOT_DIR = ".."

DATA_DIR = "%s/Data"%ROOT_DIR
RAW_DIR = "%s/Raw"%DATA_DIR
SPLIT_DIR = "%s/Split"%DATA_DIR
CLEAN_DATA_DIR = "%s/Clean"%DATA_DIR

OUTPUT_DIR = "%s/Output"%ROOT_DIR
SUBM_DIR = "%s/Subm"%OUTPUT_DIR
MODEL_DIR = "%s/Model"%ROOT_DIR

LOG_DIR = "%s/Log"%ROOT_DIR
FIG_DIR = "%s/Fig"%ROOT_DIR
NOTEBOOK_DIR = "%s/Notebook"%ROOT_DIR

# Data
CLCK_TRN = "%s/Raw/train.csv"%DATA_DIR
CLCK_TST = "%s/Raw/test.csv"%DATA_DIR

# ----- Columns -----
FEATURES = ['fecha_dato','ncodpers','ind_empleado','pais_residencia','sexo','age','fecha_alta', \
            'ind_nuevo','antiguedad','indrel','ult_fec_cli_1t','indrel_1mes', \
            'tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall', \
            'tipodom','cod_prov','nomprov','ind_actividad_cliente','renta','segmento']
TARGETS = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1', \
           'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1', \
           'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1', \
           'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1', \
           'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1', \
           'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1', \
           'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1', \
           'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

# ----- Param -----
N_RUNS = 1
N_FOLDS = 1
MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.

# ----- Other -----
RANDOM_SEED = 7
PLATFORM = platform.system()
NUM_CORES = 12

# ---- Create Path -----
DIRS = []
DIRS += [NOTEBOOK_DIR, DATA_DIR, SPLIT_DIR, CLEAN_DATA_DIR, RAW_DIR]

DIRS += [OUTPUT_DIR, SUBM_DIR, MODEL_DIR]

DIRS += [LOG_DIR, FIG_DIR]
os_utils._create_dirs(DIRS)
'''
