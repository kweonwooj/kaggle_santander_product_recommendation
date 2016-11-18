

import os

##### PARAMETER #####

# state = ['sample', 'validate', 'submission']
state = 'sample'

cmd = 'python preprocess.py'
os.system(cmd)

cmd = 'python fit_eval.py'
os.system(cmd)

cmd = 'python predict.py'
os.system(cmd)
