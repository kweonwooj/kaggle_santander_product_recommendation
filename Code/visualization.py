# -*- coding:utf8 -*-
"""
@author: Kweonwoo Jung
@brief: stacked bar chart over time
	- capture some useful patterns
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
from utils.log_utils import get_logger
np.random.seed(7)

LOG = get_logger('visualization.txt')

LOG.info('# Begin saving stacked bar charts')

columns = ['ind_empleado','pais_residencia','sexo','age','ind_nuevo','antiguedad','nomprov','segmento']

colors = ["blue", "grey", "green", "purple", "black", "tomato", "navy", "red"]

for col in columns:
    LOG.info('# Column : {}'.format(col))
    cols = ['fecha_dato',col]

    df = pd.read_csv('../Data/Raw/sample_trn.csv', usecols=[col])
    trn = pd.read_csv('../Data/Raw/train_ver2.csv', usecols=cols)
    tst = pd.read_csv('../Data/Raw/test_ver2.csv', usecols=cols)

    xs = trn['fecha_dato'].unique()

    le = LabelEncoder()
    y = le.fit_transform(list(trn[col].fillna('-1').values))
    ys = np.zeros((len(le.classes_), len(xs)))
    for ro in range(len(le.classes_)):
        for co in range(len(xs)):
            ys[ro][co] = (trn[(y == ro)]['fecha_dato'] == xs[co]).sum()

    # stacked bar plot
    sns.set_style('white')
    sns.set_context({'figure.figsize': (24,10)})
    for i in range(ys.shape[0]):
	color = colors[np.random.randint(8)]
        plot = sns.barplot(x = xs, y=ys[i:,:].sum(axis=0), color = color)
    plt.legend([plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for i in range(ys.shape[0])], \
               [le.classes_[i] for i in range(ys.shape[0])], loc=1, ncol=ys.shape[0], prop={'size':16})
    plot.set_xlabel(col)
    plot.set_ylabel('Count')
    plot.set_title('Change in proportion of {}'.format(col))
    #Set fonts to consistent 16pt size
    for item in ([plot.xaxis.label, plot.yaxis.label] +
                 plot.get_xticklabels() + plot.get_yticklabels()):
        item.set_fontsize(16)
    plt.savefig('../Fig/{}.png'.format(col))
LOG.info('# DONE!')
