import pickle
from sklearn.preprocessing import StandardScaler
from utils.log_utils import get_logger

LOG = get_logger('ss.txt')

for i in range(464):
    f = open('../Data/Clean/xgb_vld_trn.csv','r')
    column = []
    while 1:
        line = f.readline()[:-1]

        if line == '':
            break

        arr = line.split(',')[i]
        try:
            arr = int(float(arr))
        except:
            arr = 0
        column.append(arr)

    LOG.info('{} / 463'.format(i))
    ss = StandardScaler()
    ss.fit(column)
    pickle.dump(ss, open('../Data/ss/ss_{}.pkl'.format(i),'wb'))
