# -*- coding: utf8 -*-
"""
@author: Kweonwoo Jung
@brief: get overallbest
"""

# getting the overallbest purchase products

from collections import defaultdict
import operator
import pickle

g = open('../Data/Raw/train_ver2.csv','r')
fline = g.readline()

customer = dict()
overallbest = defaultdict(int)

while 1:
    line = g.readline()[:-1]
    
    if line =='':
        break
        
    tmp1 = line.split("\"")
    arr = tmp1[0][:-1].split(",") + [tmp1[1]] + tmp1[2][1:].split(',')
    arr = [a.strip() for a in arr]
    
    targets = arr[24:]
    ncodpers = arr[1]
    
    for i in range(24):
        if targets[i] == '1':
            if ncodpers in customer:
                if customer[ncodpers][i] == '0':
                    overallbest[i] += 1
            else:
                overallbest[i]
    customer[ncodpers] = targets

# sort list
overallbest = sorted(overallbest.items(), key=operator.itemgetter(1), reverse=True)

# this is the overallbest result
"""
[(23, 153205),
 (22, 84767),
 (21, 73799),
 (2, 69997),
 (18, 69311),
 (4, 37187),
 (12, 26378),
 (11, 12707),
 (17, 9238),
 (6, 7002),
 (19, 4850),
 (7, 3882),
 (13, 3699),
 (9, 3074),
 (8, 2420),
 (15, 618),
 (5, 490),
 (10, 250),
 (16, 147),
 (3, 136),
 (14, 75),
 (20, 70),
 (1, 4),
 (0, 2)]
"""

pickle.dump(overallbest, open('../Data/Clean/overallbest.pkl','wb'))
