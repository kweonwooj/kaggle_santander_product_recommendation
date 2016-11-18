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
                overallbest[i] += 1
    customer[ncodpers] = targets

# sort list
overallbest = sorted(overallbest.items(), key=operator.itemgetter(1), reverse=True)

# this is the overallbest result
"""
[(2, 651903), 
(23, 250196),
(22, 129034), 
(21, 113702), 
(7, 110395), 
(18, 105292), 
(4, 96956), 
(12, 86981), 
(11, 49735), 
(17, 47566), 
(8, 38630), 
(19, 25193), 
(6, 18124), 
(13, 17640), 
(5, 9030), 
(15, 8083), 
(9, 6718), 
(14, 5082), 
(20, 3304), 
(16, 2504), 
(10, 1870), 
(3, 443), 
(0, 90), 
(1, 23)]
"""

pickle.dump(overallbest, open('../Data/Clean/overallbest.pkl','wb'))
