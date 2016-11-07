from __future__ import division
import datetime
import os
from collections import defaultdict
import time
import re
import operator
import numpy as np

def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def add_data_to_main_arrays(arr, best, history, all_count):
    ncodpers = arr[1]
    part = arr[24:]

    # check history
    if ncodpers in history:
        old_part = history[ncodpers]
        for i in range(24):
            if part[i] == '1' and old_part[i] == '0':
                all_count[0] += 1
                best[i][ncodpers] += 1
                best[i]['sum'] += 1
    else:
        for i in range(24):
            if part[i] == '1':
                all_count[0] += 1
                best[i][ncodpers] += 1
                best[i]['sum'] += 1
    history[ncodpers] = part

def get_predictions(arr1, best, history, all_count, n_uniq_user):
    predicted = []
    rank = dict()
    ncodpers = arr1[1]

    for i in range(24):
        prob = 1. * best[i][ncodpers] / best[i]['sum']
        rank[i] = prob
    rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    for i, k in enumerate(rank):
        if i == 7: break
        predicted.append(k[0])
    return predicted

def get_real_values(arr1, history):
    real = []
    ncodpers = arr1[1]
    arr2 = arr1[24:]

    for i in range(len(arr2)):
        if arr2[i] == '1':
            if ncodpers in history:
                if history[ncodpers][i] == '0':
                    real.append(i)
            else:
                real.append(i)
    return real

def run_solution():

    print('Preparing arrays...')
    f = open("../Data/Raw/train_ver2.csv", "r")
    # import header
    first_line = f.readline().strip()
    first_line = first_line.replace('"', '')
    # separate feature cols
    map_names = first_line.split(",")[24:]

    # history
    history = dict()
    history_valid = dict()
    all_count = [0]
    all_count_valid =[0]

    # Normal variables
    best = defaultdict(lambda: defaultdict(int)) # dict in dict storing count of each target per customer

    # Validation variables
    best_valid = defaultdict(lambda: defaultdict(int))

    valid_part = []
    # Calc counts
    total = 0
    while 1:
        # reading each rows
        line = f.readline()[:-1]
        total += 1

        if line == '':
            break
    
        # 'nomprov' has " as its separator
        tmp1 = line.split('"')
        arr = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')
        arr = [a.strip() for a in arr]
        if len(arr) != 48:
            print 'Error: len(arr) = {} !!!! {}'.format(len(arr), line)
            exit()

        # Normal part
        add_data_to_main_arrays(arr, best, history, all_count)

        # Valid part
        if arr[0] != '2016-05-28':
            add_data_to_main_arrays(arr, best_valid, history_valid, all_count_valid)
        else:
            valid_part.append(arr)

        if total % 1000000 == 0:
            print('Process {} lines ...'.format(total))

    f.close()

    print '='*50
    print('Sort best arrays...')
    print('all_count: ', all_count[0])
    print('all_count_valid: ', all_count_valid[0])
    print('Valid part: ', len(valid_part))

    map7 = 0.0
    total = 0
    print('Validation...')
    # for each valid user, iterate..
    for arr1 in valid_part:
        predicted = get_predictions(arr1, best_valid, history_valid, all_count_valid, len(valid_part))
        # Find real
        real = get_real_values(arr1, history_valid)

        # evalute map@7 score
        score = apk(real, predicted)
        map7 += score
        
        total += 1
        if total % 1000000 == 0:
            print('Process {} lines ...'.format(total))

    if len(valid_part) > 0:
        map7 /= len(valid_part)
    print('Predicted score: {}'.format(map7))

    return

    print('Generate submission...')
    sub_file = os.path.join('..','Output','Subm','submission_' + str(map7) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    out = open(sub_file, "w")
    f = open("../Data/Raw/test_ver2.csv", "r")
    f.readline()
    total = 0
    out.write("ncodpers,added_products\n")

    while 1:
        line = f.readline()[:-1]
        total += 1

        if line == '':
            break

        tmp1 = line.split('"')
        arr = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')
        arr = [a.strip() for a in arr]
        if len(arr) != 24:
            print 'Error: len(arr) = {} !!!! {}\n {}'.format(len(arr), line, total)
            exit()

        ncodpers = arr[1]
        out.write(ncodpers + ',')

        predicted = get_predictions(arr, best, history, all_count, 100000)

        for p in predicted:
            out.write(map_names[p] + ' ')

        if total % 1000000 == 0:
            print('Read {} lines ...'.format(total))
            # break

        out.write("\n")

    print('Total cases:', str(total))
    out.close()
    f.close()


if __name__ == "__main__":
    run_solution()
