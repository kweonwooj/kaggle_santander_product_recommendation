import datetime
import os
from collections import defaultdict
import time
import re
import operator
import numpy as np
from datetime import datetime

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

def get_hash(arr, type = 0, selected_n=None, bools=None):
    (fecha_dato, ncodpers, ind_empleado, pais_residencia, \
    sexo, age, fecha_alta, ind_nuevo, antiguedad, indrel, \
    ult_fec_cli_1t, indrel_1mes, tiprel_1mes, indresi, \
    indext, conyuemp, canal_entrada, indfall, tipodom, \
    cod_prov, nomprov, ind_actividad_cliente, renta, segmento) = arr[:24]

    if type == 1:
        return (pais_residencia, sexo, age, ind_nuevo, ind_empleado, segmento, ind_actividad_cliente, indresi)

    # generate tuple from selected_n
    hash = tuple([arr[:24][i] for i in selected_n])

    ### enriching hashing function is key to improvement ###
    # deceased would not buy anymore
    if indfall == 'S':
        return (ncodpers, indfall)
    if bools['indfall'] == 1:
        hash + (indfall,)

    # existing purchase counts
    p_count = []
    for i in arr[24:]:
        try:
            p_count.append(int(float(i)))
        except:
            p_count.append(0)
    p_count = sum(p_count)
    if bools['p_count'] == 1:
        hash + (p_count,)

    # bin age values here
    try: 
        age = int(np.clip(int(age),0,80)/5.)
    except:
        age = -1
    if bools['age'] == 1:
        hash + (age,)

    # ult_fec_cli_1t - fecha_dato
    try:
        ult_fec_cli_1t = datetime.strptime(ult_fec_cli_1t, '%Y-%m-%d')
        fecha_dato = datetime.strptime(fecha_dato, '%Y-%m-%d')
        ult_fec = (fecha_dato - ult_fec_cli_1t).days / 30
    except:
        ult_fec = '-1'
    if bools['ult_fec'] == 1:
        hash + (ult_fec,)

    # renta
    # 25% : 4158 / 50% : 8364 / 75% : 13745
    try:
        renta = int(float(renta))
        if renta < 4158:
            renta = '25%'
        elif renta < 8364:
            renta = '50%'
        elif renta < 13745:
            renta = '75%'
        else:
            renta = '100%'
    except:
        renta = '-1'
    if bools['renta'] == 1:
        hash + (renta,)

    return hash

def add_data_to_main_arrays(arr, best, overallbest, customer, selected_n, bools):
    ncodpers = arr[1]
    hash1 = get_hash(arr,0,selected_n,bools)
    hash2 = get_hash(arr,1)
    part = arr[24:]
    for i in range(24):
        if '1' in part[i]:
            if ncodpers in customer:
                if '1' not in customer[ncodpers][i]:
                    best[hash1][i] += 1
                    best[hash2][i] += 1
                    overallbest[i] += 1
            else:
                best[hash1][i] += 1
                best[hash2][i] += 1
                overallbest[i] += 1
    customer[ncodpers] = part

def sort_main_arrays(best, overallbest):
    out = dict()
    for b in best:
        arr = best[b]
        srtd = sorted(arr.items(), key=operator.itemgetter(1), reverse=True)
        out[b] = srtd
    best = out
    overallbest = sorted(overallbest.items(), key=operator.itemgetter(1), reverse=True)
    return best, overallbest

def get_predictions(arr1, best, overallbest, customer, selected_n, bools):
    predicted = []

    hash1 = get_hash(arr1,0,selected_n,bools)
    hash2 = get_hash(arr1,1)
    ncodpers = arr1[1]

    # hash1
    if len(predicted) < 7:
        if hash1 in best:
            for a in best[hash1]:
                # if user is not new
                if ncodpers in customer:
                    if '1' in customer[ncodpers][a[0]]:
                      continue
                if a[0] not in predicted:
                    predicted.append(a[0])
                    if len(predicted) == 7:
                        break

    # hash2
    if len(predicted) < 7:
        if hash2 in best:
            for a in best[hash2]:
                if ncodpers in customer:
                    if '1' in customer[ncodpers][a[0]]:
                      continue
                if a[0] not in predicted:
                    predicted.append(a[0])
                    if len(predicted) == 7:
                        break

    # overall
    if len(predicted) < 7:
        for a in overallbest:
            if ncodpers in customer:
                if '1' in customer[ncodpers][a[0]]:
                    continue
            if a[0] not in predicted:
                predicted.append(a[0])
                if len(predicted) == 7:
                    break

    return predicted

def get_real_values(arr1, customer):
    real = []
    ncodpers = arr1[1]
    arr2 = arr1[24:]

    for i in range(len(arr2)):
        if '1' in arr2[i]:
            if ncodpers in customer:
                if '1' not in customer[ncodpers][i]:
                    real.append(i)
            else:
                real.append(i)
    return real

def run_solution(selected_n, bools, LOG, train=False):

    print('Preparing arrays...')
    f = open("../Data/Raw/train_ver2.csv", "r")
    # import header
    first_line = f.readline().strip()
    first_line = first_line.replace('"', '')
    # separate feature cols
    map_names = first_line.split(",")[24:]

    # Normal variables
    customer = dict()
    best = defaultdict(lambda: defaultdict(int)) # dict in dict storing count of each target per customer
    overallbest = defaultdict(int) # count of each target overall

    # Validation variables
    customer_valid = dict()
    best_valid = defaultdict(lambda: defaultdict(int))
    overallbest_valid = defaultdict(int)

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

        if train:
            # Normal part
            add_data_to_main_arrays(arr, best, overallbest, customer, selected_n, bools)

        else:
            # Valid part
            if arr[0] != '2016-05-28':
                add_data_to_main_arrays(arr, best_valid, overallbest_valid, customer_valid, selected_n, bools)
            else:
                valid_part.append(arr)

        if total % 1000000 == 0:
            print('Process {} lines ...'.format(total))
            LOG.info('Process {} lines ...'.format(total))
            #break

    f.close()

    print '='*50
    print('Sort best arrays... (Train: {})'.format(train))

    if train:
        # Normal
        # sorting normal 'best' array to out
        best, overallbest = sort_main_arrays(best, overallbest)

    else:
        # Valid
        best_valid, overallbest_valid = sort_main_arrays(best_valid, overallbest_valid)

        map7 = 0.0
        print('Validation...')
        LOG.info('Validation...')
        # for each valid user, iterate..
        for arr1 in valid_part:
            predicted = get_predictions(arr1, best_valid, overallbest_valid, customer_valid, selected_n, bools)
            # Find real
            real = get_real_values(arr1, customer_valid)

            # evalute map@7 score
            score = apk(real, predicted)
            map7 += score

        if len(valid_part) > 0:
            map7 /= len(valid_part)
        print('Predicted score: {}'.format(map7))

        return len(best), map7

    print('Generate submission...')
    sub_file = os.path.join('..','Output','Subm','submission_' + str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
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

        predicted = get_predictions(arr, best, overallbest, customer, selected_n, bools)

        for p in predicted:
            out.write(map_names[p] + ' ')

        if total % 1000000 == 0:
            print('Read {} lines ...'.format(total))
            # break

        out.write("\n")

    print('Total cases:', str(total))
    out.close()
    f.close()
    return 0, 0

