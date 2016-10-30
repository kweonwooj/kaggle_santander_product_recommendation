__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import os
from collections import defaultdict
import time
import re
import operator


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

def get_hash(arr, type = 0):
    (fecha_dato, ncodpers, ind_empleado, pais_residencia, \
    sexo, age, fecha_alta, ind_nuevo, antiguedad, indrel, \
    ult_fec_cli_1t, indrel_1mes, tiprel_1mes, indresi, \
    indext, conyuemp, canal_entrada, indfall, tipodom, \
    cod_prov, nomprov, ind_actividad_cliente, renta, segmento) = arr[:24]

    if type == 0:
        return (pais_residencia, sexo, age, ind_nuevo, segmento, ind_empleado, ind_actividad_cliente, indresi)
    else:
        return (sexo, age, segmento)


def run_solution():

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
    
        # put \ separator in 'nomprov' because it contains ',' in its value
        tmp1 = line.split('"')
        arr = tmp1[0][:-1].split(',') + [tmp1[1]] + tmp1[2][1:].split(',')
        arr = [a.strip() for a in arr]
        if len(arr) != 48:
            print 'Error: len(arr) = {} !!!! {}'.format(len(arr), line)
            exit()

        ncodpers = arr[1]
        hash = get_hash(arr)
        part = arr[24:]

        # Normal part
        for i in range(24):
            # if product is purchased
            if part[i] == '1':
                if ncodpers in customer:
                    if customer[ncodpers][i] == '0':
                        best[hash][i] += 1
                        overallbest[i] += 1
                else:
                    best[hash][i] += 1
                    overallbest[i] += 1
        customer[ncodpers] = part # gets updated to latest customer status

        # Valid part
        if arr[0] != '2016-05-28':
            for i in range(24):
                if part[i] == '1':
                    if ncodpers in customer_valid:
                        if customer_valid[ncodpers][i] == '0':
                            best_valid[hash][i] += 1
                            overallbest_valid[i] += 1
                    else:
                        best_valid[hash][i] += 1
                        overallbest_valid[i] += 1 
            customer_valid[ncodpers] = part 
        else:
            valid_part.append(arr)

        if total % 1000000 == 0:
            print('Process {} lines ...'.format(total))
            #break

    f.close()

    print '='*50
    print('Sort best arrays...')
    print('Hashes num: ', len(best))
    print('Valid part: ', len(valid_part))

    # Normal
    # sorting normal 'best' array to out
    out = dict()
    for b in best:
        arr = best[b]
        srtd = sorted(arr.items(), key=operator.itemgetter(1), reverse=True)
        out[b] = srtd
    best = out
    # sorting 'overallbest' according to frequency
    overallbest = sorted(overallbest.items(), key=operator.itemgetter(1), reverse=True)

    # Valid
    out = dict()
    for b in best_valid:
        arr = best_valid[b]
        srtd = sorted(arr.items(), key=operator.itemgetter(1), reverse=True)
        out[b] = srtd
    best_valid = out
    overallbest_valid = sorted(overallbest_valid.items(), key=operator.itemgetter(1), reverse=True)

    map7 = 0.0
    print('Validation...')
    # for each valid user, iterate..
    for arr1 in valid_part:
        ncodpers = arr1[1]
        hash = get_hash(arr1)

        # if user has same feature in best_valid, arr = frequency of same demographic
        if hash in best_valid:
            arr = best_valid[hash]
        # if user is a new demographic, arr = overall best frequency
        else:
            arr = overallbest_valid

        predicted = []
        # iterate over most frequently purchased products
        for a in arr:
            # If user is not new
            if ncodpers in customer_valid:
                # skip if user already purchased the product
                if customer_valid[ncodpers][a[0]] == '1':
                    continue	
            # add product to predicted list 
            if a[0] not in predicted:
                predicted.append(a[0])
                if len(predicted) == 7:
                    break
        # if predicted list less than 7, fill them with overall best
        if len(predicted) < 7:
            for a in overallbest_valid:
                # If user is not new
                if ncodpers in customer_valid:
                    # skip if user already purchased the product
                    if customer_valid[ncodpers][a[0]] == '1':
                        continue
                # add product to predicted list
                if a[0] not in predicted:
                    predicted.append(a[0])
                    if len(predicted) == 7:
                        break

        # Find real
        real = []
        arr2 = arr1[24:]

        for i in range(len(arr2)):
            if arr2[i] == '1':
                # if existing user in valid data holds a new product, add to real
                if ncodpers in customer_valid:
                    if customer_valid[ncodpers][i] == '0':
                        real.append(i)
                # if new user, add product to real
                else:
                    real.append(i)

        # evalute map@7 score
        score = apk(real, predicted)
        map7 += score

    if len(valid_part) > 0:
        map7 /= len(valid_part)
    print('Predicted score: {}'.format(map7))

    print('Generate submission...')
    sub_file = os.path.join('..','Output','Subm','submission_' + str(map7) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    out = open(sub_file, "w")
    f = open("../Data/Raw/test_ver2.csv", "r")
    f.readline()
    total = 0
    count_empty = 0
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
        hash = get_hash(arr)

        out.write(ncodpers + ',')
        # If class exists output for class
        if hash in best:
            arr = best[hash]
        else:
            arr = overallbest
            count_empty += 1

        predicted = []
        for a in arr:
            # If user is not new
            if ncodpers in customer:
                if customer[ncodpers][a[0]] == '1':
                    continue
            if a[0] not in predicted:
                predicted.append(a[0])
                if len(predicted) == 7:
                    break
        if len(predicted) < 7:
            for a in overallbest:
                # If user is not new
                if ncodpers in customer:
                    if customer[ncodpers][a[0]] == '1':
                        continue
                if a[0] not in predicted:
                    predicted.append(a[0])
                    if len(predicted) == 7:
                        break

        for p in predicted:
            out.write(map_names[p] + ' ')

        if total % 1000000 == 0:
            print('Read {} lines ...'.format(total))
            # break

        out.write("\n")

    print('Total cases:', str(total))
    print('Empty cases:', str(count_empty))
    out.close()
    f.close()


if __name__ == "__main__":
    run_solution()
