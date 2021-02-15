#!usr/bin/python3

import sys
import os
import argparse
import numpy as np
import math
import random
import time

from collections import Counter

from joblib import dump

from sklearn.metrics.pairwise import pairwise_distances

from sklearn.preprocessing import normalize
from scipy import stats

from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics

from DBCV import DBCV
from scipy.spatial.distance import euclidean

from cluster_acfg import ACFG
from cluster_acfg_plus import ACFG_plus

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='dataset types help', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('acfg', help='acfg features')
    sp.set_defaults(cmd='acfg')
    sp.add_argument('--prediction', help='predictions file', required=True)
    sp.add_argument('--map', help='class map path', required=True)
    sp.add_argument('--data', help='data folder', required=True)
    sp.add_argument('--parse', help='parse folder', required=True)
    sp.add_argument('--output', help='output folder', required=True)
    sp.add_argument('--normalize', help='normalize features', required=False, default=False)
    sp.add_argument('--normalize-log', help='normalize log features', required=False, default=False)
    sp.add_argument('--area', help='takes neighboring ACFG feature vectors. +/- int(area) before and after highlighted block', type=int, required=False, default=-1)
    sp.add_argument('--area-type', help='type of area to get', required=False)

    sp = subparsers.add_parser('acfg_plus', help='acfg plus features')
    sp.set_defaults(cmd='acfg_plus')
    sp.add_argument('--data', help='autoencoder errors folder', required=True)
    sp.add_argument('--bndb-func', help='bndb function folder', required=True)
    sp.add_argument('--acfg', help='acfg folder', required=True)
    sp.add_argument('--func', help='use all basic blocks in each highlighted function', action='store_true')
    sp.add_argument('--window', help='use sequential basic blocks from one highlight to the next', action='store_true')
    sp.add_argument('--bb', help='use highlighted basic blocks', action='store_true')
    sp.add_argument('--avg', help='calculate average', action='store_true')
    sp.add_argument('--avgstdev', help='calculate average standard deviation', action='store_true')
    sp.set_defaults(func=False,window=False,bb=False,avg=False,avgstdev=False)
    sp.add_argument('--thresh', help='threshold', required=True)
    sp.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    dataset = args.cmd
    data_path = args.data
    outputFolder = args.output

    # Get arguments
    if dataset == 'acfg':
        predictionFN = args.prediction
        map_path = args.map
        parse_path = args.parse
        normalize = bool(args.normalize)
        log = bool(args.normalize_log)
        area = int(args.area)

        if area != -1:
            area_type = args.area_type

            if area_type not in ['avg', 'avgstdev', 'avgother']:
                sys.stderr.write('Error. Invalid type {0}\n'.format(area_type))
                sys.stderr.write('Should be either avg, avgstdev, or avgother\n')
                sys.exit(1)

    elif dataset == 'acfg_plus':
        bndb_func_path = args.bndb_func
        acfg_path = args.acfg
        normalize=False
        log=False

        # Determine which metric to use
        avgFlag = args.avg
        avgstdevFlag = args.avgstdev

        if avgFlag == avgstdevFlag:
            sys.stderr.write('Error. Choose either "avg" or "avgstdev".\n')
            sys.exit(1)

        # Determine which area to use
        funcFlag = args.func
        windowFlag = args.window
        bbFlag = args.bb
        c = Counter([funcFlag,windowFlag,bbFlag])

        if c[True] != 1:
            sys.stderr.write('Error. Choose either "func" or "window" or "bb".\n')
            sys.exit(1)

        thresh = float(args.thresh)

        sys.stdout.write('Using Threshold: {0}\n'.format(thresh))

    sample = dict()

    # Get each sample
    if dataset == 'acfg':
        sys.stdout.write('Only considering samples which got classified correctly by model\n')

        with open(predictionFN,'r') as fr:
            for line in fr:
                line = line.strip('\n')
                fn,confidence,true,predict,predict_conf = line.split(',')

                # If this sample was predicted correctly, we're interested in it
                if true == predict:
                    h = fn.split('/')[-1]
                    f = fn.split('/')[-2]

                    # Get location of binary
                    fn = os.path.join(data_path,f,h)

                    if f not in sample:
                        sample[f] = list()
                    sample[f].append((fn,true[1:-1]))

    elif dataset == 'acfg_plus':
        # Get bndb function files
        bndb_func_map = dict()
        for root,dirs,files in os.walk(bndb_func_path):
            for fn in files:
                bndb_func_map[fn[:-4]] = os.path.join(root,fn)

        # Get acfg files
        acfg_map = dict()
        for root,dirs,files in os.walk(acfg_path):
            for fn in files:
                acfg_map[fn[:-4]] = os.path.join(root,fn)

        # Get autoencoder error files
        for root,dirs,files in os.walk(data_path):
            for fn in files:
                family = root.split('/')[-1]

                if family not in sample:
                    sample[family] = list()

                # Retrieve bndb and acfg locations
                bndbfuncFN = bndb_func_map[fn[:-4]]
                acfgFN = acfg_map[fn[:-4]]

                sample[family].append((os.path.join(root,fn),bndbfuncFN,acfgFN,family))

    train = list()
    test = list()

    num_classes = 0

    sys.stdout.write('Number of samples in each family:\n')

    # NOTE: We're splitting up each individual family into train and test
    #       sets, but we're expecting that the clustering may reveal overlaps
    #       and common activities inter-family

    # Split sample files into train and test
    for l in sorted(sample, key=lambda x : len(sample[x]), reverse=True):
        # If large family, split 80/10
        if len(sample[l]) > 5:
            # 80% training
            index_train = math.ceil(len(sample[l])*0.8)
            # 20% testing
            index_test = len(sample[l])

        # If small family, split 50/50
        elif len(sample[l]) >= 2:
            # 80% training
            index_train = math.ceil(len(sample[l])*0.5)
            # 20% testing
            index_test = len(sample[l])

        # If too small
        else:
            # Put in training set
            index_train = len(sample[l])
            index_test = index_train

        num_classes += 1

        sys.stdout.write('    Class: {0}  Size: {1}\n'.format(l,len(sample[l])))

        # Randomize list
        random.shuffle(sample[l])

        # Store splits
        train.extend(sample[l][:index_train])
        test.extend(sample[l][index_train:index_test])

    sys.stdout.write('========================\n')
    sys.stdout.write('Number train: {0}\n'.format(len(train)))
    sys.stdout.write('Number test: {0}\n'.format(len(test)))
    sys.stdout.write('Number classes: {0}\n'.format(num_classes))
    sys.stdout.write('========================\n')

    if dataset == 'acfg':
        if area != -1:
            data = ACFG(train,test,map_path,parse_path,area,area_type)
        else:
            data = ACFG(train,test,map_path,parse_path)

    elif dataset == 'acfg_plus':
        data = ACFG_plus(train,test,thresh,avgFlag=avgFlag,avgstdevFlag=avgstdevFlag,funcFlag=funcFlag,windowFlag=windowFlag,bbFlag=bbFlag)

    X_train = np.array([])
    y_train = np.array([])
    X_test = np.array([])
    y_test = np.array([])

    # Data structures for parsing output of clusters
    train_fn = list()
    test_fn = list()
    train_addr = list()
    test_addr = list()

    total = 0

    start = time.time()

    # Get data from samples
    count = 0
    for fn,bb_addr,x,y in data.generator('train',1):
        if fn is None:
            break

        if len(X_train) == 0:
            X_train = x
            y_train = y
        else:
            X_train = np.vstack((X_train,x))
            y_train = np.vstack((y_train,y))

        count += 1

        train_fn.append(fn)
        train_addr.append(bb_addr)

    total += count
    sys.stdout.write('Total number of training samples: {0}\n'.format(count))

    # Get data from samples
    count = 0
    for fn,bb_addr,x,y in data.generator('test',1):
        if fn is None:
            break

        if len(X_test) == 0:
            X_test = x
            y_test = y
        else:
            X_test = np.vstack((X_test,x))
            y_test = np.vstack((y_test,y))

        count += 1

        test_fn.append(fn)
        test_addr.append(bb_addr)

    total += count
    sys.stdout.write('Total number of testing samples: {0}\n'.format(count))

    sys.stdout.write('Took {0} seconds to read in data\n'.format(time.time()-start))

    sys.stdout.write('Total number of samples: {0}\n'.format(total))

    # NOTE
    # Combine datasets (in case we do not want train/test sets)
    if len(X_test) > 0:
        X_train = np.vstack((X_train,X_test))
        y_train = np.vstack((y_train,y_test))
        train_fn.extend(test_fn)
        train_addr.extend(test_addr)

    # Normalize dataset if needbe
    if normalize:
        # From: https://stackoverflow.com/questions/44324781/normalize-numpy-ndarray-data
        X_train = X_train/np.linalg.norm(X_train, ord=np.inf, axis=0, keepdims=True)
    elif log:
        # Take zscore of log of values
        X_train = stats.zscore(np.log(X_train + 0.1), axis=0)

    # If any data is nan, replace it with 0's
    tmp = np.isnan(X_train)
    i,j = np.where(tmp)
    for index in range(len(i)):
        sys.stdout.write('Had nan values: {0},{1}: {2}\n'.format(str(i[index]),str(j[index]),train_fn[index]))
    X_train[tmp] = 0

    # Save x_train and y_train
    np.save(os.path.join(outputFolder,'x_train.npy'),X_train)
    np.save(os.path.join(outputFolder,'y_train.npy'),y_train)
    np.save(os.path.join(outputFolder,'train_fn.npy'),np.asarray(train_fn))
    np.save(os.path.join(outputFolder,'train_addr.npy'),np.asarray(train_addr))

    sys.stdout.write('Saved x_train and y_train contents\n')

if __name__ == '__main__':
    _main()
