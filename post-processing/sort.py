#!/usr/bin/python3

import sys
import argparse
import random
import numpy as np
from sklearn import metrics

import binaryninja as binja

sys.path.append('../')
from dr import RoI
sys.path.append('../grader/')
from roc import get_gt


def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mse', help='mse produced by autoencoder', required=True)
    parser.add_argument('--feature', help='features file', required=True)
    parser.add_argument('--bndb-func', help='bndb function file', required=True)

    parser.add_argument('--bndb', help='bndb file', required=True)

    parser.add_argument('--threshold', help='threshold', type=float, required=True)

    parser.add_argument('--sort-random', help='sort functions by MSE values randomly', action='store_true')
    parser.add_argument('--sort-addr', help='sort functions by address values (increasing order)', action='store_true')
    parser.add_argument('--sort-bb', help='sort functions by number of basic blocks (decreasing order)', action='store_true')
    parser.add_argument('--sort-mse', help='sort functions by MSE values (decreasing order)', action='store_true')
    parser.add_argument('--sort-callee', help='sort functions by number of callees (decreasing order)', action='store_true')

    parser.add_argument('--annotation', help='function annotations', required=True)

    args = parser.parse_args()

    # Store arguments
    mseFN = args.mse
    featureFN = args.feature
    funcFN = args.bndb_func
    bndbFN = args.bndb
    threshold = float(args.threshold)
    annotationFN = args.annotation

    mseSplit = mseFN.strip().split(' ')
    featureSplit = featureFN.strip().split(' ')
    funcSplit = funcFN.strip().split(' ')
    bndbSplit = bndbFN.strip().split(' ')
    annotationSplit = annotationFN.strip().split(' ')

    # Structure to store top-k precision values
    k = [10,50,100,200,500,1000]
    topk = dict()
    for i in k:
        topk[i] = list()

    # Calculate average over all samples (could be multiple passed)
    for i in range(len(mseSplit)):
        sample = list()
        sample.append([mseSplit[i],funcSplit[i],featureSplit[i]])

        # Load dataset
        data = RoI(sample,threshold,None)

        # Import database file
        bv = binja.BinaryViewType.get_view_of_file(bndbSplit[i])

        # Ground-truth labels
        sample_y = list()
        # MSE for each function
        sample_score = list()
        # Addresses for each function
        sample_addr = list()
        # Basic block counts for each function
        sample_bb = list()
        # Callee counts for each function
        sample_callee = list()

        # Get MSE values of highlighted functions
        # There will only be one function returned here
        for mse_func in data.function_highlight_generator():
            # Read in annotations
            gt_f_addr,gt_f_name = get_gt(annotationSplit[i])

            sys.stdout.write('Number of functions highlighted: {0}\n'.format(len(mse_func.keys())))

            for f_addr,t in mse_func.items():

                # Calculate average BB MSE values for this function
                s = 0
                for bb_addr,m in t:
                    s += m
                avg = s/len(t)

                if f_addr in gt_f_addr:
                    sample_y.append(1.0)
                else:
                    sample_y.append(0.0)

                sample_score.append(avg)

                sample_addr.append(f_addr)

                sample_bb.append(len(t))

                # Get function object and function callees
                function = bv.get_function_at(f_addr)
                callee_count = 0
                for callee in function.callees:
                    symbol_type = callee.symbol.type

                    if symbol_type in [0,1,2,4,5,6]:
                        callee_count += 1
                sample_callee.append(callee_count)

        # Convert to numpy arrays
        sample_y = np.asarray(sample_y)
        sample_score = np.asarray(sample_score)
        sample_addr = np.asarray(sample_addr)
        sample_bb = np.asarray(sample_bb)
        sample_callee = np.asarray(sample_callee)

        # Apply each sorting methodology
        index = None

        if args.sort_random is True:
            # Randomize
            index = np.arange(len(sample_y))
            np.random.shuffle(index)

        elif args.sort_addr is True:
            # Sort increasing function address values
            index = np.argsort(sample_addr)

        elif args.sort_bb is True:
            # Sort decreasing number of basic blocks per function
            index = np.argsort(sample_bb)[::-1]

        elif args.sort_mse is True:
            # Sort decreasing MSE values for each function
            index = np.argsort(sample_score)[::-1]

        elif args.sort_callee is True:
            # Sort decreasing number of callees per function
            index = np.argsort(sample_callee)[::-1]

        else:
            sys.stderr.write('Error, no sorting option chosen\n')
            sys.exit(1)

        # Calculations of top-k
        # Precision = TP/(TP+FP)
        for k in topk.keys():
            y_true = sample_y[index][:k] >= threshold
            y_pred = sample_score[index][:k] >= threshold   # Should all be True because the only functions returned are those at or above the threshold

            p = metrics.precision_score(y_true,y_pred) * 100

            topk[k].append(p)

    # Print top-k precision values (average them over multiple binaries if given)
    for k,s in topk.items():
        avg = sum(s)/len(s)
        sys.stdout.write('Top-{0}: Precision: {1}%\n'.format(k,avg))

if __name__ == '__main__':
    _main()
