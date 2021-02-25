#!usr/bin/python3

import sys
import os
import argparse
import numpy as np
import time
from collections import Counter

sys.path.append('../')
from dr import RoI

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--bndb-func', help='bndb function folder', required=True)
    parser.add_argument('--feature', help='feature folder', required=True)
    parser.add_argument('--mse', help='autoencoder mse folder', required=True)

    parser.add_argument('--func', help='use all basic blocks in each highlighted function', action='store_true')
    parser.add_argument('--window', help='use sequential basic blocks from one highlight to the next', action='store_true')
    parser.add_argument('--bb', help='use highlighted basic blocks', action='store_true')
    parser.add_argument('--avg', help='calculate average', action='store_true')
    parser.add_argument('--avgstdev', help='calculate average standard deviation', action='store_true')
    parser.set_defaults(func=False,window=False,bb=False,avg=False,avgstdev=False)

    parser.add_argument('--normalize', help='normalize features', required=False, default=None)
    parser.add_argument('--thresh', help='threshold', type=float, required=True)
    parser.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    mse_path = args.mse
    bndb_func_path = args.bndb_func
    feature_path = args.feature

    normalizeFN = args.normalize
    thresh = float(args.thresh)
    outputFolder = args.output

    # Determine which area to use
    funcFlag = args.func
    windowFlag = args.window
    bbFlag = args.bb
    c = Counter([funcFlag,windowFlag,bbFlag])
    if c[True] != 1:
        sys.stderr.write('Error. Choose either "func" or "window" or "bb".\n')
        sys.exit(1)

    # Determine which metric to use
    avgFlag = args.avg
    avgstdevFlag = args.avgstdev
    if avgFlag == avgstdevFlag:
        sys.stderr.write('Error. Choose either "avg" or "avgstdev".\n')
        sys.exit(1)

    sys.stdout.write('Using Threshold: {0}\n'.format(thresh))

    # Get bndb function files
    bndb_func_map = dict()
    for root,dirs,files in os.walk(bndb_func_path):
        for fn in files:
            bndb_func_map[fn[:-4]] = os.path.join(root,fn)

    # Get feature files
    feature_map = dict()
    for root,dirs,files in os.walk(feature_path):
        for fn in files:
            feature_map[fn[:-4]] = os.path.join(root,fn)

    # Get autoencoder error files
    sample = list()
    for root,dirs,files in os.walk(mse_path):
        for fn in files:
            mseFN = os.path.join(root,fn)

            # Retrieve bndb and feature locations
            funcFN = bndb_func_map[fn[:-4]]
            featureFN = feature_map[fn[:-4]]

            sample.append((mseFN,funcFN,featureFN))

    # Load dataset
    data = RoI(sample,thresh,normalizeFN,funcFlag=funcFlag,windowFlag=windowFlag,bbFlag=bbFlag,avgFlag=avgFlag,avgstdevFlag=avgstdevFlag)

    # Data for outputs
    output_x = np.array([])
    output_fn = list()
    output_addr = list()

    start = time.time()

    # Get data from samples
    count = 0
    for fn,bb_addr,x in data.roi_generator():
        sys.stdout.write('Number of functions highlighted: {0}\r'.format(count))
        sys.stdout.flush()

        if len(output_x) == 0:
            output_x = x
        else:
            output_x = np.vstack((output_x,x))

        output_fn.append(fn)
        output_addr.append(bb_addr)

        count += 1

    sys.stdout.write('\n')
    sys.stdout.write('Number of samples which had highlights: {0}\n'.format(len(set(output_fn))))
    sys.stdout.write('Took {0} seconds to retrieve RoI feature values\n'.format(time.time()-start))

    # If any data is nan, replace it with 0's
    tmp = np.isnan(output_x)
    i,j = np.where(tmp)
    for index in range(len(i)):
        sys.stdout.write('Had nan values: {0},{1}: {2}\n'.format(str(i[index]),str(j[index]),output_fn[index]))
    output_x[tmp] = 0

    # Save RoI feature data (to be used for clustering)
    np.save(os.path.join(outputFolder,'x.npy'),output_x)
    np.save(os.path.join(outputFolder,'fn.npy'),np.asarray(output_fn))
    np.save(os.path.join(outputFolder,'addr.npy'),np.asarray(output_addr))

if __name__ == '__main__':
    _main()
