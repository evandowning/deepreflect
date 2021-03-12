#!/usr/bin/python3

import sys
import os
import argparse
import numpy as np

sys.path.append('../')
from dr import RoI

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--bndb-func', help='bndb function folder', required=True)
    parser.add_argument('--feature', help='feature folder', required=True)

    parser.add_argument('--roiFN', help='RoI fn file', required=True)
    parser.add_argument('--roiAddr', help='RoI addr file', required=True)

    parser.add_argument('--thresh', help='threshold', type=float, required=True)
    parser.add_argument('--output', help='output file', required=True)

    args = parser.parse_args()

    # Store arguments
    funcFolder = args.bndb_func
    featFolder = args.feature
    roiFN = args.roiFN
    roiAddr = args.roiAddr

    threshold = float(args.thresh)
    outFN = args.output

    # Load RoI data
    X_fn = np.load(roiFN)
    X_addr = np.load(roiAddr)

    # Read in score data (for storing in database, not for clustering)
    sample = list()
    for mseFN in X_fn:
        base = '/'.join(mseFN.split('/')[-2:])
        funcFN = os.path.join(funcFolder,base[:-3]+'txt')
        featFN = os.path.join(featFolder,base)
        sample.append([mseFN,funcFN,featFN])

    data = RoI(sample,threshold,None)

    # Get MSE values of highlighted functions
    count = 0
    func_score = dict()
    for mse_func,mseFN in data.function_highlight_generator():
        sys.stderr.write('Processing functions: {0}/{1}\r'.format(count+1,len(X_addr)))
        sys.stderr.flush()

        for f_addr,t in mse_func.items():
            # Calculate average RoI MSE values for this function
            s = 0
            for bb_addr,m in t:
                s += m
            avg = s/len(t)

            if mseFN not in func_score.keys():
                func_score[mseFN] = dict()
            func_score[mseFN][f_addr] = avg

        count += 1
    sys.stderr.write('\n')

    # Construct numpy array in proper order
    X_score = list()
    for e,mseFN in enumerate(X_fn):
        addr = int(X_addr[e],16)

        #TODO - why does this happen?
        if addr not in func_score[mseFN]:
            sys.stderr.write('{0} {1} {2}\n'.format(e, mseFN, hex(addr)))
            X_score.append(-1)
            continue

        X_score.append(func_score[mseFN][addr])
    X_score = np.asarray(X_score)

    # Save data
    np.save(outFN,X_score)

if __name__ == '__main__':
    _main()
