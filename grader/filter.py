import sys
import os
import argparse
import numpy as np
import time

import binaryninja as binja

# Treat functions with small number of basic blocks as benign
def filter_size(dataFN,bv,thresh):
    # Load DeepReflect results
    deepreflect_result = np.load(dataFN)
    dr_addr = deepreflect_result['addr']
    dr_y = deepreflect_result['y']
    dr_score = deepreflect_result['score']

    # Get addresses below this threshold
    result = dict()
    for a in dr_addr:
        func = bv.get_function_at(a)

        # Get number of BBs
        bbs = func.basic_blocks
        bb_num = len(bbs)

        result[a] = bb_num

    addr = list()
    score = list()
    label = list()

    # For each address, determine if function is < threshold
    for e,a in enumerate(dr_addr):
        l = dr_y[e]

        # If address is under this threshold, label it as benign
        if result[a] < thresh:
            s = 0.0
        # Else, score is DR score
        else:
            s = dr_score[e]

        addr.append(a)
        score.append(s)
        label.append(l)

    return addr,score,label

# Treat functions with small number of function callees as benign
def filter_callee(dataFN,bv,thresh):
    # Load DeepReflect results
    deepreflect_result = np.load(dataFN)
    dr_addr = deepreflect_result['addr']
    dr_y = deepreflect_result['y']
    dr_score = deepreflect_result['score']

    # Get addresses below this threshold
    result = dict()
    for a in dr_addr:
        # Get function
        func = bv.get_function_at(a)

        int_num = 0
        ext_num = 0

        # Get callees (functions that this function calls)
        for callee in func.callees:
            symbol_type = callee.symbol.type

            # If internal function
            if symbol_type == 0:
                int_num += 1
            # If external function
            elif symbol_type in [1,2,4,5,6]:
                ext_num += 1
            # If data call (or any other future defined call)
            else:
                continue

        result[a] = int_num+ext_num

    addr = list()
    score = list()
    label = list()

    # For each address, determine if function is < threshold
    for e,a in enumerate(dr_addr):
        l = dr_y[e]

        # If address is under this threshold, label it as benign
        if result[a] < thresh:
            s = 0.0
        # Else, score is DR score
        else:
            s = dr_score[e]

        addr.append(a)
        score.append(s)
        label.append(l)

    return addr,score,label

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data file produced by roc.py', required=True)
    parser.add_argument('--bndb', help='bndb file', required=True)

    parser.add_argument('--size', help='function size (number of BBs) threshold', type=int, required=True)
    parser.add_argument('--out-size', help='output data file after filter via size', required=True)

    parser.add_argument('--callee', help='number of function callees threshold', type=int, required=True)
    parser.add_argument('--out-callee', help='output data file after filter via callee', required=True)

    args = parser.parse_args()

    # Store arguments
    dataFN = args.data
    bndbFN = args.bndb

    # Load BNDB file
    start = time.time()
    bv = binja.BinaryViewType.get_view_of_file(bndbFN)
    sys.stdout.write('{0} Importing BNDB file took {1} seconds\n'.format(bndbFN,time.time()-start))

    # Filter by size
    thresh = int(args.size)
    outFN = args.out_size
    addr,score,label = filter_size(dataFN,bv,thresh)
    np.savez(outFN,
             y=np.asarray(label),
             score=np.asarray(score),
             addr=np.asarray(addr))

    # Filter by callees
    thresh = int(args.callee)
    outFN = args.out_callee
    addr,score,label = filter_callee(dataFN,bv,thresh)
    np.savez(outFN,
             y=np.asarray(label),
             score=np.asarray(score),
             addr=np.asarray(addr))

if __name__ == '__main__':
    _main()
