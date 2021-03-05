#!/usr/bin/python3

import sys
import argparse
import time

import binaryninja as binja

from roc import get_gt

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bndb', help='bndb file', required=True)
    parser.add_argument('--annotation', help='function annotations', required=True)

    args = parser.parse_args()

    # Store arguments
    bndbFN = args.bndb
    annotationFN = args.annotation

    start = time.time()

    # Import database file
    bv = binja.BinaryViewType.get_view_of_file(bndbFN)

    sys.stdout.write('{0} Importing BNDB file took {1} seconds\n'.format(bndbFN,time.time()-start))

    # Read in annotations
    gt_f_addr,gt_f_name = get_gt(annotationFN)

    # BB counts
    bb_mal = list()
    bb_ben = list()

    # Internal function call counts
    int_mal = list()
    int_ben = list()

    # External function call counts
    ext_mal = list()
    ext_ben = list()

    # Data function call counts (https://api.binary.ninja/binaryninja.types.Symbol.html)
    data_mal = list()
    data_ben = list()

    # For each function, make observations
    for func in bv.functions:
        f_addr = func.start

        # Get number of BBs
        bbs = func.basic_blocks
        bb_num = len(bbs)

        int_num = 0
        ext_num = 0
        data_num = 0

        # Get callees (functions that this function calls)
        for callee in func.callees:
            symbol_type = callee.symbol.type

            # If internal function
            if symbol_type == 0:
                int_num += 1
            # If external function
            elif symbol_type in [1,2,4,5,6]:
                ext_num += 1
            # If data call
            elif symbol_type == 3:
                data_num += 1
            # Doesn't exist, but here for whatever future symbol types exist
            else:
                continue

        # If malicious
        if f_addr in gt_f_addr:
            bb_mal.append(bb_num)
            int_mal.append(int_num)
            ext_mal.append(ext_num)
            data_mal.append(data_num)
        # Else
        else:
            bb_ben.append(bb_num)
            int_ben.append(int_num)
            ext_ben.append(ext_num)
            data_ben.append(data_num)

    # Print information
    sys.stdout.write('Average number of BBs per function: Mal: {0} | Ben: {1}\n'.format(sum(bb_mal)/len(bb_mal),sum(bb_ben)/len(bb_ben)))
    sys.stdout.write('Average number of internal calls per function: Mal: {0} | Ben: {1}\n'.format(sum(int_mal)/len(int_mal),sum(int_ben)/len(int_ben)))
    sys.stdout.write('Average number of external calls per function: Mal: {0} | Ben: {1}\n'.format(sum(ext_mal)/len(ext_mal),sum(ext_ben)/len(ext_ben)))
    sys.stdout.write('Average number of data calls per function: Mal: {0} | Ben: {1}\n'.format(sum(data_mal)/len(data_mal),sum(data_ben)/len(data_ben)))

if __name__ == '__main__':
    _main()
