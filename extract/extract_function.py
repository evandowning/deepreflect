#!/usr/bin/python3

import sys
import os
import argparse
import time

import binaryninja as binja

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--bndb', help='bndb file', required=True)
    parser.add_argument('--output', help='output file', required=True)

    args = parser.parse_args()

    # Store arguments
    bndbFN = args.bndb
    outFN = args.output

    # If bndb file doesn't exist
    if not os.path.exists(bndbFN):
        sys.stderr.write('{0} does not exist\n'.format(bndbFN))
        sys.exit(1)

    # If function file already exists
    if os.path.exists(outFN):
        sys.stderr.write('{0} already exists\n'.format(outFN))
        sys.exit(1)

    start = time.time()

    # Get functions & bb's in binary
    bv = binja.BinaryViewType.get_view_of_file(bndbFN)

    sys.stdout.write('{0}: Took {1} seconds to load BinaryNinja file\n'.format(bndbFN,time.time()-start))

    start = time.time()

    # Get folder of function file
    # Create it if it doesn't exist
    root = os.path.dirname(outFN)
    if not os.path.exists(root):
        os.makedirs(root)

    with open(outFN,'w') as fw:
        # Iterate over each function
        for func in bv.functions:
            f_name = func.name
            f_type = -1
            f_type_name = 'None'
            if func.symbol is not None:
                f_type = func.symbol.type
                f_type_name = func.symbol.type.name

            bbs = func.basic_blocks

            # Iterate over each basic block
            for bb in bbs:
                # Start of function address, start of basic block address,
                # function's name, and its symbol type & symbol type name.
                fw.write('{0} {1} {2} {3} {4}\n'.format(func.start,bb.start,f_name,f_type,f_type_name))

    sys.stdout.write('{0}: Took {1} seconds to finish processing\n'.format(bndbFN,time.time()-start))

if __name__ == '__main__':
    _main()
