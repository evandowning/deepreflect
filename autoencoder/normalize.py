#!/usr/bin/python3

import sys
import argparse

sys.path.append('../')
from dr_feature import DR

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='training set files', required=True)
    parser.add_argument('--test', help='testing set files', required=True)
    parser.add_argument('--output', help='output file', required=True)
    parser.add_argument('--max-bb', help='max number of basic blocks to consider', type=int, required=False, default=20000)

    args = parser.parse_args()

    # Store arguments
    trainFN = args.train
    testFN = args.test
    outputFN = args.output
    max_len = int(args.max_bb)

    # Import dataset
    data = DR(trainFN,testFN,max_len,None)

    # Get maximum values
    data.get_max(outputFN)

if __name__ == '__main__':
    _main()
