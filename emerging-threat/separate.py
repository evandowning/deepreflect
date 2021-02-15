#!usr/bin/python3

import sys
import os
import argparse
import numpy as np
import time

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--x', help='x train', required=True)
    parser.add_argument('--fn', help='train fn', required=True)
    parser.add_argument('--addr', help='train addr', required=True)
    parser.add_argument('--family', help='family name', required=True)
    parser.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    xFN = args.x
    fnFN = args.fn
    addrFN = args.addr
    family = args.family
    outputFolder = args.output

    # Read in highlighted function addresses
    X_train = np.load(xFN)
    train_fn = np.load(fnFN)
    train_addr = np.load(addrFN)

    sys.stdout.write('Functions highlighted: {0}\n'.format(len(train_addr)))

    # Get indices of family
    index = [i for i,fn in enumerate(train_fn) if family == fn.split('/')[-2]]
    h = set(train_fn[index])

    sys.stdout.write('Number of {0} samples and functions: {1} and {2}\n'.format(family,len(h),len(index)))

    index = [i for i,fn in enumerate(train_fn) if family != fn.split('/')[-2]]
    h = set(train_fn[index])
    sys.stdout.write('Number of samples and functions (minus {0}): {1} and {2}\n'.format(family,len(h),len(index)))

    # Save arrays without family
    np.save(os.path.join(outputFolder,'x_train_minus.npy'),X_train[index])
    np.save(os.path.join(outputFolder,'train_fn_minus.npy'),train_fn[index])
    np.save(os.path.join(outputFolder,'train_addr_minus.npy'),train_addr[index])

    # Save arrays with family (just a copy of the original arrays)
    np.save(os.path.join(outputFolder,'x_train_plus.npy'),X_train)
    np.save(os.path.join(outputFolder,'train_fn_plus.npy'),train_fn)
    np.save(os.path.join(outputFolder,'train_addr_plus.npy'),train_addr)

if __name__ == '__main__':
    _main()
