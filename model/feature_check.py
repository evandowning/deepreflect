#!/usr/bin/python3

import sys
import numpy as np

def summary(FN):
    summation = np.array([])

    with open(FN, 'r') as fr:
        for e,line in enumerate(fr):
            sys.stdout.write('Adding : {0}\r'.format(e+1))
            sys.stdout.flush()

            line = line.strip('\n')

            # Load data
            data = np.load(line)

            # Error, do not sum data
            if len(data) == 0:
                continue

            # Sum all basic block data
            data = np.sum(data,axis=0)

            if len(summation) == 0:
                summation = np.copy(data)
            else:
                summation = np.vstack((summation,data))
                summation = np.sum(summation,axis=0)

    sys.stdout.write('\n')

    return summation

def usage():
    sys.stderr.write('usage: python feature_check.py file.txt\n')
    sys.exit(1)

def _main():
    if len(sys.argv) != 2:
        usage()

    fn = sys.argv[1]

    summation = summary(fn)
    sys.stdout.write('{0}: {1}\n'.format(fn,summation))
    sys.stdout.write('\n')

if __name__ == '__main__':
    _main()
