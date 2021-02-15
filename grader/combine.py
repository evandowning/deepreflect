#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def _main():
    if len(sys.argv) < 3:
        sys.stderr.write('usage: python combine.py data1.npz data2.npz ... out.npz\n')
        sys.exit(2)

    # Get output filename
    outFN = sys.argv[-1]

    y = np.array([])
    score = np.array([])
    addr = np.array([])

    # For each file
    for fn in sys.argv[1:-1]:
        a = np.load(fn)

        if len(y) == 0:
            y = a['y']
            score = a['score']
            addr = a['addr']
        else:
            y = np.append(y,a['y'])
            score = np.append(score,a['score'])
            addr = np.append(addr,a['addr'])

    np.savez('{0}'.format(outFN),
             y=np.asarray(y),
             score=np.asarray(score),
             addr=np.asarray(addr))

if __name__ == '__main__':
    _main()
