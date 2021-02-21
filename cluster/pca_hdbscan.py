#!/usr/bin/python3

import sys
import argparse
import numpy as np
import time

from sklearn.decomposition import PCA

import hdbscan

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--x', help='x train', required=True)
    parser.add_argument('--fn', help='train fn', required=True)
    parser.add_argument('--addr', help='train addr', required=True)

    args = parser.parse_args()

    # Store arguments
    xFN = args.x
    fnFN = args.fn
    addrFN = args.addr

    highlight = dict()

    # Read in highlighted function addresses
    X = np.load(xFN)
    train_fn = np.load(fnFN)
    train_addr = np.load(addrFN)

    sys.stdout.write('Functions highlighted: {0}\n'.format(len(train_addr)))

    start = time.time()

    # Run PCA
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    sys.stdout.write('Running PCA took: {0} seconds\n'.format(time.time()-start))

    start = time.time()

    # Run clustering
    # https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X_pca)
    label = clusterer.labels_
    prob = clusterer.probabilities_

    sys.stdout.write('Running clustering took: {0} seconds\n'.format(time.time()-start))

    c = dict()
    # Print cluster stats and what samples belong to each
    for i in range(len(X)):
        if label[i] not in c:
            c[label[i]] = list()
        c[label[i]].append((train_fn[i],train_addr[i],prob[i]))

    sys.stdout.write('Number of clusters (including noise cluster): {0}\n'.format(len(c.keys())))

    sys.stdout.write('Cluster size: id: size\n')
    for k,v in sorted(c.items(), key=lambda x:len(x[1]), reverse=True):
        sys.stdout.write('{0}: {1}\n'.format(k,len(v)))

    sys.stdout.write('\n')
    sys.stdout.write('Cluster contents: filename address id probability\n')
    for k,v in sorted(c.items(), key=lambda x:len(x[1]), reverse=True):
        for fn,addr,p in v:
            sys.stdout.write('{0} {1} {2} {3}\n'.format(fn,addr,k,p))

if __name__ == '__main__':
    _main()
