#!/usr/bin/python3

import sys
import os
import argparse
import configparser
import numpy as np
import time
import hashlib

sys.path.append('../')
from dr import RoI

from sklearn.decomposition import PCA

import hdbscan

import psycopg2

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='config file', required=True)
    args = parser.parse_args()

    # Store arguments
    cfgFN = args.cfg

    # Read config file
    config = configparser.ConfigParser()
    config.read(cfgFN)
    xFN = config['data']['x']
    fnFN = config['data']['fn']
    addrFN = config['data']['addr']
    scoreFN = config['data']['score']
    dbName = config['db']['name']
    dbUser = config['db']['username']
    dbPass = config['db']['password']

    args = parser.parse_args()

    highlight = dict()

    # Read in highlighted function addresses
    X = np.load(xFN)
    X_fn = np.load(fnFN)
    X_addr = np.load(addrFN)
    X_score = np.load(scoreFN)

    sys.stdout.write('Functions highlighted: {0}\n'.format(len(X_addr)))

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
        c[label[i]].append((X_fn[i],X_addr[i],prob[i],X_score[i]))

    sys.stdout.write('Number of clusters (including noise cluster): {0}\n'.format(len(c.keys())))

    sys.stdout.write('Cluster size: id: size\n')
    for k,v in sorted(c.items(), key=lambda x:len(x[1]), reverse=True):
        sys.stdout.write('{0}: {1}\n'.format(k,len(v)))

    sys.stdout.write('\n')

    # Connect to database
    try:
        conn = psycopg2.connect("dbname='{0}' user='{1}' host='localhost' password='{2}'".format(dbName,dbUser,dbPass))
        sys.stdout.write('Connection to database established\n')
    except Exception as e:
        sys.stderr.write('No connection made to db: {0}\n'.format(str(e)))
        sys.exit(1)

    # First, reset all cluster IDs and scores
    with conn:
        # Cursor to create query
        cur = conn.cursor()

        cur.execute('UPDATE functions SET cid = -2')
        cur.execute('UPDATE functions SET score = -1')

        # Commit transaction
        conn.commit()

    # Insert function highlights
    with conn:
        # Cursor to create queries
        cur = conn.cursor()

        sys.stdout.write('\n')
        sys.stdout.write('Cluster contents: filename address id probability\n')
        for k,v in sorted(c.items(), key=lambda x:len(x[1]), reverse=True):
            for fn,addr,p,score in v:
                cid = str(k)
                sample_hash = fn.split('/')[-1][:-4]
                family = fn.split('/')[-2]

                sys.stdout.write('{0} {1} {2} {3}\n'.format(fn,addr,k,p))

                # If entry already exists, just update cluster ID and score
                cur.execute("INSERT INTO functions (hash,family,func_addr,cid,score) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (hash,family,func_addr) DO UPDATE SET cid=%s, score=%s", (sample_hash, family, addr, cid, score, cid, score))

        # Commit transactions
        conn.commit()

    # Close connect
    conn.close()

if __name__ == '__main__':
    _main()
