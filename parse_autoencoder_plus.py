# Extract BB addresses based on autoencoder "highlights"

# python parse.py --data m2_4shared_30cd298feff2e2ad25a73ef0365b1f69d152d17011b6679e53071665f9e032ba_BB_sq_err.npy --thresh 750 --acfg /data/arsa/final_binaries_unipacker_bndb_acfg_feature_hellsing/4shared/30cd298feff2e2ad25a73ef0365b1f69d152d17011b6679e53071665f9e032ba

import sys
import os
import numpy as np
import argparse

def parse(dataFN,acfgFN,thresh):
    rv_addr = list()
    rv_mse = list()

    # Read data
    data = np.load(dataFN)

    addr = list()

    # Read ACFG feature addresses
    acfg = np.load(acfgFN)
    for a in acfg:
        addr.append(int(a[0]))

    # Extend addr if necessary (address of -1 denotes padding)
    if len(addr) < len(data):
        diff = len(data) - len(addr)
        addr.extend(['-1']*diff)

    # Identify highlighted basic blocks
    index = np.where(data >= thresh)[0]

    for i in index:
        a = int(addr[i])
        m = float(data[i])
        rv_addr.append(a)
        rv_mse.append(m)

    return rv_addr,rv_mse

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='numpy data file', required=True)
    parser.add_argument('--thresh', type=float, help='threshold value', required=True)
    parser.add_argument('--acfg', type=str, help='ACFG feature file', required=True)

    args = parser.parse_args()

    # Store arguments
    dataFN = args.data
    thresh = float(args.thresh)
    acfgFN = args.acfg

    rv_addr,rv_mse = parse(dataFN,acfgFN,thresh)

    for i in range(len(rv_addr)):
        a = rv_addr[i]
        m = rv_mse[i]
        sys.stdout.write('{0}, {1}\n'.format(hex(a),m))

if __name__ == '__main__':
    _main()
