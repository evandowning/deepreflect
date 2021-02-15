#!/usr/bin/python3

import sys
import os
import argparse
import time

import numpy as np

from keras.models import load_model

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='dataset types help', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('acfg_plus', help='basic block features of malware dataset')
    sp.set_defaults(cmd='acfg_plus')
    sp.add_argument('--acfg-feature', help='ACFG feature folder', required=True)
    sp.add_argument('--model', help='model path', required=True)
    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)
    sp.add_argument('--normalize', help='normalize features', required=False, default=False)
    sp.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    dataset = args.cmd
    folder = args.acfg_feature
    model_path = args.model
    output = args.output

    if dataset == 'acfg_plus':
        max_len = int(args.max_bb)

    # Get samples
    samples = list()
    for root,dirs,files in os.walk(folder):
        for fn in files:
            samples.append(os.path.join(root,fn))

    start = time.time()

    # Load trained model
    model = load_model(model_path)

    sys.stdout.write('Took {0} seconds to load model.\n'.format(time.time() - start))

    count = 0
    count_save = 0
    total = 0

    # For each sample
    for e,fn in enumerate(samples):
        sys.stdout.write('Processing {0}/{1}\r'.format(e+1,len(samples)))
        sys.stdout.flush()

        if dataset == 'acfg_plus':
            # Only look at first max_len of data (and pad with empty feature vector)
            b = np.array([[0]*18]*max_len, dtype=np.float)
            bytez = np.load(fn)

            # If nothing was loaded, ignore this sample
            if len(bytez) == 0:
                sys.stderr.write('Error. Sample {0} has no data.\n'.format(fn))
                continue

            bytez = bytez[:max_len]
            # First element is the entry point, so we should ignore this
            bytez = bytez[:,1:]
            b[:len(bytez)] = bytez

            if args.normalize:
                # Autoencoder (benign unpacked plus, filtered)
                maximum_val = np.array([2.18000000e+02,4.69411765e-01,1.59328500e+06,1.00820000e+04,7.68000000e+02,5.19373000e+05,1.20040000e+04,4.36000000e+02,4.10000000e+01,6.00000000e+00,7.00000000e+00,3.00000000e+00,6.00000000e+00,1.50000000e+01,2.00000000e+00,3.00000000e+00,5.00000000e+00,5.00000000e+00])

                b /= maximum_val

        total += 1

        start = time.time()

        # Get output
        out = model.predict(x=np.array([b]))

        count += (time.time() - start)

        start = time.time()

        # Output mean-squared-error (input vs. output) for each basic block (row)
        mse = (np.square(b - out[0])).mean(axis=1)

        fname = fn.split('/')[-1]
        family = fn.split('/')[-2]
        try:
            os.makedirs(os.path.join(output,family))
        except FileExistsError:
            pass
        outputFN = os.path.join(output,family,fname)

        np.save(outputFN,mse)

        count_save += (time.time() - start)

    sys.stdout.write('\n')
    sys.stdout.write('On average took {0} seconds to get autoencoder output\n'.format(count/total))
    sys.stdout.write('On average took {0} seconds to save outputs\n'.format(count_save/total))

if __name__ == '__main__':
    _main()
