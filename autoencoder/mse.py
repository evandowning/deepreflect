#!/usr/bin/python3

import sys
import os
import argparse
import time
import numpy as np

import tensorflow as tf

sys.path.append('../')
from dr_feature import DR

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature', help='feature files', required=True)
    parser.add_argument('--model', help='model path', required=True)

    parser.add_argument('--normalize', help='normalize features', required=False, default=None)
    parser.add_argument('--max-bb', help='max number of basic blocks to consider', type=int, required=False, default=20000)

    parser.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    featureFN = args.feature
    model_path = args.model
    normalizeFN = args.normalize
    max_len = int(args.max_bb)
    output = args.output

    # Load dataset
    data = DR(featureFN,None,max_len,normalizeFN)

    start = time.time()

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    sys.stdout.write('Took {0} seconds to load model.\n'.format(time.time() - start))

    # Get number of samples
    num_samples = data.get_train_num()

    count = 0

    # For each sample
    for e,t in enumerate(data.generator('train',1)):
        sys.stdout.write('Processing {0}/{1}\r'.format(e+1,num_samples))
        sys.stdout.flush()

        # Get input
        x,_ = t

        # Get input file path
        fn = data.get_path('train',e)

        start = time.time()

        # Get output
        out = model.predict(x=x)

        # Output mean-squared-error (input vs. output) for each basic block (row)
        mse = (np.square(x - out)).mean(axis=1)

        # Create folder
        fname = fn.split('/')[-1]
        family = fn.split('/')[-2]
        outputFolder = os.path.join(output,family)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        outputFN = os.path.join(outputFolder,fname)

        # Save MSE values into file
        np.save(outputFN,mse)

        count += (time.time() - start)

        if e+1 == num_samples:
            break

    sys.stdout.write('\n')
    sys.stdout.write('On average took {0} seconds to retrieve & save outputs\n'.format(count/num_samples))

if __name__ == '__main__':
    _main()
