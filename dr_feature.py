#!/usr/bin/python3

import sys
import os
import numpy as np
import random

# Holds parsed dataset
class DR(object):
    # Get samples
    def __init__(self, trainFN, testFN, max_len, normalizeFN):
        self.train = list()
        self.test = list()
        self.valid = list()

        self.normalizeFN = normalizeFN

        self.max_len = max_len

        label_set = set()

        # Get samples
        with open(trainFN,'r') as fr:
            for line in fr:
                line = line.strip('\n')

                self.train.append(line)

        with open(testFN,'r') as fr:
            for line in fr:
                line = line.strip('\n')

                self.test.append(line)

        # If normalizing the DR feature vectors
        if self.normalizeFN is not None:
            self.maximum_val = np.load(self.normalizeFN)

    # Some getter functions
    def get_train_num(self):
        return len(self.train)
    def get_test_num(self):
        return len(self.test)

    def get_train(self):
        return self.train
    def get_test(self):
        return self.test

    # Determines max values to normalize
    def get_max(self,outputFN):
        sample = self.train + self.test

        maximum_val = np.array([0.0]*18)

        # Get maximum values for each feature vector
        for e,fn in enumerate(sample):
            sys.stdout.write('Normalizing samples: {0} / {1}\r'.format(e+1,len(sample)))
            sys.stdout.flush()

            # Only look at first max_len of data (and pad with empty feature vector)
            b = np.array([[0]*18]*self.max_len, dtype=float)
            bytez = np.load(fn)

            # If nothing was loaded, ignore this sample
            if len(bytez) == 0:
                sys.stderr.write('Error. Sample {0} has no data.\n'.format(fn))
                continue

            bytez = bytez[:self.max_len]
            # First element is the entry point, so we should ignore this
            bytez = bytez[:,1:]
            b[:len(bytez)] = bytez

            # Get maximum values for each sample's feature vector
            b_max = b.max(axis=0)
            maximum_val_index = np.where(b.max(axis=0) > maximum_val)[0]
            maximum_val[maximum_val_index] = b_max[maximum_val_index]

        sys.stdout.write('\n')

        # Autoencoder (benign unpacked plus, valid, filtered)
       #maximum_val = np.array([2.18000000e+02,4.66131907e-01,1.59328500e+06,1.00820000e+04,7.68000000e+02,5.19373000e+05,1.20040000e+04,4.36000000e+02,4.10000000e+01,6.00000000e+00,7.00000000e+00,3.00000000e+00,6.00000000e+00,1.50000000e+01,2.00000000e+00,3.00000000e+00,5.00000000e+00,5.00000000e+00])

        # Output numpy array
        np.save(outputFN, maximum_val)

    # Data Generator
    def generator(self,t,batch_size):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test

        x = list()

        while True:
            for fn in sample:
                # Only look at first max_len of data (and pad with empty feature vector)
                b = np.array([[0]*18]*self.max_len, dtype=float)
                bytez = np.load(fn)

                # If nothing was loaded, ignore this sample
                if len(bytez) == 0:
                    sys.stderr.write('Error. Sample {0} has no data.\n'.format(fn))
                    continue

                bytez = bytez[:self.max_len]
                # First element is the entry point, so we should ignore this
                bytez = bytez[:,1:]
                b[:len(bytez)] = bytez

                # Organize data to be fed to keras
                x.append(b)

                print(len(x))

                if len(x) == batch_size:
                    if self.normalizeFN is not None:
                        yield (np.asarray(x) / self.maximum_val , np.asarray(x) / self.maximum_val)
                    else:
                        yield (np.asarray(x), np.asarray(x))

                    x = list()
