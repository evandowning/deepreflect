#!/usr/bin/python3

import sys
import os
import random
import math
from prettytable import PrettyTable

def usage():
    sys.stdout.write('usage: python split.py features/ train.txt test.txt\n')
    sys.exit(2)

def _main():
    if len(sys.argv) != 4:
        usage()

    path = sys.argv[1]
    trainFN = sys.argv[2]
    testFN = sys.argv[3]

    sample = dict()
    for root, dirs, files in os.walk(path,followlinks=True):
        for fn in files:
            label = root.split('/')[-1]

            # If this is not the label, ignore it
            if len(root) <= len(path):
                continue

            if label not in sample:
                sample[label] = list()
            sample[label].append(os.path.join(root,fn))

    # Output train/test sets
    train = list()
    test = list()

    colnames = ['Label','Total','Train','Test']
    table = PrettyTable(colnames)

    numSample = 0
    numFamily = 0

    # Create splits for each family
    for l in sorted(sample, key=lambda x : len(sample[x]), reverse=True):
        # If sample set is large enough
        if len(sample[l]) >= 10:
            # 80% training
            index_train = math.ceil(len(sample[l])*0.8)
            # 20% testing
            index_test = len(sample[l])
        # If it's too small
        else:
            continue

        # Randomize list
        random.shuffle(sample[l])

        table.add_row([l,len(sample[l]),len(sample[l][:index_train]),len(sample[l][index_train:index_test])])

        numSample += len(sample[l])
        numFamily += 1

        # Store splits
        train.extend(sample[l][:index_train])
        test.extend(sample[l][index_train:index_test])

    # Print table
    sys.stdout.write('{0}\n'.format(table))

    # Write out splits
    with open(trainFN,'w') as fw:
        for s in train:
            fw.write('{0}\n'.format(s))
    with open(testFN,'w') as fw:
        for s in test:
            fw.write('{0}\n'.format(s))

    # Print extra stats
    sys.stdout.write('Number of samples: {0}\n'.format(numSample))
    sys.stdout.write('Number of families: {0}\n'.format(numFamily))

if __name__ == '__main__':
    _main()
