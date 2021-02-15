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
    sys.stderr.write('usage: python feature_check.py train.txt test.txt valid.txt\n')
    sys.exit(1)

def _main():
    if len(sys.argv) != 4:
        usage()

    trainFN = sys.argv[1]
    testFN = sys.argv[2]
    validFN = sys.argv[3]

    summation = summary(trainFN)
    sys.stdout.write('Train: {0}\n'.format(summation))
    sys.stdout.write('\n')

    summation = summary(testFN)
    sys.stdout.write('Test: {0}\n'.format(summation))
    sys.stdout.write('\n')

    summation = summary(validFN)
    sys.stdout.write('Valid: {0}\n'.format(summation))
    sys.stdout.write('\n')

if __name__ == '__main__':
    _main()
