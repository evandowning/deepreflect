#!/usr/bin/python3

import sys
import os
import subprocess
import argparse
import binaryninja

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='action', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('extract', help='extract features and information from dataset')
    sp.set_defaults(cmd='extract')
    sp.add_argument('--binaries', help='directory of binaries', required=True)

    sp = subparsers.add_parser('train', help='train autencoder')
    sp.set_defaults(cmd='train')
    sp.add_argument('--features', help='directory of benign feature files', required=True)
    sp.add_argument('--output', help='output directory', required=True)

    #TODO
    sp = subparsers.add_parser('roi', help='extracts RoIs')
    sp.set_defaults(cmd='roi')
    sp.add_argument('--blah', help='blah', required=True)

    #TODO
    sp = subparsers.add_parser('cluster', help='cluster RoIs')
    sp.set_defaults(cmd='cluster')
    sp.add_argument('--blah', help='blah', required=True)

    args = parser.parse_args()

    # Store arguments
    action = args.cmd

    sys.stdout.write('Command: {0}\n'.format(action))

    if action == 'extract':
        folder = args.binaries
        result = subprocess.run(["bash","extract.sh","{0}".format(folder)], capture_output=True, text=True, check=True)

        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

    elif action == 'train':
        benignFolder = args.features
        outFolder = args.output

        # Get working directory to run scripts
        cwd = os.getcwd()
        root = os.path.join(cwd,'autoencoder')

        # Create output folder (exit if it already exists)
        os.makedirs(outFolder)

        trainFile = os.path.join(outFolder,'train.txt')
        testFile = os.path.join(outFolder,'test.txt')

        # Split & Shuffle dataset
        sys.stdout.write('\n')
        sys.stdout.write('Calling split.py\n')
        result = subprocess.run(["python","split.py",benignFolder,trainFile,testFile], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Check that samples use features
        sys.stdout.write('\n')
        sys.stdout.write('Calling feature_check.py\n')
        result = subprocess.run(["python","feature_check.py",trainFile], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))
        result = subprocess.run(["python","feature_check.py",testFile], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Normalize dataset
        sys.stdout.write('\n')
        sys.stdout.write('Calling normalize.py\n')
        normalizeFN = os.path.join(outFolder,'normalize.npy')
        result = subprocess.run(["python","normalize.py","--train",trainFile,"--test",testFile,"--output",normalizeFN], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Train model
        sys.stdout.write('\n')
        sys.stdout.write('Calling autoencoder.py\n')
        modelFN = os.path.join(outFolder,'dr.h5')
        result = subprocess.run(["python","autoencoder.py","--train",trainFile,"--test",testFile,"--normalize",normalizeFN,"--model",modelFN], cwd=root, capture_output=True, text=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

    #TODO
    elif action == 'roi':
        pass

    #TODO
    elif action == 'cluster':
        pass

if __name__ == '__main__':
    _main()
