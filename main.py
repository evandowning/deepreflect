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

    sp = subparsers.add_parser('roi', help='extracts RoIs')
    sp.set_defaults(cmd='roi')
    sp.add_argument('--features', help='directory of feature files', required=True)
    sp.add_argument('--bndb-func', help='directory of function files', required=True)
    sp.add_argument('--model', help='directory of model files', required=True)
    sp.add_argument('--thresh', type=float, help='mse threshold value', required=True)
    sp.add_argument('--out-mse', help='directory of mse files', required=True)
    sp.add_argument('--out-roi', help='directory of roi files', required=True)

    sp = subparsers.add_parser('cluster', help='cluster RoIs')
    sp.set_defaults(cmd='cluster')
    sp.add_argument('--bndb-func', help='directory of function files', required=True)
    sp.add_argument('--roi', help='directory of roi files', required=True)
    sp.add_argument('--output', help='directory of cluster files', required=True)

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
        result = subprocess.run(["python","autoencoder.py","--train",trainFile,"--test",testFile,"--normalize",normalizeFN,"--model",modelFN], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

    elif action == 'roi':
        featFolder = args.features
        funcFolder = args.bndb_func
        modelFolder = args.model
        threshold = float(args.thresh)
        outMSE = args.out_mse
        outRoI = args.out_roi

        normalizeFN = os.path.join(modelFolder,'normalize.npy')
        modelFN = os.path.join(modelFolder,'dr.h5')

        featureFN = os.path.join('/tmp','files.txt')

        # Create output folder (exit if it already exists)
        os.makedirs(outMSE)
        os.makedirs(outRoI)

        # Find malicious feature files
        result = subprocess.run(["find",featFolder,"-type","f"], capture_output=True, text=True)
        with open(featureFN,'w') as fw:
            fw.write('{0}'.format(result.stdout))

        # Get working directory to run scripts
        cwd = os.getcwd()
        root = os.path.join(cwd,'autoencoder')

        # Extract MSE values
        sys.stdout.write('\n')
        sys.stdout.write('Calling mse.py\n')
        result = subprocess.run(["python","mse.py","--feature",featureFN,"--model",modelFN,"--normalize",normalizeFN,"--output",outMSE], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Extract RoIs
        sys.stdout.write('\n')
        sys.stdout.write('Calling roi.py\n')
        result = subprocess.run(["python","roi.py","--bndb-func",funcFolder,"--feature",featFolder,"--mse",outMSE,"--normalize",normalizeFN,"--output",outRoI,"--bb","--avg","--thresh",str(threshold)], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        roiFN = os.path.join(outRoI,'fn.npy')
        roiAddr = os.path.join(outRoI,'addr.npy')
        roiMSE = os.path.join(outRoI,'mse_func.npy')

        # Extract MSE values for each highlighted function
        sys.stdout.write('\n')
        sys.stdout.write('Calling mse_func.py\n')
        result = subprocess.run(["python","mse_func.py","--bndb-func",funcFolder,"--feature",featFolder,"--roiFN",roiFN,"--roiAddr",roiAddr,"--thresh",str(threshold),"--output",roiMSE], cwd=root, capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

    elif action == 'cluster':
        funcFolder = args.bndb_func
        roiFolder = args.roi
        outFolder = args.output

        roiX = os.path.join(roiFolder,'x.npy')
        roiFN = os.path.join(roiFolder,'fn.npy')
        roiAddr = os.path.join(roiFolder,'addr.npy')
        roiMSE = os.path.join(roiFolder,'mse_func.npy')

        # Create output folder (exit if it already exists)
        os.makedirs(outFolder)

        # Create database
        sys.stdout.write('\n')
        sys.stdout.write('Creating database\n')
        result = subprocess.run(["bash","db/setup.sh"], capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Get working directory to run scripts
        cwd = os.getcwd()
        root = os.path.join(cwd,'cluster')

        cfgFN = os.path.join(root,'cluster.cfg')

        # Create cluster config file
        with open(cfgFN,'w') as fw:
            fw.write('[data]\n')
            fw.write('x = {0}\n'.format(roiX))
            fw.write('fn = {0}\n'.format(roiFN))
            fw.write('addr = {0}\n'.format(roiAddr))
            fw.write('score = {0}\n'.format(roiMSE))

            fw.write('\n')

            fw.write('[db]\n')
            fw.write('name = dr\n')
            fw.write('username = root\n')
            fw.write('password = pass\n')

        # Cluster
        sys.stdout.write('\n')
        sys.stdout.write('Clustering\n')
        outFN = os.path.join(outFolder,'cluster.txt')
        result = subprocess.run(["python","pca_hdbscan.py","--cfg",cfgFN], cwd=root, capture_output=True, text=True, check=True)
        with open(outFN,'w') as fw:
            fw.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Output function coverage
        sys.stdout.write('\n')
        sys.stdout.write('Outputting function coverage\n')
        outPNG = os.path.join(outFolder,'function_coverage.png')
        outFN = os.path.join(outFolder,'function_coverage_stdout.txt')
        result = subprocess.run(["python","function_coverage.py","--functions",funcFolder,"--fn",roiFN,"--addr",roiAddr,"--output",outPNG], cwd=root, capture_output=True, text=True, check=True)
        with open(outFN,'w') as fw:
            fw.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

        # Output database
        sys.stdout.write('\n')
        sys.stdout.write('Outputting database contents\n')
        outDB = os.path.join(outFolder,'export.sql')
        result = subprocess.run(["pg_dump","-O","dr","-f",outDB], capture_output=True, text=True, check=True)
        sys.stdout.write('stdout:\n')
        sys.stdout.write('{0}'.format(result.stdout))
        sys.stderr.write('stderr:\n')
        sys.stderr.write('{0}'.format(result.stderr))

if __name__ == '__main__':
    _main()
