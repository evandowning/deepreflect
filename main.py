#!/usr/bin/python3

import sys
import os
import argparse
import binaryninja

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='action', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('extract', help='extract features and information from dataset')
    sp.set_defaults(cmd='extract')
    sp.add_argument('--blah', help='blah', required=True)

    sp = subparsers.add_parser('train', help='train autencoder')
    sp.set_defaults(cmd='train')
    sp.add_argument('--blah', help='blah', required=True)

    sp = subparsers.add_parser('roi', help='extracts RoIs')
    sp.set_defaults(cmd='roi')
    sp.add_argument('--blah', help='blah', required=True)

    sp = subparsers.add_parser('cluster', help='cluster RoIs')
    sp.set_defaults(cmd='cluster')
    sp.add_argument('--blah', help='blah', required=True)

    args = parser.parse_args()

    # Store arguments
    action = args.cmd

    sys.stdout.write('Command: {0}\n'.format(cmd))

if __name__ == '__main__':
    _main()
