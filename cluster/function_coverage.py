#!usr/bin/python3

import sys
import os
import argparse
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def print_stat(title,array):
    sys.stdout.write('{0} min/max/average: {1} {2} {3}\n'.format(title,min(array),max(array),sum(array)/len(array)))

# Calculates averages
def avg(func_addr,func_size):
    bbs = 0

    for addr in func_addr:
        bbs += func_size[addr]['bb']

    avg_bbs = bbs / len(func_addr)

    return avg_bbs

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--functions', help='functions folder', required=True)
    parser.add_argument('--fn', help='fn', required=True)
    parser.add_argument('--addr', help='addr', required=True)
    parser.add_argument('--output', help='output graph file', required=True)

    args = parser.parse_args()

    # Store arguments
    functions = args.functions
    fnFN = args.fn
    addrFN = args.addr
    outputFN = args.output

    highlight = dict()

    # Read in highlighted function addresses
    train_fn = np.load(fnFN)
    train_addr = np.load(addrFN)

    sys.stdout.write('Functions highlighted: {0}\n'.format(len(train_addr)))

    # Store autoencoder bb highlights
    for i in range(len(train_fn)):
        fn = train_fn[i]
        addr = int(train_addr[i],16)

        fn = '/'.join(fn.split('/')[-2:])
        fn = fn[:-4]

        if fn not in highlight:
            highlight[fn] = set()
        highlight[fn].add(addr)

    # Values for graphing (for each malware file)
    percent_coverage = list()
    total_roi_func = list()
    total_func = list()
    avg_bb_roi = list()
    avg_bb_func = list()

    count = 0

    # For each malware sample determine: # of highlighted functions, # of total functions, sizes of each function (bytes and # of basic blocks)
    for k,v in highlight.items():
        check = 0

        roi_func = set()
        func_size = dict()

        # Open functions file and read all basic blocks belonging to functions
        funcFN = os.path.join(functions,k+'.txt')
        with open(funcFN,'r') as fr:
            for line in fr:
                line = line.strip('\n')
                line = line.split(' ')

                # If any error parsing the line exists, continue
                try:
                    f_addr = int(line[0])
                    bb_addr = int(line[1])
                    symbol = int(line[3])
                except Exception as e:
                    continue

                # If not an internal function, continue
                if symbol != 0:
                    continue

                # If this function was highlighted
                if bb_addr in v:
                    check += 1

                    # Add function to set
                    roi_func.add(f_addr)

                # Store function size (sizes of each basic block)
                if f_addr not in func_size:
                    func_size[f_addr] = dict()
                    func_size[f_addr]['bb'] = 0

                func_size[f_addr]['bb'] += 1

        # Check that data matches
        if check != len(v):
            sys.stderr.write('Error, basic blocks highlighted and basic blocks found do not match\n')

        sys.stdout.write('==========================================\n')

        sys.stdout.write('{0}\n'.format(funcFN))

        # Append values to lists for graphing
        percent_coverage.append(len(roi_func) / len(func_size.keys()))
        total_roi_func.append(len(roi_func))
        total_func.append(len(func_size.keys()))

        # Record stats
        sys.stdout.write('# of highlighted functions: {0}\n'.format(len(roi_func)))
        avg_bbs = avg(roi_func,func_size)
        sys.stdout.write('Avg. # of bb\'s: {0}\n'.format(avg_bbs))

        avg_bb_roi.append(avg_bbs)

        sys.stdout.write('# of functions: {0}\n'.format(len(func_size.keys())))
        avg_bbs = avg(func_size.keys(),func_size)
        sys.stdout.write('Avg. # of bb\'s: {0}\n'.format(avg_bbs))

        avg_bb_func.append(avg_bbs)

    # Print min/max/avg
    sys.stdout.write('==========================================\n')
    print_stat('Percent Coverage', percent_coverage)
    print_stat('Highlighted Function Counts per sample', total_roi_func)
    print_stat('All Function Counts per sample', total_func)
    print_stat('Avg. BBs in Highlighted Functions', avg_bb_roi)
    print_stat('Avg. BBs in All Functions', avg_bb_func)

    # Colors from LibreOffice
    color = ['004586','ff420e','ffd320','579d1c','7e0021','83caff','314004','aecf00']

    # Coverage Histogram
    fig= plt.figure(figsize=(5,3))
    percent_coverage = [x*100 for x in percent_coverage]
    n, bins, patches = plt.hist(percent_coverage, 50, facecolor='#{0}'.format(color[0]), rwidth=0.7)

    plt.xlabel('% of Functions Highlighted in Sample')
    plt.ylabel('# of Malware Samples')
    plt.title('Function Highlight Percent Histogram')
    plt.tight_layout()

    plt.savefig(outputFN)

if __name__ == '__main__':
    _main()
