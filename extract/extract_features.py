#!/usr/bin/python3

import sys
import os
import argparse
import re
import numpy as np
import time

self_type = dict()
def set_type():
    global self_type

    # Get labels
    for root,dirs,files in os.walk('type'):
        for f in files:
            k = root.split('/')[-1]
            if k not in self_type:
                self_type[k] = dict()

            fn = os.path.join(root,f)
            label = f

            with open(fn,'r') as fr:
                for line in fr:
                    line = line.strip('\n')
                    self_type[k][line] = label

                    if fn.split('/')[-2] == 'api':
                        self_type[k][line+'A'] = label
                        self_type[k][line+'W'] = label

# Create DR feature vector
class DR:
    def __init__(self, addr, inst, offspring, betweenness, api):
        self.entry_addr = int(addr,16)

        self.inst = inst.split(';')

        self.offspring = int(offspring)
        self.betweenness = float(betweenness)

        self.api = api.split(';')

    # Gets type labels for api calls and instructions
    # API labels from : https://github.com/evandowning/monitor/tree/extending-api/sigs/modeling
    # Instruction labels from: https://www.felixcloutier.com/x86/
    def get_type(self):
        global self_type

        # Initialize variables
        self.arith_basic_math = 0
        self.arith_logic_ops = 0
        self.arith_bit_shift = 0

        self.trans_stack = 0
        self.trans_reg = 0
        self.trans_port = 0

        self.api_dll = 0
        self.api_file = 0
        self.api_network = 0
        self.api_object = 0
        self.api_process = 0
        self.api_registry = 0
        self.api_service = 0
        self.api_sync = 0
        self.api_sysinfo = 0
        self.api_time = 0

        # Count parsed labels
        for i in self.inst:
            if i in self_type['inst'].keys():
                if self_type['inst'][i] == 'arith-basic-math':
                    self.arith_basic_math += 1
                elif self_type['inst'][i] == 'arith-logic-ops':
                    self.arith_logic_ops += 1
                elif self_type['inst'][i] == 'arith-bit-shift':
                    self.arith_bit_shift += 1

                elif self_type['inst'][i] == 'trans-stack':
                    self.trans_stack += 1
                elif self_type['inst'][i] == 'trans-register':
                    self.trans_reg += 1
                elif self_type['inst'][i] == 'trans-port':
                    self.trans_port += 1

                else:
                    sys.stderr.write('Error. Unknown instruction type: {0}: {1}\n'.format(i,self_type['inst'][i]))

        for i in self.api:
            # Because when BinaryNinja resolves an api call symbol, it adds '@IAT'
            i = i[:-len('@IAT')]

            if (i != '') and (i in self_type['api'].keys()):
                if self_type['api'][i] == 'dll':
                    self.api_dll += 1
                elif self_type['api'][i] == 'file':
                    self.api_file += 1
                elif self_type['api'][i] == 'network':
                    self.api_network += 1
                elif self_type['api'][i] == 'object':
                    self.api_object += 1
                elif self_type['api'][i] == 'process':
                    self.api_process += 1
                elif self_type['api'][i] == 'registry':
                    self.api_registry += 1
                elif self_type['api'][i] == 'service':
                    self.api_service += 1
                elif self_type['api'][i] == 'sync':
                    self.api_sync += 1
                elif self_type['api'][i] == 'system-info':
                    self.api_sysinfo += 1
                elif self_type['api'][i] == 'time':
                    self.api_time += 1

                else:
                    sys.stderr.write('Error. Unknown api type: {0}: {1}\n'.format(i,self_type['api'][i]))

    def __str__(self):
        rv = 'Basic Block Addr: {0}\n'.format(hex(self.entry_addr))
        rv += '    ++++++ Instruction Features ++++++\n'
        rv += '    Insts: {0}\n'.format(';'.join(self.inst))
        rv += '\n'
        rv += '    ++++++ Structural Features ++++++\n'
        rv += '    Num offspring: {0}\n'.format(self.offspring)
        rv += '    Betweenness: {0}\n'.format(self.betweenness)
        rv += '\n'
        rv += '    ++++++ API Features ++++++\n'
        rv += '    APIs: {0}\n'.format(';'.join(self.api))
        return rv

    def __repr__(self):
        return '<DR for {0}>'.format(hex(self.entry_addr))

# Dump features to output file
def dump(output,dr):
    # Get folder of features file
    # Create it if it doesn't exist
    root = os.path.dirname(output)
    if not os.path.exists(root):
        os.makedirs(root)

    # Create numpy array
    array = np.array([], dtype=float)

    # For each basic block
    for bb in sorted(dr, key=lambda x:x.entry_addr):
        a = np.array([], dtype=float)

        # [0] Entry address
        a = np.append(a,bb.entry_addr)

        # [1] Offspring
        a = np.append(a,bb.offspring)
        # [2] Betweenness
        a = np.append(a,bb.betweenness)

        # [3] Arithmetic - basic math (functionality)
        a = np.append(a,bb.arith_basic_math)
        # [4] Arithmetic - logical operations (programmatic / control flow)
        a = np.append(a,bb.arith_logic_ops)
        # [5] Arithmetic - bit shifting (efficency)
        a = np.append(a,bb.arith_bit_shift)

        # [6] Transfer - stack operations
        a = np.append(a,bb.trans_stack)
        # [7] Transfer - register operations
        a = np.append(a,bb.trans_reg)
        # [8] Transfer - port operations
        a = np.append(a,bb.trans_port)

        # [9] API - dll
        a = np.append(a,bb.api_dll)
        # [10] API - file
        a = np.append(a,bb.api_file)
        # [11] API - network
        a = np.append(a,bb.api_network)
        # [12] API - object
        a = np.append(a,bb.api_object)
        # [13] API - process
        a = np.append(a,bb.api_process)
        # [14] API - registry
        a = np.append(a,bb.api_registry)
        # [15] API - service
        a = np.append(a,bb.api_service)
        # [16] API - sync
        a = np.append(a,bb.api_sync)
        # [17] API - system info
        a = np.append(a,bb.api_sysinfo)
        # [18] API - time
        a = np.append(a,bb.api_time)

        if len(array) == 0:
            array = np.array([a])
        else:
            array = np.vstack((array,a))

    # Output numpy array
    np.save(output, array)

# Gets DR contents from preprocessed file
def extract(fn):
    rv = list()

    # Read all contents from file
    with open(fn,'r') as fr:
        content = fr.read()

    # Create pattern
    pattern = r''
    pattern += r'Basic Block Addr: (.*)\n'
    pattern += r'.*\n'
    pattern += r'.*Insts: (.*)\n'
    pattern += r'.*\n'
    pattern += r'.*\n'
    pattern += r'.*Num offspring: (.*)\n'
    pattern += r'.*Betweenness: (.*)\n'
    pattern += r'.*\n'
    pattern += r'.*\n'
    pattern += r'.*APIs: (.*)\n'

    # Parse DR features
    match = re.findall(pattern, content, re.MULTILINE)
    for m in match:
        addr,inst,offspring,betweenness,api = m
        dr = DR(addr,inst,offspring,betweenness,api)
        dr.get_type()
        rv.append(dr)

    return rv

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw', help='raw features file', required=True)
    parser.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    fn = args.raw
    output = args.output

    # If raw file doesn't exist
    if not os.path.exists(fn):
        sys.stderr.write('{0} does not exist\n'.format(fn))
        sys.exit(1)

    # If features file already exists
    if os.path.exists(output):
        sys.stderr.write('{0} already exists\n'.format(output))
        sys.exit(1)

    start = time.time()

    # Set types from files
    set_type()

    sys.stdout.write('{0} : Setting types took {1} seconds\n'.format(fn,time.time()-start))
    start = time.time()

    # Get dr contents
    rv = extract(fn)

    sys.stdout.write('{0} : Extracting features took {1} seconds\n'.format(fn,time.time()-start))
    start = time.time()

    # Output feature vector
    dump(output,rv)

    sys.stdout.write('{0} : Dumping features took {1} seconds\n'.format(fn,time.time()-start))

if __name__ == '__main__':
    _main()
