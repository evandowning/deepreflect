import sys
import os
import argparse
import re

class ACFG:
    def __init__(self, addr, num_inst, trans_inst, arith_inst, call_inst, string, numeric, offspring, betweenness):
        self.entry_addr = int(addr,16)

        self.num_inst = int(num_inst)
        self.trans_inst = int(trans_inst)
        self.arith_inst = int(arith_inst)
        self.call_inst = int(call_inst)

        self.string = string
        self.numeric = numeric

        self.offspring = int(offspring)
        self.betweenness = float(betweenness)

    def __str__(self):
        rv = 'Basic Block Addr: {0}\n'.format(hex(self.entry_addr))
        rv += '\n'
        rv += '    ++++++ Statistical Features ++++++\n'
        rv += '    String constants: {0}\n'.format(self.string)
        rv += '    Numeric constants: {0}\n'.format(self.numeric)
        rv += '    Num transfer insts: {0}\n'.format(self.trans_inst)
        rv += '    Num call insts: {0}\n'.format(self.call_inst)
        rv += '    Num insts: {0}\n'.format(self.num_inst)
        rv += '    Num arithmetic insts: {0}\n'.format(self.arith_inst)
        rv += '\n'
        rv += '    ++++++ Structural Features ++++++\n'
        rv += '    Num offspring: {0}\n'.format(self.offspring)
        rv += '    Betweenness: {0}\n'.format(self.betweenness)
        return rv

    def __repr__(self):
        return '<ACFG for {0}>'.format(hex(self.entry_addr))

# Dump features to output file
def dump(output,fn,acfg):
    h = fn.split('/')[-1]
    f = fn.split('/')[-2]

    newdir = os.path.join(output,f)

    # Create new directory
    if not os.path.exists(newdir):
        os.makedirs(newdir)

    outputFN = os.path.join(newdir,h)

    with open(outputFN,'w') as fw:
        # Sort ACFG basic blocks by address
        for bb in sorted(acfg, key=lambda x:x.entry_addr):
            vector = [
                        bb.entry_addr,
                        bb.num_inst,
                        bb.trans_inst,
                        bb.arith_inst,
                        bb.call_inst,
                        bb.offspring,
                        bb.betweenness
                     ]

            fw.write('{0}\n'.format(str(vector)))

# Gets ACFG contents from preprocessed file
def extract(fn):
    rv = list()

    # Read all contents from file
    with open(fn,'r') as fr:
        content = fr.read()

    # Create pattern
    pattern = r''
    pattern += r'Basic Block Addr: (.*)\n'
    pattern += r'.*\n'
    pattern += r'.*\n'
    pattern += r'.*String constants: (.*)\n'
    pattern += r'.*Numeric constants: (.*)\n'
    pattern += r'.*Num transfer insts: (.*)\n'
    pattern += r'.*Num call insts: (.*)\n'
    pattern += r'.*Num insts: (.*)\n'
    pattern += r'.*Num arithmetic insts: (.*)\n'
    pattern += r'.*\n'
    pattern += r'.*\n'
    pattern += r'.*Num offspring: (.*)\n'
    pattern += r'.*Betweenness: (.*)\n'

    # Parse ACFGs
    match = re.findall(pattern, content, re.MULTILINE)
    for m in match:
        addr,string,numeric,trans_inst,call_inst,num_inst,arith_inst,offspring,betweenness = m
        acfg = ACFG(addr,num_inst,trans_inst,arith_inst,call_inst,string,numeric,offspring,betweenness)
        rv.append(acfg)

    return rv

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', help='output debug information', required=False, default=False)
    parser.add_argument('--acfg', help='acfg preprocess folder', required=True)
    parser.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    debug = bool(args.debug)
    folder = args.acfg
    output = args.output

    # Get samples in folder
    sample = list()
    for root,dirs,files in os.walk(folder):
        for fn in files:
            sample.append(os.path.join(root,fn))

    # For each sample
    for e,fn in enumerate(sample):
        sys.stdout.write('Processing {0} / {1}\r'.format(e+1,len(sample)))
        sys.stdout.flush()

        # Get acfg contents
        rv = extract(fn)

        # Output feature vector
        dump(output,fn,rv)

    sys.stdout.write('\n')

if __name__ == '__main__':
    _main()
