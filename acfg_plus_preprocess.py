import sys
import os
import argparse
import time
import re
from multiprocessing import Pool

import networkx as nx
import pygraphviz
from networkx.drawing import nx_agraph

import binaryninja as binja

class ACFG:
    def __init__(self, addr, num_inst):
        self.entry_addr = addr

        self.inst = list()

        self.offspring = 0
        self.betweenness = 0

        self.api = list()

    # Count instruction mnemonics
    def count(self, mnemonic):
        self.inst.append(mnemonic)

    # NOTE: Might contain FPs. E.g., "call [ebx + 0x4]"
    # Extract call symbol if it exists
    def get_symbol(self, token):
        token = [str(t) for t in token]

        if '[' not in token:
            return

        start = token.index('[')
        end = token.index(']')
        symbol = ' '.join(token[start+1:end])

        self.api.append(symbol)

    def set_offspring(self, offspring):
        self.offspring = int(offspring)

    def set_betweenness(self, betweenness):
        self.betweenness = betweenness

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
        return '<ACFG for {0}>'.format(hex(self.entry_addr))

# Dump ACFG to file
def dump(output,rv):
    with open(output,'w') as fw:
        for acfg in rv:
            fw.write('{0}\n'.format(acfg))

# Extract ACFG from BNDB file using Binary Ninja
def get_acfg_binja(bFN, debug):
    rv = list()

    start = time.time()

    # Import database file
    bv = binja.BinaryViewType.get_view_of_file(bFN)

    sys.stdout.write('{0} Importing BNDB file took {1} seconds\n'.format(bFN,time.time()-start))
    time_previous = time.time()

    # Construct control flow networkx graph to determine betweenness
    G = nx.DiGraph()
    for func in bv.functions:
        bbs = func.basic_blocks
        for bb in bbs:
            # Add node
            if bb.start not in G.nodes():
                G.add_node(bb.start)

            # Add edges
            for edge in bb.incoming_edges:
                # If source node doesn't exist, create it
                if edge.source.start not in G.nodes():
                    G.add_node(edge.source.start)
                G.add_edge(edge.source.start,bb.start)

            for edge in bb.outgoing_edges:
                # If target node doesn't exist, create it
                if edge.target.start not in G.nodes():
                    G.add_node(edge.target.start)
                G.add_edge(bb.start,edge.target.start)

    sys.stdout.write('{0} Constructing CFG took {1} seconds\n'.format(bFN,time.time()-time_previous))
    time_previous = time.time()

    # Get betweenness of CFG
    betweenness = nx.betweenness_centrality(G)

    sys.stdout.write('{0} Calculating betweenness took {1} seconds\n'.format(bFN,time.time()-time_previous))
    time_previous = time.time()

    # Iterate over each function
    for func in bv.functions:
        if debug:
            sys.stdout.write('{0}\n'.format(func.name))

        bbs = func.basic_blocks

        # Iterate over each basic block
        for bb in bbs:
            insts = bb.get_disassembly_text()

            # Create ACFG object
            acfg = ACFG(bb.start, bb.instruction_count)

            # Set betweenness of this basic block
            if bb.start in betweenness:
                acfg.set_betweenness(betweenness[bb.start])
            else:
                acfg.set_betweenness(0)

            # Set number of offspring
            acfg.set_offspring(len(bb.outgoing_edges))

            # For each instruction of basic block
            for e,inst in enumerate(insts):
                # First text of disassembly is function name and address
                if e == 0:
                    continue

                if debug:
                    sys.stdout.write('{0} | {1} \n'.format(hex(inst.address),inst))

                # Get mnemonic
                acfg.count('{0}'.format(inst.tokens[0]))

                # Get call symbol (if it exists)
                acfg.get_symbol(inst.tokens)

            # Append ACFG to return
            rv.append(acfg)

            if debug:
                sys.stdout.write('\n')
        if debug:
            sys.stdout.write('\n')

    sys.stdout.write('{0} Getting ACFG features took {1} seconds\n'.format(bFN,time.time()-time_previous))

    sys.stdout.write('{0} Extracting ACFG from BNDB file took a total of {1} seconds\n'.format(bFN,time.time()-start))

    # Close file
    bv.file.close()

    return rv

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', help='output debug information', required=False, default=False)

    subparsers = parser.add_subparsers(help='disassembler types help', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('binja', help='use BinaryNinja to get CFG')
    sp.set_defaults(cmd='binja')
    sp.add_argument('--bndb', help='BNDB file', required=True)
    sp.add_argument('--output', help='output file', required=True)

    args = parser.parse_args()

    # Store arguments
    tool = args.cmd
    debug = bool(args.debug)
    output = args.output

    if tool == 'binja':
        fn = args.bndb
    else:
        sys.stderr.write('Invalid disassembler: {0}\n'.format(tool))
        sys.exit(1)

    # If bndb file doesn't exist
    if not os.path.exists(fn):
        sys.stderr.write('{0} does not exist\n'.format(fn))
        sys.exit(1)

    # If ACFG plus file already exists
    if os.path.exists(output):
        sys.stderr.write('{0} already exists\n'.format(output))
        sys.exit(1)

    # Extract acfg
    if tool == 'binja':
        rv = get_acfg_binja(fn,debug)
        time_previous = time.time()
        dump(output,rv)
        sys.stdout.write('{0} Finished dumping features took {1} seconds\n'.format(fn,time.time()-time_previous))

if __name__ == '__main__':
    _main()
