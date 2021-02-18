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

# Features from "Scalable Graph-based Bug Search for Firmware Images" (CCS 2016)
# paper

# From: https://github.com/yangshouguo/Graph-based_Bug_Search
#   transfer_instructions = ['MOV','PUSH','POP','XCHG','IN','OUT','XLAT','LEA','LDS','LES','LAHF', 'SAHF' ,'PUSHF', 'POPF']
#   arithmetic_instructions = ['ADD', 'SUB', 'MUL', 'DIV', 'XOR', 'INC','DEC', 'IMUL', 'IDIV', 'OR', 'NOT', 'SLL', 'SRL']

# From: https://github.com/qian-feng/Gencoding/blob/7dcb04cd577e62a6394f5f68b751902db552ebd3/raw-feature-extractor/graph_analysis_ida.py
#   arithmetic_instructions = ['add', 'sub', 'div', 'imul', 'idiv', 'mul', 'shl', 'dec', 'inc']
#   transfer_instrucitons = ['jmp', 'jz', 'jnz', 'js', 'je', 'jne', 'jg', 'jle', 'jge', 'ja', 'jnc', 'call']
#   call_instructions = ['call', 'jal', 'jalr']

# Mine
transfer_instructions = ['mov','push','pop','xchg','in','out','xlat','lea','lds','les','lahf', 'sahf' ,'pushf', 'popf']
arithmetic_instructions = ['add', 'sub', 'div', 'imul', 'idiv', 'mul', 'shl', 'dec', 'inc', 'xor', 'or', 'not', 'sll', 'srl']
call_instructions = ['call']

class ACFG:
    def __init__(self, addr, num_inst):
        self.entry_addr = addr

        self.num_inst = int(num_inst)
        self.trans_inst = 0
        self.arith_inst = 0
        self.call_inst = 0

        self.string = ''
        self.numeric = ''

        self.offspring = 0
        self.betweenness = 0

        self.api = list()

    # Count instruction mnemonics
    def count(self, mnemonic):
        if mnemonic in transfer_instructions:
            self.trans_inst += 1

        elif mnemonic in arithmetic_instructions:
            self.arith_inst += 1

        elif mnemonic in call_instructions:
            self.call_inst += 1

    # Extract call symbol if it exists
    def get_symbol(self, token):
        token = [str(t) for t in token]

        if token[0] not in call_instructions:
            return

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

    # Add numeric constant
    def add_numeric(self, i):
        self.numeric += hex(i) + ' '

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

                #TODO
                # Get strings
                # Get numerical constants

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

# Wrapper for multiprocessing
def get_acfg_binja_wrapper(args):
    fn = args[0]

    # Assuming that the input files end in the ".bndb" extension
    ext = '.bndb'
    loc = '/'.join(fn.split('/')[-2:])[:-len(ext)]

    return loc,get_acfg_binja(*args)

# Extract ACFG from PE file using radare2
def get_acfg_radare2(peFN, debug):
    import r2pipe
    rv = list()

    # Open PE file
    r = r2pipe.open(peFN)

    # Analyze all of file
    r.cmd('aaa')

    # Get list of functions
    funcs = r.cmdj('aflj')

    start = time.time()

    betweenness = dict()

    # Construct control flow networkx graph and determine betweenness

    # Iterate over each function and construct global basic block CFG
    # and betweenness scores
    for f in funcs:
        # Get name of function
        fname = f['name']

        # Get entry point address of function
        faddr = f['offset']

        # Get basic block CFG of function
        cfg_dot = r.cmd('agfd {0}'.format(hex(faddr)))

        # From: https://stackoverflow.com/questions/42172548/read-dot-graph-in-networkx-from-a-string-and-not-file
        cfg_networkx = nx_agraph.from_agraph(pygraphviz.AGraph(cfg_dot,strict=False,directed=True), create_using=nx.DiGraph)

        # Get betweenness of CFG
        betweenness.update(nx.betweenness_centrality(cfg_networkx))

    # Update all dictionary keys to be integers
    betweenness = dict((int(key,16), value) for (key, value) in betweenness.items())

    sys.stdout.write('{0} Extracting CFG and betweenness from PE file took {1} seconds\n'.format(peFN,time.time()-start))

    if debug:
        sys.stdout.write('Number of basic blocks with betweenness: {0}\n'.format(len(betweenness)))

    start = time.time()

    # For each function
    for f in funcs:
        # Get name of function
        fname = f['name']
        faddr = f['offset']

        # For debugging
        if debug:
            sys.stdout.write('{0}\n'.format(fname))

        # Get basic blocks of this function
        r.cmd('s {0}'.format(faddr))
        bb = r.cmdj('afbj')

        # Prints entire disassembled function for debugging
        if debug:
            sys.stdout.write('{0}\n'.format(r.cmd('pdf')))
            sys.stdout.write('-+-+-+-+-+\n')

        # For each basic block
        for b in bb:
            # Prints basic block information for debugging
            if debug:
                sys.stdout.write('{0}\n'.format(b))

            # Get entry point of basic block
            entry_addr = b['addr']

            # Disassemble basic block
            r.cmd('s {0}'.format(entry_addr))
            disas = r.cmdj('pdbj')

            # If couldn't disassemble basic block
            if disas is None:
                continue

            # Prints entire disassembled basic block for debugging
            if debug:
                sys.stdout.write('{0}\n'.format(r.cmd('pdf')))

            # Get number of instructions
            num_insts = b['ninstr']

            # Create ACFG object
            acfg = ACFG(entry_addr, num_insts)

            # Get number of offspring
            acfg.set_offspring(b['outputs'])

            # Set betweenness of this basic block
            if b['addr'] in betweenness:
                acfg.set_betweenness(betweenness[b['addr']])
            else:
                acfg.set_betweenness(0)

            # For each instruction
            for inst in disas:
                if debug:
                    sys.stdout.write('{0}\n'.format(inst))

                # Get address of instruction
                inst_addr = inst['offset']
                r.cmd('s {0}'.format(inst_addr))

                # If can't get instruction information
                if len(r.cmdj('aoj')) == 0:
                    continue

                # Get information about instruction
                inst_info = r.cmdj('aoj')[0]

                # Get mnemonic
                acfg.count(inst_info['mnemonic'])

                # Get operands of this instruction
                for o in inst_info['opex']['operands']:
                    # Print operand information for debugging
                    if debug:
                        sys.stdout.write('{0}\n'.format(o))

                    # Get numeric constants
                    if o['type'] == 'imm':
                        acfg.add_numeric(int(o['value']))

                #TODO
                # Get string constants
                # From: https://insinuator.net/2016/08/reverse-engineering-with-radare2-part-2/
                # Or maybe just quotations: https://monosource.gitbooks.io/radare2-explorations/content/intro/basics.html
                # or https://medium.com/@jacob16682/reverse-engineering-with-radare2-part-2-83b71df7ffe4
#               if '"' in r.cmd('pdf'):
#                   print('HERE')
#                   print(inst)
#                   sys.stdout.write('{0}\n'.format(r.cmd('pdf')))
#                   sys.exit()

            if debug:
                sys.stdout.write('{0}\n'.format(acfg))

            rv.append(acfg)

        if debug:
            sys.stdout.write('===========\n')

    sys.stdout.write('{0} Extracting ACFG from PE file took {1} seconds\n'.format(peFN,time.time()-start))

    # Close PE file
    r.quit()

    return rv

# Extract ACFG from PE file using angr
def get_acfg_angr(peFN,debug):
    import pefile
    from disassemble import Disassemble
    from angr.errors import SimEngineError
    from capstone.x86 import X86_OP_IMM

    rv = list()

    # Open PE file
    pe = pefile.PE(peFN)

    start = time.time()
    sys.stdout.write('{0} Disassembling PE file...'.format(start))
    sys.stdout.flush()

    # Disassemble PE file
    d = Disassemble(pe,peFN)

    sys.stdout.write('Done: Took {0} seconds\n'.format(time.time() - start))

    # Print size of CFG
    sys.stdout.write('Size of CFG: {0} basic blocks, {1} edges\n'.format(d.aw.cfg.graph.number_of_nodes(),d.aw.cfg.graph.number_of_edges()))

    start = time.time()
    sys.stdout.write('{0} Calculating betweenness of CFG...'.format(start))
    sys.stdout.flush()

    # Get betweenness of control flow graph
    # Should estimate this value: https://stackoverflow.com/questions/32465503/networkx-never-finishes-calculating-betweenness-centrality-for-2-mil-nodes
    # NOTE: for speed, I chose to limit the search space to 256
    betweenness = nx.betweenness_centrality(d.aw.cfg.graph, k=256)

    sys.stdout.write('Done: Took {0} seconds\n'.format(time.time() - start))

    # From: https://docs.angr.io/built-in-analyses/cfg#function-manager
    funcs = d.aw.cfg.kb.functions

    # For each function
    for e,t in enumerate(funcs.items()):
        k,v = t

        sys.stdout.write('Processing function: {0} / {1}\r'.format(e+1,len(funcs.keys())))
        sys.stdout.flush()

        # From: https://docs.angr.io/core-concepts/toplevel#blocks
        # For each basic block
        for bb in v.blocks:

            # CapstoneBlock: https://github.com/angr/angr/blob/988128beeb08324ea4379e4acb6e353677b41c14/angr/lifter.py#L398
            # CapstoneInsn:  https://github.com/angr/angr/commit/dd1b816a70d8e92497542c373387246c228012ef#diff-8da7ca6fe6ed51c99673f2daa783853eL410
#           print(bb)
#           print(bb.instructions)
#           print(bb.capstone)
#           print(bb.capstone.addr)
#           print(bb.capstone.insns)

            # Get entry address of basic block
            bb_addr = bb.capstone.addr

            # Determine if this basic block can be simulated
            try:
                _ = bb.instruction_addrs
            except SimEngineError as e:
                sys.stderr.write('{0}\n'.format(str(e)))
                continue

            # Create ACFG object
            acfg = ACFG(bb_addr, bb.instructions)

            # Get number of offspring of this basic block
            node = d.aw.cfg.model.get_any_node(bb_addr)

            if node is None:
                sys.stderr.write('Basic block {0} doesn\'t exist in graph\n'.format(hex(bb_addr)))
                continue

            acfg.set_offspring(len(list(d.aw.cfg.graph.successors(node))))

            # Get betweenness of this basic block
            acfg.set_betweenness(betweenness[node])

            # Get instructions in basic block
            for inst in bb.capstone.insns:
                # Count instruction mnemonic
                acfg.count(inst.mnemonic)

                # For each operand in this instruction
                for i in inst.operands:

                    # If this is a numeric constant
                    if i.type == X86_OP_IMM:
                        acfg.add_numeric(i.value.imm)

                    #TODO - string constants
                    # Maybe use emulation: https://bitbucket.org/snippets/Alexander_Hanel/AroeA#cap_stack_str.py-32
                    # Get string constants in basic block

            # Add basic block to list
            rv.append(acfg)

#           print(acfg)

    sys.stdout.write('\n')

    # Return annotated basic blocks (which make up ACFG - Attributed Control Flow Graph)
    return rv

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', help='output debug information', required=False, default=False)

    subparsers = parser.add_subparsers(help='disassembler types help', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('angr', help='use angr to get CFG')
    sp.set_defaults(cmd='angr')
    sp.add_argument('--exe', help='PE file', required=True)
    sp.add_argument('--output', help='output file', required=True)

    sp = subparsers.add_parser('radare2', help='use radare2 to get CFG')
    sp.set_defaults(cmd='radare2')
    sp.add_argument('--exe', help='PE file', required=True)
    sp.add_argument('--output', help='output file', required=True)

    sp = subparsers.add_parser('binja', help='use BinaryNinja to get CFG')
    sp.set_defaults(cmd='binja')
    sp.add_argument('--bndb', help='BNDB file', required=True)
    sp.add_argument('--output', help='output file', required=True)

    sp = subparsers.add_parser('binja_multi', help='use BinaryNinja to get CFG')
    sp.set_defaults(cmd='binja_multi')
    sp.add_argument('--bndb_find', help='BNDB files returned by "find" command', required=True)
    sp.add_argument('--output', help='output folder', required=True)

    args = parser.parse_args()

    # Store arguments
    tool = args.cmd
    debug = bool(args.debug)
    output = args.output

    if tool == 'angr':
        fn = args.exe
    elif tool == 'radare2':
        fn = args.exe
    elif tool == 'binja':
        fn = args.bndb
    elif tool == 'binja_multi':
        fn = args.bndb_find
    else:
        sys.stderr.write('Invalid disassembler: {0}\n'.format(tool))
        sys.exit(1)

    if not os.path.exists(fn):
        sys.stderr.write('{0} does not exist\n'.format(fn))
        sys.exit(1)

    if os.path.exists(output):
        sys.stderr.write('{0} already exists\n'.format(output))
        sys.exit(0)

    # Extract acfg
    if tool == 'angr':
        rv = get_acfg_angr(fn)
        dump(output,rv)
        sys.stdout.write('{0} Finished dumping features\n'.format(fn))
    elif tool == 'radare2':
        rv = get_acfg_radare2(fn,debug)
        dump(output,rv)
        sys.stdout.write('{0} Finished dumping features\n'.format(fn))
    elif tool == 'binja':
        rv = get_acfg_binja(fn,debug)
        time_previous = time.time()
        dump(output,rv)
        sys.stdout.write('{0} Finished dumping features took {1} seconds\n'.format(fn,time.time()-time_previous))

    # NOTE: we have to do it this way or else the OS will kill our processes
    #      for using too much memory
    elif tool == 'binja_multi':
        args = list()

        # Get samples and construct output files
        with open(fn, 'r') as fr:
            for line in fr:
                line = line.strip('\n')

                args.append((line,debug))

        # Extract features
        pool = Pool(8)
        results = pool.imap_unordered(get_acfg_binja_wrapper, args)
        for e,r in enumerate(results):
            loc, rv = r

            # Determine output folder
            outFN = os.path.join(output,loc)

            # Dump features
            dump(outFN,rv)

            sys.stdout.write('{0}\n'.format(outFN))

        pool.close()
        pool.join()

if __name__ == '__main__':
    _main()
