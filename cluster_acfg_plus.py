#!/usr/bin/python3

import sys
import os
import numpy as np
import re

from parse_autoencoder_plus import parse

# Holds parsed dataset
class ACFG_plus(object):
    # Get samples
    def __init__(self, train, test, thresh, avgFlag=False, avgstdevFlag=False,funcFlag=False,windowFlag=False,bbFlag=False):
        self.train = list()
        self.test = list()

        self.thresh = thresh

        self.avgFlag = avgFlag
        self.avgstdevFlag = avgstdevFlag
        self.funcFlag = funcFlag
        self.windowFlag = windowFlag
        self.bbFlag = bbFlag

        # Get samples
        self.train = train
        self.test = test

    # Some getter functions
    def get_train_num(self):
        return len(self.train)
    def get_test_num(self):
        return len(self.test)

    # Data Generator
    def generator(self,t,batch_size):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test

        while True:
            for fn,bndbfuncFN,acfgFN,l in sample:
                sys.stdout.write('{0}\n'.format(fn))
                sys.stdout.flush()

                acfg_map = dict()

                # Get acfg features for basic blocks
                acfg = np.load(acfgFN)

                for a in acfg:
                    acfg_map[int(a[0])] = a[1:]

                # Get highlighted basic block addresses and corresponding MSE values
                addr,mse = parse(fn,acfgFN,self.thresh)

                #NOTE: debugging, print out basic block highlights
                sys.stderr.write('BB HIGHLIGHT: {0} {1}\n'.format(fn,addr))

                if len(addr) == 0:
                    sys.stderr.write('{0}: Note: Nothing was highlighted\n'.format(fn))
                    continue
                if (len(set(addr)) == 1) and (-1 in set(addr)):
                    sys.stderr.write('{0}: Note: Only padding was highlighted\n'.format(fn))
                    continue

                bb_map = dict()
                func_map = dict()

                # Get functions & bb's in binary
                with open(bndbfuncFN,'r') as fr:
                    for line in fr:
                        line = line.strip('\n')
                        split = line.split(' ')

                        # Corrupted line. Most likely obfuscated function name
                        if len(split) < 4:
                            continue

                        funcAddr = split[0]
                        bbAddr = split[1]
                        # NOTE: this is because sometimes the function's name has spaces in it
                        funcSymbolType = split[-2]
                        funcSymbolTypeName = split[-1]

                        # If not an function symbol, ignore
                        if funcSymbolType != '0':
                            continue

                        funcAddr = int(funcAddr)
                        bbAddr = int(bbAddr)

                        bb_map[bbAddr] = funcAddr

                        if funcAddr not in func_map:
                            func_map[funcAddr] = list()
                        func_map[funcAddr].append(bbAddr)

                # Retrieve data
                x = list()
                x_func = list()

                # Autoencoder (benign unpacked plus, valid, filtered)
                maximum_val = np.array([2.18000000e+02,4.66131907e-01,1.59328500e+06,1.00820000e+04,7.68000000e+02,5.19373000e+05,1.20040000e+04,4.36000000e+02,4.10000000e+01,6.00000000e+00,7.00000000e+00,3.00000000e+00,6.00000000e+00,1.50000000e+01,2.00000000e+00,3.00000000e+00,5.00000000e+00,5.00000000e+00])

                funcs = set()

                # For each highlighted basic block, get its function
                for bb in addr:
                    # Ignore padding
                    if bb == -1:
                        continue

                    if bb not in bb_map:
                        sys.stderr.write('{0}: Error: Highlighted BB {1} not found\n'.format(fn,hex(bb)))
                        continue

                    func = bb_map[bb]
                    funcs.add(func)

                # Take all basic blocks in highlighted functions
                if self.funcFlag:
                    # For each highlighted function
                    for f in funcs:
                        tmp = np.array([])

                        # For each basic block in this function
                        for bb in sorted(func_map[f]):
                            # NOTE: not sure why this happened with binaryninja. maybe different/updated versions?
                            if bb not in acfg_map:
                                sys.stderr.write('{0}: Error: BB {1} in binary, but not in ACFG features\n'.format(fn,hex(bb)))
                                continue

                            acfg = acfg_map[bb]
                            # Normalize data
                            acfg = acfg / maximum_val

                            if len(tmp) == 0:
                                tmp = acfg
                            else:
                                tmp = np.vstack((tmp,acfg))

                        x.append(tmp)
                        x_func.append(f)

                # Take only basic blocks window-wise within each highlighted function
                elif self.windowFlag:
                    # For each highlighted function
                    for f in funcs:
                        tmp = np.array([])

                        # Get first and last highlighted basic block in function
                        sorted_bb = sorted(func_map[f])
                        common = sorted(set(sorted_bb).intersection(set(addr)))
                        start = common[0]
                        end = common[-1]
                        start_index = sorted_bb.index(start)
                        end_index = sorted_bb.index(end)

                        # For each basic block in this function's window
                        for bb in sorted_bb[start_index:end_index+1]:
                            # NOTE: not sure why this happened with binaryninja. maybe different/updated versions?
                            if bb not in acfg_map:
                                sys.stderr.write('{0}: Error: BB {1} in binary, but not in ACFG features\n'.format(fn,hex(bb)))
                                continue

                            acfg = acfg_map[bb]
                            # Normalize data
                            acfg = acfg / maximum_val

                            if len(tmp) == 0:
                                tmp = acfg
                            else:
                                tmp = np.vstack((tmp,acfg))

                        x.append(tmp)
                        x_func.append(f)

                # Take only highlighted basic blocks within each highlighted function
                elif self.bbFlag:
                    # For each highlighted function
                    for f in funcs:
                        tmp = np.array([])

                        # For each basic block in this function
                        for bb in sorted(func_map[f]):
                            # If this basic block was not highlighted, ignore it
                            if bb not in addr:
                                continue

                            # NOTE: not sure why this happened with binaryninja. maybe different/updated versions?
                            if bb not in acfg_map:
                                sys.stderr.write('{0}: Error: BB {1} in binary, but not in ACFG features\n'.format(fn,hex(bb)))
                                continue

                            acfg = acfg_map[bb]
                            # Normalize data
                            acfg = acfg / maximum_val

                            if len(tmp) == 0:
                                tmp = acfg
                            else:
                                tmp = np.vstack((tmp,acfg))

                        x.append(tmp)
                        x_func.append(f)

                    # For each highlighted basic block
                    for bb in addr:
                        # Ignore padding
                        if bb == -1:
                            continue

                        if bb not in bb_map:
                            # Error is already being reported above
                            continue

                        # NOTE: not sure why this happened with binaryninja. maybe different/updated versions?
                        if bb not in acfg_map:
                            sys.stderr.write('{0}: Error: BB {1} in binary, but not in ACFG features\n'.format(fn,hex(bb)))
                            continue

                        func = bb_map[bb]
                        acfg = acfg_map[bb]
                        # Normalize data
                        acfg = acfg / maximum_val

                        #print(hex(bb),hex(func),acfg,len(acfg))

                # Final computation on data
                if self.avgFlag:
                    rv = np.array([])

                    for s in x:
                        # Average these values
                        if s.ndim == 2:
                            avg = np.average(s,axis=0)
                        else:
                            avg = s

                        if len(rv) == 0:
                            rv = avg
                        else:
                            rv = np.vstack((rv,avg))

                elif self.avgstdevFlag:
                    rv = np.array([])

                    for s in x:
                        # Average these values
                        if s.ndim == 2:
                            avg = np.average(s,axis=0)
                            std = np.std(s,axis=0)
                            avgstd = np.append(avg,std)
                        else:
                            avgstd = np.append(s,s)

                        if len(rv) == 0:
                            rv = avgstd
                        else:
                            rv = np.vstack((rv,avgstd))

                # If nothing was extracted, ignore this sample
                if len(rv) == 0:
                    sys.stderr.write('{0}: Note, no highlights contain internal functions.\n'.format(fn))
                    continue

                # For each array (representing each function highlighted)
                if rv.ndim == 2:
                    for e,r in enumerate(rv):
                        yield (fn,hex(x_func[e]),r,np.array([l]))
                else:
                    yield (fn,hex(x_func[0]),rv,np.array([l]))

            yield (None,None,None,None)
