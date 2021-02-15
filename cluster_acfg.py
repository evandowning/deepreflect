#!/usr/bin/python3

import sys
import os
import numpy as np
import re

# Holds parsed dataset
class ACFG(object):
    # Get samples and labels
    def __init__(self, train, test, map_path, parse_path, area=-1, area_type=None, new_parse_path=None):
        self.train = list()
        self.test = list()

        label_set = set()

        # Import map of labels
        self.label = dict()

        # Determines if we need to look at ACFGs surrounding highlighted
        # ACFG
        self.area = area
        self.area_type = area_type

        # If map_path is none, then dbscan_predict.py is constructing this object
        if map_path is not None:
            with open(map_path,'r') as fr:
                for line in fr:
                    line = line.strip('\n')
                    k = line.split(' ')[0]
                    v = line.split(' ')[1]

                    self.label[k] = int(v)

        # Get samples
        self.train = train
        self.test = test

        self.parse_path = parse_path
        self.new_parse_path = new_parse_path

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

        x = list()
        y = list()

        while True:
            for fn,l in sample:
                acfg = dict()

                # Read in ACFG features
                with open(fn,'r') as fr:
                    for line in fr:
                        line = line.strip('\n')
                        line = eval(line.replace('nan','0.0'))

                        addr = line[0]

                        acfg[hex(addr)] = np.array(line[1:], dtype=np.float)

                h = fn.split('/')[-1]
                f_name = fn.split('/')[-2]

                parseFN = os.path.join(self.parse_path,h,l+'.txt')

                if not os.path.exists(parseFN):
                    # If not exists, try new parse path
                    if self.new_parse_path is not None:
                        parseFN = os.path.join(self.new_parse_path,h,l+'.txt')

                    if not os.path.exists(parseFN):
                        sys.stderr.write('Error. {0} does not exist\n'.format(parseFN))
                        continue

                # Multiple features within a single BB can be highlighted, but we're
                # just interested in each unique highlighted BB
                bb_addr = set()

                # Read through highlighted basic blocks
                with open(parseFN,'r') as fr:
                    for line in fr:
                        m = re.match(r'^BB: (\w+) | Feature: .* \|.*',line)

                        # Get basic block address
                        if m:
                            bb_addr.add(m.group(1))

                # For each basic block, get ACFG
                for entry_addr in bb_addr:
                    if self.area_type is None:
                        x.append(acfg[entry_addr])
                        y.append([int(l)])

                        if len(x) == batch_size:
                            yield (fn,entry_addr,np.asarray(x),np.asarray(y))
                            x = list()
                            y = list()
                    else:
                        sorted_keys = sorted(acfg.keys())
                        index = sorted_keys.index(entry_addr)

                        # If the number of keys < area then we can't take
                        # a window
                        if len(sorted_keys) < (self.area*2 + 1):
                            sys.stderr.write('Error. {0}: Number of BB\s ({1}) < area ({2} + 1)\n'.format(fn,len(sorted_keys),self.area))
                            continue

                        # Get two before and two after
                        tmp = acfg[entry_addr]
                        for i in range(1,self.area+1):
                            if index-i >= 0:
                                tmp = np.vstack((tmp,acfg[sorted_keys[index-i]]))
                            if index+i < len(sorted_keys):
                                tmp = np.vstack((tmp,acfg[sorted_keys[index+i]]))

                        # Calculate average
                        if self.area_type == 'avg':
                            fv = np.average(tmp, axis=0)

                        # Calculate average and standard deviation
                        elif self.area_type == 'avgstdev':
                            fv_avg = np.average(tmp, axis=0)
                            fv_std = np.std(tmp, axis=0)

                            fv = np.append(fv_avg,fv_std)

                        # Calculate average of just other blocks
                        elif self.area_type == 'avgother':
                            tmp = tmp[1:]
                            fv_avg = np.average(tmp, axis=0)

                            fv = np.append(acfg[entry_addr],fv_avg)

                        x.append(fv)
                        y.append([int(l)])

                        if len(x) == batch_size:
                            yield (fn,entry_addr,np.asarray(x),np.asarray(y))
                            x = list()
                            y = list()

            yield (None,None,None,None)
