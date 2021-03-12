#!/usr/bin/python3

import sys
import os
import numpy as np

# Get DR features for training autoencoder
class DR(object):
    # Get samples
    def __init__(self, trainFN, testFN, max_len, normalizeFN):
        self.train = list()
        self.test = list()
        self.valid = list()

        self.normalizeFN = normalizeFN

        self.max_len = max_len

        label_set = set()

        # Get samples
        with open(trainFN,'r') as fr:
            for line in fr:
                line = line.strip('\n')

                self.train.append(line)

        # For mse.py
        if testFN is not None:
            with open(testFN,'r') as fr:
                for line in fr:
                    line = line.strip('\n')

                    self.test.append(line)

        # If normalizing the DR feature vectors
        if self.normalizeFN is not None:
            self.maximum_val = np.load(self.normalizeFN)

    # Some getter functions
    def get_train_num(self):
        return len(self.train)
    def get_test_num(self):
        return len(self.test)

    def get_train(self):
        return self.train
    def get_test(self):
        return self.test

    # Determines max values to normalize
    def get_max(self,outputFN):
        sample = self.train + self.test

        maximum_val = np.array([0.0]*18)

        # Get maximum values for each feature vector
        for e,fn in enumerate(sample):
            sys.stdout.write('Normalizing samples: {0} / {1}\r'.format(e+1,len(sample)))
            sys.stdout.flush()

            # Only look at first max_len of data (and pad with empty feature vector)
            b = np.array([[0]*18]*self.max_len, dtype=float)
            bytez = np.load(fn)

            # If nothing was loaded, ignore this sample
            if len(bytez) == 0:
                sys.stderr.write('Error. Sample {0} has no data.\n'.format(fn))
                continue

            bytez = bytez[:self.max_len]
            # First element is the entry point, so we should ignore this
            bytez = bytez[:,1:]
            b[:len(bytez)] = bytez

            # Get maximum values for each sample's feature vector
            b_max = b.max(axis=0)
            maximum_val_index = np.where(b.max(axis=0) > maximum_val)[0]
            maximum_val[maximum_val_index] = b_max[maximum_val_index]

        sys.stdout.write('\n')

        # Output numpy array
        np.save(outputFN, maximum_val)

    # Gets path of index
    def get_path(self,t,e):
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test

        return sample[e]

    # Data generator for training autoencoder
    def generator(self,t,batch_size):
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test

        x = list()

        while True:
            for fn in sample:
                # Only look at first max_len of data (and pad with empty feature vector)
                b = np.array([[0]*18]*self.max_len, dtype=float)
                bytez = np.load(fn)

                # If nothing was loaded, ignore this sample
                if len(bytez) == 0:
                    sys.stderr.write('Error. Sample {0} has no data.\n'.format(fn))
                    continue

                bytez = bytez[:self.max_len]
                # First element is the entry point, so we should ignore this
                bytez = bytez[:,1:]
                b[:len(bytez)] = bytez

                # Organize data to be fed to keras
                x.append(b)

                if len(x) == batch_size:
                    if self.normalizeFN is not None:
                        yield (np.asarray(x) / self.maximum_val , np.asarray(x) / self.maximum_val)
                    else:
                        yield (np.asarray(x), np.asarray(x))

                    x = list()

# Extract RoIs (basic blocks) and function highlights
class RoI(object):
    # Get samples
    def __init__(self, sample, thresh, normalizeFN, funcFlag=False,windowFlag=False,bbFlag=False,avgFlag=False,avgstdevFlag=False):
        self.sample = sample

        self.funcFlag = funcFlag
        self.windowFlag = windowFlag
        self.bbFlag = bbFlag

        self.avgFlag = avgFlag
        self.avgstdevFlag = avgstdevFlag

        self.normalizeFN = normalizeFN
        self.thresh = thresh

        # If normalizing the DR feature vectors
        if self.normalizeFN is not None:
            self.maximum_val = np.load(self.normalizeFN)

    # Returns MSE values (and BB addresses) over or equal to threshold
    def parse(self, mseFN, featureFN, thresh):
        rv_addr = list()
        rv_mse = list()

        # Read data
        mse = np.load(mseFN)

        addr = list()

        # Read feature addresses
        feature = np.load(featureFN)
        for a in feature:
            addr.append(int(a[0]))

        # Extend addr if necessary (address of -1 denotes padding)
        if len(addr) < len(mse):
            diff = len(mse) - len(addr)
            addr.extend(['-1']*diff)

        # Identify highlighted basic blocks
        index = np.where(mse >= thresh)[0]

        for i in index:
            a = int(addr[i])
            m = float(mse[i])
            rv_addr.append(a)
            rv_mse.append(m)

        return rv_addr,rv_mse

    # Retreives mapping between basic blocks and the function they belong to
    def get_mapping(self,funcFN):
        bb_map = dict()
        func_map = dict()

        # Get functions & bb's in binary
        with open(funcFN,'r') as fr:
            for line in fr:
                line = line.strip('\n')
                split = line.split(' ')

                # Corrupted line. Most likely obfuscated function name
                if len(split) < 4:
                    continue

                funcAddr = split[0]
                bbAddr = split[1]
                funcSymbolType = split[-2] # NOTE: this is because sometimes the function's name has spaces in it
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

        return bb_map,func_map

    # Some getter functions
    def get_sample_num(self):
        return len(self.sample)

    # RoI data generator for extracting feature vectors used for clustering
    def roi_generator(self):
        for mseFN,funcFN,featureFN in self.sample:
            sys.stdout.write('{0}\n'.format(mseFN))
            sys.stdout.flush()

            feature_map = dict()

            # Get features of basic blocks
            feature = np.load(featureFN)

            for a in feature:
                feature_map[int(a[0])] = a[1:]

            # Get highlighted basic block addresses and corresponding MSE values
            addr,mse = self.parse(mseFN,featureFN,self.thresh)

            if len(addr) == 0:
                sys.stderr.write('{0}: Nothing was highlighted.\n'.format(mseFN))
                continue
            if (len(set(addr)) == 1) and (-1 in set(addr)):
                sys.stderr.write('{0}: Only padding was highlighted.\n'.format(mseFN))
                continue

            sys.stdout.write('Number of RoIs (basic blocks): {0}\n'.format(len(addr)))
#           sys.stderr.write('BB HIGHLIGHT: {0} {1}\n'.format(mseFN,addr))

            # Get mapping between basic blocks and the functions they belong to
            bb_map,func_map = self.get_mapping(funcFN)

            # Retrieve data
            x = list()
            x_func = list()

            funcs = set()

            # For each highlighted basic block, get its function
            for bb in addr:
                # Ignore padding
                if bb == -1:
                    continue

                # If BB not in a valid function
                if bb not in bb_map.keys():
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
                        if bb not in feature_map:
                            sys.stderr.write('{0}: Error: BB {1} in binary, but not in features\n'.format(mseFN,hex(bb)))
                            sys.exit()
                            continue

                        feature = feature_map[bb]
                        # Normalize data
                        if self.normalizeFN is not None:
                            feature = feature / self.maximum_val

                        if len(tmp) == 0:
                            tmp = feature
                        else:
                            tmp = np.vstack((tmp,feature))

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
                        if bb not in feature_map:
                            sys.stderr.write('{0}: Error: BB {1} in binary, but not in features\n'.format(mseFN,hex(bb)))
                            sys.exit()
                            continue

                        feature = feature_map[bb]
                        # Normalize data
                        if self.normalizeFN is not None:
                            feature = feature / self.maximum_val

                        if len(tmp) == 0:
                            tmp = feature
                        else:
                            tmp = np.vstack((tmp,feature))

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
                        if bb not in feature_map:
                            sys.stderr.write('{0}: Error: BB {1} in binary, but not in features\n'.format(mseFN,hex(bb)))
                            sys.exit()
                            continue

                        feature = feature_map[bb]
                        # Normalize data
                        if self.normalizeFN is not None:
                            feature = feature / self.maximum_val

                        if len(tmp) == 0:
                            tmp = feature
                        else:
                            tmp = np.vstack((tmp,feature))

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
                    if bb not in feature_map:
                        sys.stderr.write('{0}: Error: BB {1} in binary, but not in features\n'.format(mseFN,hex(bb)))
                        sys.exit()
                        continue

                    func = bb_map[bb]
                    feature = feature_map[bb]
                    # Normalize data
                    if self.normalizeFN is not None:
                        feature = feature / self.maximum_val

                    #print(hex(bb),hex(func),feature,len(feature))

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
                sys.stderr.write('{0}: Note, no highlights contain internal functions.\n'.format(mseFN))
                continue

            # For each array (representing each function highlighted and its feature vector based on RoIs)
            if rv.ndim == 2:
                for e,r in enumerate(rv):
                    yield (mseFN,hex(x_func[e]),r)
            else:
                yield (mseFN,hex(x_func[0]),rv)

    # Generator for outputting MSE values for highlighted functions
    def function_highlight_generator(self):
        for mseFN,funcFN,featureFN in self.sample:
            sys.stdout.write('{0}\n'.format(mseFN))
            sys.stdout.flush()

            # Get highlighted basic block addresses and corresponding MSE values
            addr,mse = self.parse(mseFN,featureFN,self.thresh)

            if len(addr) == 0:
                sys.stderr.write('{0}: Nothing was highlighted.\n'.format(mseFN))
                continue
            if (len(set(addr)) == 1) and (-1 in set(addr)):
                sys.stderr.write('{0}: Only padding was highlighted.\n'.format(mseFN))
                continue

            sys.stdout.write('Number of RoIs (basic blocks): {0}\n'.format(len(addr)))

            # Get mapping between basic blocks and the functions they belong to
            bb_map,func_map = self.get_mapping(funcFN)

            # Aggregrate MSE values for each function
            mse_func = dict()

            # Return each RoI MSE score for each function
            for i,bb_addr in enumerate(addr):
                # Ignore padding highlights
                if bb_addr == -1:
                    continue

                # Ignore basic blocks not in relevant functions
                if bb_addr not in bb_map.keys():
                    continue

                # Get function this basic block belongs to
                f_addr = bb_map[bb_addr]

                # Get MSE value of basic block
                m = mse[i]

                # Append MSE value of basic block to dictionary of function MSE values
                if f_addr not in mse_func.keys():
                    mse_func[f_addr] = list()
                mse_func[f_addr].append((bb_addr,m))

            yield (mse_func,mseFN)
