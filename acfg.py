#!/usr/bin/python3

import sys
import os
import numpy as np
import random

# Holds parsed dataset
class ACFG(object):
    # Get samples and labels & shuffle samples
    def __init__(self, train, test, valid, max_len, map_path, shuffle_bb, joint, normalize=False, autoencoder=False):
        self.max_len = max_len

        self.train = list()
        self.test = list()
        self.valid = list()

        self.shuffle_bb = shuffle_bb

        self.joint = joint

        self.normalize = normalize

        self.autoencoder = autoencoder

        label_set = set()

        # Get samples and labels
        with open(train,'r') as fr:
            for line in fr:
                line = line.strip('\n')

                label = line.split('/')[-2]

                self.train.append((line,label))
                label_set.add(label)

        with open(test,'r') as fr:
            for line in fr:
                line = line.strip('\n')

                label = line.split('/')[-2]

                self.test.append((line,label))
                label_set.add(label)

        with open(valid,'r') as fr:
            for line in fr:
                line = line.strip('\n')

                label = line.split('/')[-2]

                self.valid.append((line,label))
                label_set.add(label)

        # Import map of labels
        if os.path.exists(map_path):
            self.label = dict()

            with open(map_path,'r') as fr:
                for line in fr:
                    line = line.strip('\n')
                    k = line.split(' ')[0]
                    v = line.split(' ')[1]

                    self.label[k] = int(v)

        # Export map of labels
        else:
            # Create map of labels
            self.label = dict()

            # If joint classifier, make sure benign is label 0
            if self.joint is True:
                label_set = list(label_set)
                label_set.remove('benign')
                label_set.insert(0,'benign')

            for e,l in enumerate(label_set):
                self.label[l] = e

            with open(map_path,'w') as fw:
                for k,v in self.label.items():
                    fw.write('{0} {1}\n'.format(k,v))

        # If normalizing the ACFG feature vectors
        if self.normalize is True:
            sample = self.train + self.test + self.valid

            self.maximum_val = np.array([0.0]*6)

           ##NOTE: run this once because it takes a while
           ## Get maximum values for each feature vector
           #for e,t in enumerate(sample):
           #    sys.stdout.write('Normalizing samples: {0} / {1}\r'.format(e,len(sample)))
           #    sys.stdout.flush()

           #    fn,l = t

           #    # Only look at first max_len of data (and pad with empty feature vector)
           #    b = np.array([[0,0,0,0,0,0]]*self.max_len, dtype=np.float)
           #    bytez = [eval(line.replace('nan','0.0'))[1:] for line in open(fn,'r').read().split('\n')[:self.max_len] if len(line) > 0]
           #    b[:len(bytez)] = bytez

           #    # Get maximum values for each sample's feature vector
           #    b_max = b.max(axis=0)
           #    maximum_val_index = np.where(b.max(axis=0) > self.maximum_val)[0]
           #    self.maximum_val[maximum_val_index] = b_max[maximum_val_index]

           #sys.stdout.write('\n')

            #NOTE: replace this with found maximum values
            # PEDAL experiments
           #self.maximum_val = np.array([1.09612310e+07,4.26387000e+05,1.09612240e+07,1.15700000e+04,6.97000000e+02,8.82040816e-01])

            # Autoencoder (benign unpacked)
            #self.maximum_val = np.array([1.59328600e+06,2.95246000e+05,1.59328500e+06,1.93900000e+03,1.64000000e+02,4.63423831e-01])

            # Autoencoder (benign unpacked plus, valid, filtered)
            self.maximum_val = np.array([1.59328600e+06,6.23115000e+05,1.59328500e+06,2.07993000e+05,2.18000000e+02,4.66131907e-01])

            print(self.maximum_val)

    # Some getter functions
    def get_train_num(self):
        return len(self.train)
    def get_test_num(self):
        return len(self.test)
    def get_valid_num(self):
        return len(self.valid)
    def get_class_count(self):
        return len(self.label.keys())

    def get_train(self):
        return self.train
    def get_test(self):
        return self.test
    def get_label(self):
        return self.label

    # Data Generator
    def generator(self,t,batch_size):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test
        elif t == 'valid':
            sample = self.valid

        x = list()
        y = list()
        y2 = list()

        while True:
            for fn,l in sample:
                # Convert label
                label_int = self.label[l]

                # Only look at first max_len of data (and pad with empty feature vector)
                b = np.array([[0,0,0,0,0,0]]*self.max_len, dtype=np.float)
                bytez = [eval(line.replace('nan','0.0'))[1:] for line in open(fn,'r').read().split('\n')[:self.max_len] if len(line) > 0]
                b[:len(bytez)] = bytez

                # If we should shuffle the basic blocks
                if self.shuffle_bb:
                    np.random.shuffle(b)

                # Organize data to be fed to keras
                x.append(b)
                y.append([label_int])

                if len(x) == batch_size:
                    if (self.normalize is True) and (self.autoencoder is True):
                        yield (np.asarray(x) / self.maximum_val , np.asarray(x) / self.maximum_val)
                    elif self.normalize is True:
                        yield (np.asarray(x) / self.maximum_val , np.asarray(y))
                    elif self.autoencoder is True:
                        yield (np.asarray(x), np.asarray(x))
                    else:
                        yield (np.asarray(x), np.asarray(y))

                    x = list()
                    y = list()

    # Data Generator for Confusion Matrix
    def confusion_generator(self,t,batch_size):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test
        elif t == 'valid':
            sample = self.valid

        x = list()
        y = list()

        while True:
            for fn,l in sample:
                # Convert label
                label_int = self.label[l]

                # Only look at first max_len of data (and pad with empty feature vector)
                b = np.array([[0,0,0,0,0,0]]*self.max_len, dtype=np.float)
                bytez = [eval(line.replace('nan','0.0'))[1:] for line in open(fn,'r').read().split('\n')[:self.max_len] if len(line) > 0]
                b[:len(bytez)] = bytez

                # If we should shuffle the basic blocks
                if self.shuffle_bb:
                    np.random.shuffle(b)

                # Organize data to be fed to keras
                x.append(b)
                y.append([label_int])

                if len(x) == batch_size:
                    yield (fn,np.asarray(x), np.asarray(y))
                    x = list()
                    y = list()

    # Data generator for LIME
    def explain_generator(self,t):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test
        elif t == 'valid':
            sample = self.valid

        for fn,l in sample:
            # Convert label
            label_int = self.label[l]
            yield (fn,label_int)

    # Data generator for SHAP
    def explain_generator_shap_balanced(self,t,num_per_class):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test
        elif t == 'valid':
            sample = self.valid

        # Group samples by class
        fam = dict()
        for fn,l in sample:
            if l not in fam:
                fam[l] = list()
            fam[l].append(fn)

        # Get num_per_class samples per class
        for l,s in fam.items():
            c = 0
            for fn in s:
                # Convert label
                label_int = self.label[l]
                yield (fn,label_int)

                c += 1
                if c == num_per_class:
                    break

    # Data generator for SHAP
    def explain_generator_shap(self,t,num):
        sample = None
        if t == 'train':
            sample = self.train
        elif t == 'test':
            sample = self.test
        elif t == 'valid':
            sample = self.valid
        elif t == 'all':
            sample = self.train + self.test

        # Get num samples
        c = 0
        for fn,l in sample:
            # If label doesn't exist, that means it's a new sample
            # Give it a label of -1
            if l not in self.label:
                label_int = -1
            else:
                # Convert label
                label_int = self.label[l]
            yield (fn,label_int)

            c += 1
            if c == num:
                break