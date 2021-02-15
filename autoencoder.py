#!/usr/bin/python3

import sys
import os
import argparse
import math

from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import backend as backend
from keras import metrics

import tensorflow as tf

from acfg import ACFG
from acfg_plus import ACFG_plus

import multi_gpu

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='number of GPUs', default=1)
    parser.add_argument('--kernel', help='size of kernel', type=int, required=False, default=500)
    parser.add_argument('--strides', help='size of stride', type=int, required=False, default=500)
    parser.add_argument('--batch-size', help='batch size', type=int, required=False, default=10)

    parser.add_argument('--option', help='autoencoder option [1-7]', required=True)

    subparsers = parser.add_subparsers(help='dataset types help', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('acfg', help='basic block features of malware dataset')
    sp.set_defaults(cmd='acfg')
    sp.add_argument('--train', help='training set files', required=True)
    sp.add_argument('--test', help='testing set files', required=True)
    sp.add_argument('--valid', help='validation set files', required=True)
    sp.add_argument('--model', help='model path', required=True)
    sp.add_argument('--map', help='class map path', required=True)
    sp.add_argument('--shuffle-bb', help='shuffle basic block ordering', required=False, default=False)
    sp.add_argument('--normalize', help='normalize features', required=False, default=False)
    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)

    sp = subparsers.add_parser('acfg_plus', help='basic block features of malware dataset')
    sp.set_defaults(cmd='acfg_plus')
    sp.add_argument('--train', help='training set files', required=True)
    sp.add_argument('--test', help='testing set files', required=True)
    sp.add_argument('--valid', help='validation set files', required=True)
    sp.add_argument('--model', help='model path', required=True)
    sp.add_argument('--map', help='class map path', required=True)
    sp.add_argument('--shuffle-bb', help='shuffle basic block ordering', required=False, default=False)
    sp.add_argument('--normalize', help='normalize features', required=False, default=False)
    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)

    args = parser.parse_args()

    # Store arguments
    dataset = args.cmd
    ngpus = int(args.gpus)
    kernel_size = int(args.kernel)
    strides = int(args.strides)
    batch_size = int(args.batch_size)
    trainset = args.train
    testset = args.test
    validset = args.valid
    model_path = args.model
    map_path = args.map
    M = int(args.option)

    if dataset == 'acfg':
        max_len = int(args.max_bb)
    elif dataset == 'acfg_plus':
        max_len = int(args.max_bb)

    # Import dataset
    if dataset == 'acfg':
        data = ACFG(trainset,testset,validset,max_len,map_path,bool(args.shuffle_bb),True,bool(args.normalize),True)
    elif dataset == 'acfg_plus':
        data = ACFG_plus(trainset,testset,validset,max_len,map_path,bool(args.shuffle_bb),False,bool(args.normalize),True)

    # Get number of classes
    class_count = data.get_class_count()

    # define model structure

    # If ACFG data
    if dataset == 'acfg':
        if M == 1:
            print("Unet basic")
            inputs = Input(shape=(max_len,6))
            #inputs = Input(input_size)
            strides=1
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling1D(pool_size=2,strides=2)(conv1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling1D(pool_size=2,strides=2)(conv2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling1D(pool_size=2,strides=2)(conv3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2,strides=2)(drop4)

            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(drop5))
            merge6 = concatenate([drop4, up6], axis=2)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv6))
            merge7 = concatenate([conv3, up7], axis=2)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv7))
            merge8 = concatenate([conv2, up8], axis=2)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv8))
            merge9 = concatenate([conv1, up9], axis=2)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(6, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv1D(6, 1, activation='tanh')(conv9)

            model = Model(input=inputs, output=conv10)

            model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel = model
            basemodel.summary()
        elif M == 2:
            print("Unet extra filters")
            inputs = Input(shape=(max_len, 6))
            # inputs = Input(input_size)
            factor = 2
            strides = 1
            conv1 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)
            conv1 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling1D(pool_size=2, strides=2)(conv1)
            conv2 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool1)
            conv2 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling1D(pool_size=2, strides=2)(conv2)
            conv3 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool2)
            conv3 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling1D(pool_size=2, strides=2)(conv3)
            conv4 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool3)
            conv4 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2, strides=2)(drop4)

            conv5 = Conv1D(1024*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool4)
            conv5 = Conv1D(1024*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(drop5))
            merge6 = concatenate([drop4, up6], axis=2)
            conv6 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge6)
            conv6 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv6)

            up7 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv6))
            merge7 = concatenate([conv3, up7], axis=2)
            conv7 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge7)
            conv7 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv7)

            up8 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv7))
            merge8 = concatenate([conv2, up8], axis=2)
            conv8 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge8)
            conv8 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv8)

            up9 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv8))
            merge9 = concatenate([conv1, up9], axis=2)
            conv9 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge9)
            conv9 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(6*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv1D(6, 1, activation='tanh')(conv9)

            model = Model(input=inputs, output=conv10)

            model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel = model
            basemodel.summary()
        elif M == 3:
            print("Unet extra conv")
            inputs = Input(shape=(max_len, 6))
            # inputs = Input(input_size)
            strides = 1
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv1)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling1D(pool_size=2, strides=2)(conv1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv2)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling1D(pool_size=2, strides=2)(conv2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv3)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling1D(pool_size=2, strides=2)(conv3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv4)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2, strides=2)(drop4)

            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool4)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv5)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(drop5))
            merge6 = concatenate([drop4, up6], axis=2)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge6)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv6)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv6)

            up7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv6))
            merge7 = concatenate([conv3, up7], axis=2)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge7)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv7)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv7)

            up8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv7))
            merge8 = concatenate([conv2, up8], axis=2)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge8)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv8)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv8)

            up9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv8))
            merge9 = concatenate([conv1, up9], axis=2)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge9)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(6, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv1D(6, 1, activation='tanh')(conv9)

            model = Model(input=inputs, output=conv10)

            model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel = model
            basemodel.summary()
        elif M == 4:
            print("Unet extra conv x2")
            inputs = Input(shape=(max_len,6))
            #inputs = Input(input_size)
            strides=1
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling1D(pool_size=2,strides=2)(conv1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling1D(pool_size=2,strides=2)(conv2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling1D(pool_size=2,strides=2)(conv3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2,strides=2)(drop4)

            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(drop5))
            merge6 = concatenate([drop4, up6], axis=2)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
            conv6 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv6))
            merge7 = concatenate([conv3, up7], axis=2)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            conv7 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv7))
            merge8 = concatenate([conv2, up8], axis=2)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
            conv8 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv8))
            merge9 = concatenate([conv1, up9], axis=2)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(6, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv1D(6, 1, activation='tanh')(conv9)

            model = Model(input=inputs, output=conv10)

            model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel = model
            basemodel.summary()
        elif M == 5:
            print("Unet pix2pix stride1")
            stride=1
            merge_mode = 'concat'

            # batch norm mode
            bn_mode = 2

            # batch norm merge axis
            bn_axis = 1

            input_layer = Input(shape=(max_len,6))

            # 1 encoder C64
            # skip batchnorm on this layer on purpose (from paper)
            en_1 = Convolution1D(nb_filter=64, filter_length=kernel_size, border_mode='same', subsample_length=stride)(input_layer)
            en_1 = LeakyReLU(alpha=0.2)(en_1)

            # 2 encoder C128
            en_2 = Convolution1D(nb_filter=128, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_1)
            en_2 = BatchNormalization(name='gen_en_bn_2')(en_2)
            en_2 = LeakyReLU(alpha=0.2)(en_2)

            # 3 encoder C256
            en_3 = Convolution1D(nb_filter=256, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_2)
            en_3 = BatchNormalization(name='gen_en_bn_3')(en_3)
            en_3 = LeakyReLU(alpha=0.2)(en_3)

            # 4 encoder C512
            en_4 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_3)
            en_4 = BatchNormalization(name='gen_en_bn_4')(en_4)
            en_4 = LeakyReLU(alpha=0.2)(en_4)

            # 5 encoder C512
            en_5 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_4)
            en_5 = BatchNormalization(name='gen_en_bn_5')(en_5)
            en_5 = LeakyReLU(alpha=0.2)(en_5)

            # 6 encoder C512
            en_6 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_5)
            en_6 = BatchNormalization(name='gen_en_bn_6')(en_6)
            en_6 = LeakyReLU(alpha=0.2)(en_6)

            # 7 encoder C512
            en_7 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_6)
            en_7 = BatchNormalization(name='gen_en_bn_7')(en_7)
            en_7 = LeakyReLU(alpha=0.2)(en_7)

            # 8 encoder C512
            en_8 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_7)
            en_8 = BatchNormalization(name='gen_en_bn_8')(en_8)
            en_8 = LeakyReLU(alpha=0.2)(en_8)

            # -------------------------------
            # DECODER
            # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
            # 1 layer block = Conv - Upsample - BN - DO - Relu
            # also adds skip connections (merge). Takes input from previous layer matching encoder layer
            # -------------------------------
            # 1 decoder CD512 (decodes en_8)
            de_1 = UpSampling1D(size=stride)(en_8)
            de_1 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same')(de_1)
            de_1 = BatchNormalization(name='gen_de_bn_1')(de_1)
            de_1 = Dropout(p=0.5)(de_1)
            de_1 = concatenate([de_1, en_7], axis=2)
            de_1 = Activation('relu')(de_1)

            # 2 decoder CD1024 (decodes en_7)
            de_2 = UpSampling1D(size=stride)(de_1)
            de_2 = Convolution1D(nb_filter=1024, filter_length=kernel_size, border_mode='same')(de_2)
            de_2 = BatchNormalization(name='gen_de_bn_2')(de_2)
            de_2 = Dropout(p=0.5)(de_2)
            de_2 = concatenate([de_2, en_6], axis=2)
            de_2 = Activation('relu')(de_2)

            # 3 decoder CD1024 (decodes en_6)
            de_3 = UpSampling1D(size=stride)(de_2)
            de_3 = Convolution1D(nb_filter=1024, filter_length=kernel_size, border_mode='same')(de_3)
            de_3 = BatchNormalization(name='gen_de_bn_3')(de_3)
            de_3 = Dropout(p=0.5)(de_3)
            de_3 = concatenate([de_3, en_5], axis=2)
            de_3 = Activation('relu')(de_3)

            # 4 decoder CD1024 (decodes en_5)
            de_4 = UpSampling1D(size=stride)(de_3)
            de_4 = Convolution1D(nb_filter=1024, filter_length=kernel_size, border_mode='same')(de_4)
            de_4 = BatchNormalization(name='gen_de_bn_4')(de_4)
            de_4 = Dropout(p=0.5)(de_4)
            de_4 = concatenate([de_4, en_4], axis=2)
            de_4 = Activation('relu')(de_4)

            # 5 decoder CD1024 (decodes en_4)
            de_5 = UpSampling1D(size=stride)(de_4)
            de_5 = Convolution1D(nb_filter=1024, filter_length=kernel_size, border_mode='same')(de_5)
            de_5 = BatchNormalization(name='gen_de_bn_5')(de_5)
            de_5 = Dropout(p=0.5)(de_5)
            de_5 = concatenate([de_5, en_3], axis=2)
            de_5 = Activation('relu')(de_5)

            # 6 decoder C512 (decodes en_3)
            de_6 = UpSampling1D(size=stride)(de_5)
            de_6 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same')(de_6)
            de_6 = BatchNormalization(name='gen_de_bn_6')(de_6)
            de_6 = Dropout(p=0.5)(de_6)
            de_6 = concatenate([de_6, en_2], axis=2)
            de_6 = Activation('relu')(de_6)

            # 7 decoder CD256 (decodes en_2)
            de_7 = UpSampling1D(size=stride)(de_6)
            de_7 = Convolution1D(nb_filter=256, filter_length=kernel_size, border_mode='same')(de_7)
            de_7 = BatchNormalization(name='gen_de_bn_7')(de_7)
            de_7 = Dropout(p=0.5)(de_7)
            de_7 = concatenate([de_7, en_1], axis=2)
            de_7 = Activation('relu')(de_7)

            # After the last layer in the decoder, a convolution is applied
            # to map to the number of output channels (3 in general,
            # except in colorization, where it is 2), followed by a Tanh
            # function.
            de_8 = UpSampling1D(size=stride)(de_7)
            de_8 = Convolution1D(nb_filter=6, filter_length=kernel_size, border_mode='same')(de_8)
            de_8 = Activation('tanh')(de_8)

            basemodel = Model(input=[input_layer], output=[de_8], name='unet_generator')
            basemodel.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel.summary()
        elif M == 6:
            print("Unet pix2pix stride2")
            stride = 2
            merge_mode = 'concat'

            # batch norm mode
            bn_mode = 2

            # batch norm merge axis
            bn_axis = 1

            input_layer = Input(shape=(max_len,6))

            # 1 encoder C64
            # skip batchnorm on this layer on purpose (from paper)
            en_1 = Convolution1D(nb_filter=64, filter_length=kernel_size, border_mode='same', subsample_length=stride)(input_layer)
            en_1 = LeakyReLU(alpha=0.2)(en_1)

            # 2 encoder C128
            en_2 = Convolution1D(nb_filter=128, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_1)
            en_2 = BatchNormalization(name='gen_en_bn_2')(en_2)
            en_2 = LeakyReLU(alpha=0.2)(en_2)

            # 3 encoder C256
            en_3 = Convolution1D(nb_filter=256, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_2)
            en_3 = BatchNormalization(name='gen_en_bn_3')(en_3)
            en_3 = LeakyReLU(alpha=0.2)(en_3)

            # 4 encoder C512
            en_4 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_3)
            en_4 = BatchNormalization(name='gen_en_bn_4')(en_4)
            en_4 = LeakyReLU(alpha=0.2)(en_4)

            # 5 encoder C512
            en_5 = Convolution1D(nb_filter=1024, filter_length=kernel_size, border_mode='same', subsample_length=(stride))(en_4)
            en_5 = BatchNormalization(name='gen_en_bn_5')(en_5)
            en_5 = LeakyReLU(alpha=0.2)(en_5)
            # -------------------------------
            # DECODER
            # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
            # 1 layer block = Conv - Upsample - BN - DO - Relu
            # also adds skip connections (merge). Takes input from previous layer matching encoder layer
            # -------------------------------
            # 1 decoder CD512 (decodes en_8)
            # # 3 decoder CD1024 (decodes en_6)
            # de_3 = UpSampling1D(size=stride)(en_5)
            # de_3 = Convolution1D(nb_filter=1024, filter_length=kernel_size, border_mode='same')(de_3)
            # de_3 = BatchNormalization(name='gen_de_bn_3')(de_3)
            # de_3 = Dropout(p=0.5)(de_3)
            # # de_3 = concatenate([de_3, en_5], axis=2)
            # # de_3 = Activation('relu')(de_3)
            # #
            # 4 decoder CD1024 (decodes en_5)
            de_4 = UpSampling1D(size=stride)(en_5)
            de_4 = Convolution1D(nb_filter=512, filter_length=kernel_size, border_mode='same')(de_4)
            de_4 = BatchNormalization(name='gen_de_bn_4')(de_4)
            de_4 = Dropout(p=0.5)(de_4)
            de_4 = concatenate([de_4, en_4], axis=2)
            de_4 = Activation('relu')(de_4)

            # 5 decoder CD1024 (decodes en_4)
            de_5 = UpSampling1D(size=stride)(de_4)
            de_5 = Convolution1D(nb_filter=256, filter_length=kernel_size, border_mode='same')(de_5)
            de_5 = BatchNormalization(name='gen_de_bn_5')(de_5)
            de_5 = Dropout(p=0.5)(de_5)
            # de_5 = concatenate([de_5, en_3], axis=2)
            de_5 = Activation('relu')(de_5)

            # 6 decoder C512 (decodes en_3)
            de_6 = UpSampling1D(size=stride)(de_5)
            de_6 = Convolution1D(nb_filter=128, filter_length=kernel_size, border_mode='same')(de_6)
            de_6 = BatchNormalization(name='gen_de_bn_6')(de_6)
            de_6 = Dropout(p=0.5)(de_6)
            de_6 = concatenate([de_6, en_2], axis=2)
            de_6 = Activation('relu')(de_6)

            # 7 decoder CD256 (decodes en_2)
            de_7 = UpSampling1D(size=stride)(de_6)
            de_7 = Convolution1D(nb_filter=256, filter_length=kernel_size, border_mode='same')(de_7)
            de_7 = BatchNormalization(name='gen_de_bn_7')(de_7)
            de_7 = Dropout(p=0.5)(de_7)
            de_7 = concatenate([de_7, en_1], axis=2)
            de_7 = Activation('relu')(de_7)

            # After the last layer in the decoder, a convolution is applied
            # to map to the number of output channels (3 in general,
            # except in colorization, where it is 2), followed by a Tanh
            # function.
            de_8 = UpSampling1D(size=stride)(de_7)
            de_8 = Convolution1D(nb_filter=6, filter_length=kernel_size, border_mode='same')(de_8)
            de_8 = Activation('tanh')(de_8)

            basemodel = Model(input=[input_layer], output=[de_8], name='unet_generator')
            basemodel.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel.summary()

        elif M == 7:
            print("Unet w/dense")
            inputs = Input(shape=(max_len,6))
            #inputs = Input(input_size)
            strides=1
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling1D(pool_size=2,strides=2)(conv1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling1D(pool_size=2,strides=2)(conv2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling1D(pool_size=2,strides=2)(conv3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2,strides=2)(drop4)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)
            pool5 = MaxPooling1D(pool_size=2,strides=2)(drop5)
            conv7 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
            conv7 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            drop7 = Dropout(0.5)(conv7)
            dense8 = Dense(200,activation="relu")(Flatten()(drop7))
            dense8 = Dense(200,activation="relu")(dense8)
            dense8 = Dense(int(drop7.shape[1])*int(drop7.shape[2]),activation="relu")(dense8)
            dense8 = Reshape((int(drop7.shape[1]),int(drop7.shape[2])))(dense8)
            up10 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(dense8))
            merge10 = concatenate([drop5, up10], axis=2)
            conv10 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
            conv10 = Conv1D(1024, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
            up11 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv10))
            merge11 = concatenate([conv4, up11], axis=2)
            conv11 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
            conv11 = Conv1D(512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
            up12 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv11))
            merge12 = concatenate([conv3, up12], axis=2)
            conv12 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
            conv12 = Conv1D(256, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
            up13 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv12))
            merge13 = concatenate([conv2, up13], axis=2)
            conv13 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
            conv13 = Conv1D(128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
            up14 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv13))
            merge14 = concatenate([conv1, up14], axis=2)
            conv14 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
            conv14 = Conv1D(64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
            conv14 = Conv1D(6, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
            conv14 = Conv1D(6, 1, activation='tanh')(conv14)
            model = Model(input=inputs, output=conv14)
            model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel = model
            basemodel.summary()

        else:
            sys.stderr.write('Invalid option {0}\n'.format(option))
            sys.exit(1)

    # If ACFG data
    if dataset == 'acfg_plus':
        if M == 2:
            print("Unet extra filters")
            inputs = Input(shape=(max_len, 18))
            # inputs = Input(input_size)
            factor = 2
            strides = 1
            conv1 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(inputs)
            conv1 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling1D(pool_size=2, strides=2)(conv1)
            conv2 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool1)
            conv2 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling1D(pool_size=2, strides=2)(conv2)
            conv3 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool2)
            conv3 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling1D(pool_size=2, strides=2)(conv3)
            conv4 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool3)
            conv4 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling1D(pool_size=2, strides=2)(drop4)

            conv5 = Conv1D(1024*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(pool4)
            conv5 = Conv1D(1024*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(drop5))
            merge6 = concatenate([drop4, up6], axis=2)
            conv6 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge6)
            conv6 = Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv6)

            up7 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv6))
            merge7 = concatenate([conv3, up7], axis=2)
            conv7 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge7)
            conv7 = Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv7)

            up8 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv7))
            merge8 = concatenate([conv2, up8], axis=2)
            conv8 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge8)
            conv8 = Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv8)

            up9 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                         kernel_initializer='he_normal')(
                UpSampling1D(size=2)(conv8))
            merge9 = concatenate([conv1, up9], axis=2)
            conv9 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(merge9)
            conv9 = Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                           kernel_initializer='he_normal')(conv9)
            conv9 = Conv1D(18*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv1D(18, 1, activation='tanh')(conv9)

            model = Model(input=inputs, output=conv10)

            model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mse'])
            basemodel = model
            basemodel.summary()
        else:
            sys.stderr.write('Invalid option {0}\n'.format(option))
            sys.exit(1)

    sys.stdout.write("Using {0} GPUs\n".format(ngpus))

    if ngpus > 1:
        model = multi_gpu.make_parallel(basemodel,ngpus)
    else:
        model = basemodel

    callbacks_list = list()

    base = K.get_value( model.optimizer.lr )
    def schedule(epoch):
        return base / 10.0**(epoch//2)
    callbacks_list.append(LearningRateScheduler( schedule ))

    # Because we have "save best model" enabled, we don't need this anymore
#   early_stop = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta = 0.0001, patience = 3)
#   callbacks_list.append(early_stop)

    checkpoint = ModelCheckpoint(model_path, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    callbacks_list.append(checkpoint)

    num_epochs = 10 # maximum number of epochs

    model.fit_generator(
        data.generator('train',batch_size),
        steps_per_epoch=data.get_train_num()//batch_size,
        epochs=num_epochs,
        validation_data=data.generator('test',batch_size),
        callbacks=callbacks_list,
        validation_steps=int(math.ceil(data.get_test_num()/batch_size))
    )

if __name__ == '__main__':
    _main()
