#!/usr/bin/python3

import sys
import os
import argparse
import math

from collections import Counter

import keras.layers as layers
from keras.layers import Dense, Conv1D, Activation, GlobalMaxPooling1D, Input, Embedding, Multiply, MaxPooling1D, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import backend as backend
from keras import metrics

from sklearn.metrics import roc_curve,auc

from acfg import ACFG
from acfg_plus import ACFG_plus

import multi_gpu

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='number of GPUs', default=1)
    parser.add_argument('--shap', help='[boolean] - train for SHAP or not', type=bool, required=True)
    parser.add_argument('--kernel', help='size of kernel', type=int, required=False, default=500)
    parser.add_argument('--strides', help='size of stride', type=int, required=False, default=500)

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

    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)

    sp.add_argument('--default', help='use default model', required=False, default=False)
    sp.add_argument('--deeper', help='use a deeper model', required=False, default=False)
    sp.add_argument('--cifar10', help='use CIFAR-10 model', required=False, default=False)
    sp.add_argument('--vgg16', help='use VGG16 model', required=False, default=False)
    sp.add_argument('--vgg19', help='use VGG19 model', required=False, default=False)
    sp.add_argument('--resnet50', help='use ResNet50 model', required=False, default=False)
    sp.add_argument('--resnet50-half', help='use ResNet50 half model', required=False, default=False)
    sp.add_argument('--xception', help='use Xception model', required=False, default=False)
    sp.add_argument('--vgg19-half', help='use VGG19 half model', required=False, default=False)
    sp.add_argument('--vgg19-half-v2', help='use VGG19 half v2 model', required=False, default=False)
    sp.add_argument('--vgg19-half-v3', help='use VGG19 half v3 model', required=False, default=False)

    sp.add_argument('--vgg19-half-v4', help='use VGG19 half v4 model', required=False, default=False)
    sp.add_argument('--vgg19-half-v5', help='use VGG19 half v5 model', required=False, default=False)
    sp.add_argument('--vgg19-half-v6', help='use VGG19 half v6 model', required=False, default=False)

    sp.add_argument('--joint', help='joint classifer (adding benign class)', required=False, default=False)

    sp.add_argument('--weight-equal', help='Apply weights to classes based on size', required=False, default=False)
    sp.add_argument('--weight-50', help='Apply weights to classes (50/50 benign/malicious)', required=False, default=False)
    sp.add_argument('--weight-log', help='Apply log weights to classes', required=False, default=False)

    sp.add_argument('--normalize-acfg', help='normalizes ACFG values', required=False, default=False)


    sp = subparsers.add_parser('acfg-plus', help='basic block features of malware dataset')
    sp.set_defaults(cmd='acfg_plus')
    sp.add_argument('--train', help='training set files', required=True)
    sp.add_argument('--test', help='testing set files', required=True)
    sp.add_argument('--valid', help='validation set files', required=True)
    sp.add_argument('--model', help='model path', required=True)
    sp.add_argument('--map', help='class map path', required=True)
    sp.add_argument('--shuffle-bb', help='shuffle basic block ordering', required=False, default=False)

    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)

    sp.add_argument('--joint', help='joint classifer (adding benign class)', required=False, default=False)
    sp.add_argument('--normalize-acfg', help='normalizes ACFG values', required=False, default=False)

    args = parser.parse_args()

    # Store arguments
    dataset = args.cmd
    ngpus = int(args.gpus)
    shapFlag = bool(args.shap)
    kernel_size = int(args.kernel)
    strides = int(args.strides)
    trainset = args.train
    testset = args.test
    validset = args.valid
    model_path = args.model
    map_path = args.map

    max_len = None

    input_dim = 255+2       # between 0 and 255 + 1      char 256 is invalid so we'll use it as padding
    embedding_size = 8

    # If ACFG, change max_len. Based on results of ranked_number_of_basic_blocks.txt
    # Based on avg-ish number of basic blocks extracted from binaries
    weight_equal = False
    weight_50 = False
    weight_log = False
    normalize_acfg = False
    if dataset == 'acfg':
        max_len = int(args.max_bb)
        weight_equal = bool(args.weight_equal)
        weight_50 = bool(args.weight_50)
        weight_log = bool(args.weight_log)
        normalize_acfg = bool(args.normalize_acfg)
    elif dataset == 'acfg_plus':
        max_len = int(args.max_bb)
        normalize_acfg = bool(args.normalize_acfg)

    # Import data
    if dataset == 'acfg':
        data = ACFG(trainset,testset,validset,max_len,map_path,bool(args.shuffle_bb),bool(args.joint),normalize_acfg)
        batch_size = 10
    elif dataset == 'acfg_plus':
        data = ACFG_plus(trainset,testset,validset,max_len,map_path,bool(args.shuffle_bb),bool(args.joint),normalize_acfg,False)

    # Get number of classes
    class_count = data.get_class_count()

    # define model structure

    # If ACFG data
    if dataset == 'acfg':

        if bool(args.cifar10):
            sys.stdout.write('Using CIFAR-10 model.\n')

            batch_size = 12

            model = Sequential()
            model.add(Conv1D(32, kernel_size=kernel_size, strides=strides, padding='same',
                             input_shape=(max_len,6,)))
            model.add(Activation('relu'))
            model.add(Conv1D(32, kernel_size=kernel_size, strides=strides))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.25))

            model.add(Conv1D(64, kernel_size=kernel_size, strides=strides, padding='same'))
            model.add(Activation('relu'))
            model.add(Conv1D(64, kernel_size=kernel_size, strides=strides))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(class_count))
            model.add(Activation('softmax'))

            # initiate RMSprop optimizer
            opt = RMSprop(lr=0.0001, decay=1e-6)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg16):
            sys.stdout.write('Using VGG16 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size,strides=2,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size,strides=2,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(pool_size=2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(128, kernel_size=kernel_size,strides=2,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(128, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(pool_size=2, strides=2, name='block2_pool')(x)

            # Block 3
            x = Conv1D(256, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv1')(x)
            x = Conv1D(256, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv2')(x)
            x = Conv1D(256, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv3')(x)
            x = MaxPooling1D(pool_size=2, strides=2, name='block3_pool')(x)

            # Block 4
            x = Conv1D(512, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv1')(x)
            x = Conv1D(512, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv2')(x)
            x = Conv1D(512, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv3')(x)
            x = MaxPooling1D(pool_size=2, strides=2, name='block4_pool')(x)

            # Block 5
            x = Conv1D(512, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv1')(x)
            x = Conv1D(512, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv2')(x)
            x = Conv1D(512, kernel_size=kernel_size,strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv3')(x)
            x = MaxPooling1D(pool_size=2, strides=2, name='block5_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg16')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19):
            sys.stdout.write('Using VGG19 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size, strides=2,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size, strides=2,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(128, kernel_size=kernel_size, strides=2,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Block 3
            x = Conv1D(256, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv1')(x)
            x = Conv1D(256, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv2')(x)
            x = Conv1D(256, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv3')(x)
            x = Conv1D(256, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block3_conv4')(x)
            x = MaxPooling1D(2, strides=2, name='block3_pool')(x)

            # Block 4
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv1')(x)
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv2')(x)
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv3')(x)
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block4_conv4')(x)
            x = MaxPooling1D(2, strides=2, name='block4_pool')(x)

            # Block 5
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv1')(x)
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv2')(x)
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv3')(x)
            x = Conv1D(512, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block5_conv4')(x)
            x = MaxPooling1D(2, strides=2, name='block5_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.resnet50):
            sys.stdout.write('Using ResNet50 model.\n')

            def identity_block(input_tensor, kernel_size, filters, stage, block):
                """The identity block is the block that has no conv layer at shortcut.

                # Arguments
                    input_tensor: input tensor
                    kernel_size: default 3, the kernel size of
                        middle conv layer at main path
                    filters: list of integers, the filters of 3 conv layer at main path
                    stage: integer, current stage label, used for generating layer names
                    block: 'a','b'..., current block label, used for generating layer names

                # Returns
                    Output tensor for the block.
                """
                filters1, filters2, filters3 = filters
                bn_axis = 1
                conv_name_base = 'res' + str(stage) + block + '_branch'
                bn_name_base = 'bn' + str(stage) + block + '_branch'

                x = layers.Conv1D(filters1, 1,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2a')(input_tensor)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters2, kernel_size,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2b')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters3, 1,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2c')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

                x = layers.add([x, input_tensor])
                x = layers.Activation('relu')(x)
                return x


            def conv_block(input_tensor,
                           kernel_size,
                           filters,
                           stage,
                           block,
                           strides=2):
                """A block that has a conv layer at shortcut.

                # Arguments
                    input_tensor: input tensor
                    kernel_size: default 3, the kernel size of
                        middle conv layer at main path
                    filters: list of integers, the filters of 3 conv layer at main path
                    stage: integer, current stage label, used for generating layer names
                    block: 'a','b'..., current block label, used for generating layer names
                    strides: Strides for the first conv layer in the block.

                # Returns
                    Output tensor for the block.

                Note that from stage 3,
                the first conv layer at main path is with strides=(2, 2)
                And the shortcut should have strides=(2, 2) as well
                """
                filters1, filters2, filters3 = filters
                bn_axis = 1
                conv_name_base = 'res' + str(stage) + block + '_branch'
                bn_name_base = 'bn' + str(stage) + block + '_branch'

                x = layers.Conv1D(filters1, 1, strides=strides,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2a')(input_tensor)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters2, kernel_size, padding='same',
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2b')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters3, 1,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2c')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

                shortcut = layers.Conv1D(filters3, 1, strides=strides,
                                         kernel_initializer='he_normal',
                                         name=conv_name_base + '1')(input_tensor)
                shortcut = layers.BatchNormalization(
                    axis=bn_axis, name=bn_name_base + '1')(shortcut)

                x = layers.add([x, shortcut])
                x = layers.Activation('relu')(x)
                return x

            batch_size = 8

            img_input = layers.Input(shape=(max_len,6,))

            bn_axis = 1

            x = layers.ZeroPadding1D(padding=3, name='conv1_pad')(img_input)
            x = layers.Conv1D(64, kernel_size=kernel_size,
                              strides=strides,
                              padding='valid',
                              kernel_initializer='he_normal',
                              name='conv1')(x)
            x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
            x = layers.Activation('relu')(x)
            x = layers.ZeroPadding1D(padding=1, name='pool1_pad')(x)
            x = layers.MaxPooling1D(3, strides=strides)(x)

            x = conv_block(x, kernel_size, [64, 64, 256], stage=2, block='a', strides=1)
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

            x = conv_block(x, kernel_size, [128, 128, 512], stage=3, block='a')
            x = identity_block(x, kernel_size, [128, 128, 512], stage=3, block='b')
            x = identity_block(x, kernel_size, [128, 128, 512], stage=3, block='c')
            x = identity_block(x, kernel_size, [128, 128, 512], stage=3, block='d')

            x = conv_block(x, kernel_size, [256, 256, 1024], stage=4, block='a')
            x = identity_block(x, kernel_size, [256, 256, 1024], stage=4, block='b')
            x = identity_block(x, kernel_size, [256, 256, 1024], stage=4, block='c')
            x = identity_block(x, kernel_size, [256, 256, 1024], stage=4, block='d')
            x = identity_block(x, kernel_size, [256, 256, 1024], stage=4, block='e')
            x = identity_block(x, kernel_size, [256, 256, 1024], stage=4, block='f')

            x = conv_block(x, kernel_size, [512, 512, 2048], stage=5, block='a')
            x = identity_block(x, kernel_size, [512, 512, 2048], stage=5, block='b')
            x = identity_block(x, kernel_size, [512, 512, 2048], stage=5, block='c')

            #NOTE: SHAP can't handle Global Pooling at the moment due to an active bug
            #x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            #NOTE: Thus I added these in as a replacement
            x = MaxPooling1D(name='pool')(x)
            x = Flatten()(x)

            x = layers.Dense(class_count, activation='softmax', name='fc1000')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='resnet50')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.resnet50_half):
            sys.stdout.write('Using ResNet50 half model.\n')

            def identity_block(input_tensor, kernel_size, filters, stage, block):
                """The identity block is the block that has no conv layer at shortcut.

                # Arguments
                    input_tensor: input tensor
                    kernel_size: default 3, the kernel size of
                        middle conv layer at main path
                    filters: list of integers, the filters of 3 conv layer at main path
                    stage: integer, current stage label, used for generating layer names
                    block: 'a','b'..., current block label, used for generating layer names

                # Returns
                    Output tensor for the block.
                """
                filters1, filters2, filters3 = filters
                bn_axis = 1
                conv_name_base = 'res' + str(stage) + block + '_branch'
                bn_name_base = 'bn' + str(stage) + block + '_branch'

                x = layers.Conv1D(filters1, 1,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2a')(input_tensor)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters2, kernel_size,
                                  padding='same',
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2b')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters3, 1,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2c')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

                x = layers.add([x, input_tensor])
                x = layers.Activation('relu')(x)
                return x


            def conv_block(input_tensor,
                           kernel_size,
                           filters,
                           stage,
                           block,
                           strides=2):
                """A block that has a conv layer at shortcut.

                # Arguments
                    input_tensor: input tensor
                    kernel_size: default 3, the kernel size of
                        middle conv layer at main path
                    filters: list of integers, the filters of 3 conv layer at main path
                    stage: integer, current stage label, used for generating layer names
                    block: 'a','b'..., current block label, used for generating layer names
                    strides: Strides for the first conv layer in the block.

                # Returns
                    Output tensor for the block.

                Note that from stage 3,
                the first conv layer at main path is with strides=(2, 2)
                And the shortcut should have strides=(2, 2) as well
                """
                filters1, filters2, filters3 = filters
                bn_axis = 1
                conv_name_base = 'res' + str(stage) + block + '_branch'
                bn_name_base = 'bn' + str(stage) + block + '_branch'

                x = layers.Conv1D(filters1, 1, strides=strides,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2a')(input_tensor)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters2, kernel_size, padding='same',
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2b')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
                x = layers.Activation('relu')(x)

                x = layers.Conv1D(filters3, 1,
                                  kernel_initializer='he_normal',
                                  name=conv_name_base + '2c')(x)
                x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

                shortcut = layers.Conv1D(filters3, 1, strides=strides,
                                         kernel_initializer='he_normal',
                                         name=conv_name_base + '1')(input_tensor)
                shortcut = layers.BatchNormalization(
                    axis=bn_axis, name=bn_name_base + '1')(shortcut)

                x = layers.add([x, shortcut])
                x = layers.Activation('relu')(x)
                return x

            batch_size = 32

            img_input = layers.Input(shape=(max_len,6,))

            bn_axis = 1

            x = layers.ZeroPadding1D(padding=3, name='conv1_pad')(img_input)
            x = layers.Conv1D(64, kernel_size=kernel_size,
                              strides=strides,
                              padding='valid',
                              kernel_initializer='he_normal',
                              name='conv1')(x)
            x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
            x = layers.Activation('relu')(x)
            x = layers.ZeroPadding1D(padding=1, name='pool1_pad')(x)
            x = layers.MaxPooling1D(3, strides=strides)(x)

            x = conv_block(x, kernel_size, [64, 64, 256], stage=2, block='a', strides=1)
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

            #NOTE: SHAP can't handle Global Pooling at the moment due to an active bug
            #x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            #NOTE: Thus I added these in as a replacement
            x = MaxPooling1D(name='pool')(x)
            x = Flatten()(x)

            x = layers.Dense(class_count, activation='softmax', name='fc1000')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='resnet50_half')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.xception):
            sys.stdout.write('Using Xception model.\n')

            batch_size = 8

            img_input = layers.Input(shape=(max_len,6,))

            channel_axis = 1

            x = layers.Conv1D(32, kernel_size,
                              strides=strides,
                              use_bias=False,
                              name='block1_conv1')(img_input)
            x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
            x = layers.Activation('relu', name='block1_conv1_act')(x)
            x = layers.Conv1D(64, kernel_size=kernel_size, use_bias=False, name='block1_conv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
            x = layers.Activation('relu', name='block1_conv2_act')(x)

            residual = layers.Conv1D(128, kernel_size,
                                     strides=strides,
                                     padding='same',
                                     use_bias=False)(x)
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

            x = layers.SeparableConv1D(128, kernel_size=kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block2_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
            x = layers.Activation('relu', name='block2_sepconv2_act')(x)
            x = layers.SeparableConv1D(128, kernel_size=kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block2_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

            x = layers.MaxPooling1D(3,
                                    strides=2,
                                    padding='same',
                                    name='block2_pool')(x)
            x = layers.add([x, residual])

            residual = layers.Conv1D(256, kernel_size, strides=strides,
                                     padding='same', use_bias=False)(x)
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

            x = layers.Activation('relu', name='block3_sepconv1_act')(x)
            x = layers.SeparableConv1D(256, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block3_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
            x = layers.Activation('relu', name='block3_sepconv2_act')(x)
            x = layers.SeparableConv1D(256, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block3_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

            x = layers.MaxPooling1D(3, strides=2,
                                    padding='same',
                                    name='block3_pool')(x)
            x = layers.add([x, residual])

            residual = layers.Conv1D(728, kernel_size,
                                     strides=strides,
                                     padding='same',
                                     use_bias=False)(x)
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

            x = layers.Activation('relu', name='block4_sepconv1_act')(x)
            x = layers.SeparableConv1D(728, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block4_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
            x = layers.Activation('relu', name='block4_sepconv2_act')(x)
            x = layers.SeparableConv1D(728, kernel_size=kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block4_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

            x = layers.MaxPooling1D(3, strides=2,
                                    padding='same',
                                    name='block4_pool')(x)
            x = layers.add([x, residual])

            for i in range(8):
                residual = x
                prefix = 'block' + str(i + 5)

                x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
                x = layers.SeparableConv1D(728, kernel_size,
                                           padding='same',
                                           use_bias=False,
                                           name=prefix + '_sepconv1')(x)
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv1_bn')(x)
                x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
                x = layers.SeparableConv1D(728, kernel_size,
                                           padding='same',
                                           use_bias=False,
                                           name=prefix + '_sepconv2')(x)
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv2_bn')(x)
                x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
                x = layers.SeparableConv1D(728, kernel_size,
                                           padding='same',
                                           use_bias=False,
                                           name=prefix + '_sepconv3')(x)
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv3_bn')(x)

                x = layers.add([x, residual])

            residual = layers.Conv1D(1024, kernel_size, strides=strides,
                                     padding='same', use_bias=False)(x)
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

            x = layers.Activation('relu', name='block13_sepconv1_act')(x)
            x = layers.SeparableConv1D(728, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block13_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
            x = layers.Activation('relu', name='block13_sepconv2_act')(x)
            x = layers.SeparableConv1D(1024, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block13_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

            x = layers.MaxPooling1D(3,
                                    strides=2,
                                    padding='same',
                                    name='block13_pool')(x)
            x = layers.add([x, residual])

            x = layers.SeparableConv1D(1536, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block14_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
            x = layers.Activation('relu', name='block14_sepconv1_act')(x)

            x = layers.SeparableConv1D(2048, kernel_size,
                                       padding='same',
                                       use_bias=False,
                                       name='block14_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
            x = layers.Activation('relu', name='block14_sepconv2_act')(x)

            #NOTE: because SHAP can't handle global pooling
           #if include_top:
           #    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
           #    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
           #else:
           #    if pooling == 'avg':
           #        x = layers.GlobalAveragePooling2D()(x)
           #    elif pooling == 'max':
           #        x = layers.GlobalMaxPooling2D()(x)
            x = MaxPooling1D(name='pool')(x)
            x = Flatten()(x)
            x = layers.Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            model = Model(inputs, x, name='xception')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19_half):
            sys.stdout.write('Using VGG19 half model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19_half')

            opt = Adam(lr=0.001)

            # Let's train the model using the Adam optimizer
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19_half_v2):
            sys.stdout.write('Using VGG19 half v2 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19_half_v2')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19_half_v3):
            sys.stdout.write('Using VGG19 half v3 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19_half_v3')

            opt = SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-3)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19_half_v4):
            sys.stdout.write('Using VGG19 half v4 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(32, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(32, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19_half_v4')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19_half_v5):
            sys.stdout.write('Using VGG19 half v5 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(160, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = Conv1D(160, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19_half_v5')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.vgg19_half_v6):
            sys.stdout.write('Using VGG19 half v6 model.\n')

            batch_size = 32

            img_input = Input(shape=(max_len,6,))

            # Block 1
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(img_input)
            x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x)
            x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

            # Block 2
            x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x)
            x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(1024, activation='relu', name='fc1')(x)
            x = Dense(class_count, activation='softmax', name='predictions')(x)

            inputs = img_input

            # Create model.
            model = Model(inputs, x, name='vgg19_half_v6')

            opt = Adam(lr=0.001)

            # Let's train the model using RMSprop
            model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=[metrics.sparse_categorical_accuracy])

            basemodel = model

        elif bool(args.deeper):
            sys.stdout.write('Using custom deeper model.\n')

            inp = Input( shape=(max_len,6,))

            # version 1
#           # Based on CIFAR: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Flatten()(m)
#           m = Dense(512, activation='relu')(m)
#           m = Dropout(0.5)(m)

#           outp = Dense(class_count, activation='softmax')(m)
#           batch_size = 128


            # version 2
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Flatten()(m)
#           m = Dense(512, activation='relu')(m)
#           m = Dropout(0.5)(m)

#           outp = Dense(class_count, activation='softmax')(m)

#           batch_size = 128


            # version 3
            m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
            m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
            m = MaxPooling1D(pool_size=kernel_size)(m)
            m = Dropout(0.25)(m)

            m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
            m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
            m = MaxPooling1D(pool_size=kernel_size)(m)
            m = Dropout(0.25)(m)

            m = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
            m = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
            m = MaxPooling1D(pool_size=kernel_size)(m)
            m = Dropout(0.25)(m)

            m = Flatten()(m)
            m = Dense(512, activation='relu')(m)
            m = Dropout(0.5)(m)

            outp = Dense(class_count, activation='softmax')(m)

            batch_size = 128


            # version 4
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Flatten()(m)
#           m = Dense(512, activation='relu')(m)
#           m = Dropout(0.5)(m)

#           outp = Dense(class_count, activation='softmax')(m)

#           batch_size = 128

#           # version 5
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=32, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=64, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Conv1D( filters=512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(inp)
#           m = Conv1D( filters=512, kernel_size=kernel_size, strides=strides, activation='relu', padding='same' )(m)
#           m = MaxPooling1D(pool_size=kernel_size)(m)
#           m = Dropout(0.25)(m)

#           m = Flatten()(m)
#           m = Dense(512, activation='relu')(m)
#           m = Dropout(0.5)(m)

#           outp = Dense(class_count, activation='softmax')(m)

#           batch_size = 128

            basemodel = Model( inp, outp )

        elif bool(args.default):
            sys.stdout.write('Using default model.\n')

            batch_size = 32

            inp = Input( shape=(max_len,6,))

            filt = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, use_bias=True, activation='relu', padding='valid' )(inp)
            feat = MaxPooling1D(pool_size=kernel_size)(filt)
            feat = Flatten()(feat)

            dense = Dense(128, activation='relu')(feat)

            # NOTE: we use softmax, because sigmoid is limited to values between [0,1]
            outp = Dense(class_count, activation='softmax')(dense)

            basemodel = Model( inp, outp )

        else:
            sys.stderr.write('Error. Not a valid model option\n')
            sys.exit(2)

    elif dataset == 'acfg_plus':
        sys.stdout.write('Using VGG19 half model.\n')

        batch_size = 64

        img_input = Input(shape=(max_len,18,))

        # Block 1
        x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                   activation='relu',
                   padding='same',
                   name='block1_conv1')(img_input)
        x = Conv1D(64, kernel_size=kernel_size, strides=strides,
                   activation='relu',
                   padding='same',
                   name='block1_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

        # Block 2
        x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                   activation='relu',
                   padding='same',
                   name='block2_conv1')(x)
        x = Conv1D(128, kernel_size=kernel_size, strides=strides,
                   activation='relu',
                   padding='same',
                   name='block2_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(class_count, activation='softmax', name='predictions')(x)

        inputs = img_input

        # Create model.
        model = Model(inputs, x, name='vgg19_half')

        opt = Adam(lr=0.001)

        # Let's train the model using the Adam optimizer
        model.compile(loss='sparse_categorical_crossentropy',
          optimizer=opt,
          metrics=[metrics.sparse_categorical_accuracy])

        basemodel = model

    # If not ACFG data
    else:
        # Either load model or initialize it
        if os.path.exists(model_path):
            sys.stdout.write("restoring {0} from disk for continuation training...\n".format(model_path))
            from keras.models import load_model
            basemodel = load_model(model_path)
            _, maxlen, embedding_size = basemodel.layers[1].output_shape
            input_dim
        else:
            # define model structure
            if not shapFlag:
                inp = Input( shape=(max_len,))
                emb = Embedding( input_dim, embedding_size )( inp )
                filt = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, use_bias=True, activation='relu', padding='valid' )(emb)
                attn = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, use_bias=True, activation='sigmoid', padding='valid')(emb)
                gated = Multiply()([filt,attn])
                feat = GlobalMaxPooling1D()( gated )
                dense = Dense(128, activation='relu')(feat)

                # NOTE: we use softmax, because sigmoid is limited to values between [0,1]
                outp = Dense(class_count, activation='softmax')(dense)

                basemodel = Model( inp, outp )

            else:
                # NOTE: I removed the split in the model architecture and GlobalMaxPooling
                #       because of an ongoing bug in SHAP: https://github.com/slundberg/shap/issues/559

                # define model structure
                inp = Input( shape=(max_len,))
                emb = Embedding( input_dim, embedding_size )( inp )
                filt = Conv1D( filters=128, kernel_size=kernel_size, strides=strides, use_bias=True, activation='relu', padding='valid' )(emb)
                feat = MaxPooling1D(pool_size=kernel_size)(filt)
                feat = Flatten()(feat)

                dense = Dense(128, activation='relu')(feat)

                # NOTE: we use softmax, because sigmoid is limited to values between [0,1]
                outp = Dense(class_count, activation='softmax')(dense)

                basemodel = Model( inp, outp )

    basemodel.summary()

    sys.stdout.write("Using {0} GPUs\n".format(ngpus))

    if ngpus > 1:
        model = multi_gpu.make_parallel(basemodel,ngpus)
    else:
        model = basemodel

    # Train model
    if dataset == 'acfg' and bool(args.deeper):
        sys.stdout.write('Using RMSprop optimizer\n')
        opt = RMSprop(lr=0.0001, decay=1e-6)
        model.compile( loss='sparse_categorical_crossentropy', optimizer=opt, metrics=[metrics.sparse_categorical_accuracy] )
    elif dataset == 'acfg' and bool(args.default):
        sys.stdout.write('Using SGD optimizer\n')
        model.compile( loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-3), metrics=[metrics.sparse_categorical_accuracy] )
    elif dataset != 'acfg':
        sys.stdout.write('Using SGD optimizer\n')
        model.compile( loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-3), metrics=[metrics.sparse_categorical_accuracy] )


    callbacks_list = list()

    base = K.get_value( model.optimizer.lr )
    def schedule(epoch):
        return base / 10.0**(epoch//2)
    callbacks_list.append(LearningRateScheduler( schedule ))

    # Because we have "save best model" enabled, we don't need this anymore
#   early_stop = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta = 0.0001, patience = 3)
#   callbacks_list.append(early_stop)

    checkpoint = ModelCheckpoint(model_path, monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)

    num_epochs = 10 # maximum number of epochs

    # Get sample label counts
    class_weight = Counter()
    for fn, l in data.get_train():
        class_weight[data.get_label()[l]] += 1
    for fn, l in data.get_test():
        class_weight[data.get_label()[l]] += 1
    # TODO - need to do for validation set if needed in the future

    # Apply weights if necessary
    if weight_equal is True:
        total = sum(class_weight.values())
        class_weight = dict(class_weight)

        class_weight = {k: 1 - (v / total) for k,v in class_weight.items()}

    elif weight_50 is True:
        total_benign = class_weight[0]
        total_malicious = sum(class_weight.values()) - total_benign
        class_weight = dict(class_weight)

        class_weight = {k: v / total_malicious for k,v in class_weight.items() if k > 0}
        class_weight[0] = 1.0

    elif weight_log is True:
        class_weight = dict(class_weight)

        class_weight = {k: math.log(v) for k,v in class_weight.items()}

    # No weights are applied
    else:
        class_weight = dict(class_weight)
        class_weight = {k: 1.0 for k,v in class_weight.items()}

    # Train model
    model.fit_generator(
        data.generator('train',batch_size),
        steps_per_epoch=data.get_train_num()//batch_size,
        epochs=num_epochs,
        validation_data=data.generator('test',batch_size),
        callbacks=callbacks_list,
        validation_steps=int(math.ceil(data.get_test_num()/batch_size)),
        class_weight=class_weight
    )

if __name__ == '__main__':
    _main()
