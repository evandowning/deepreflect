#!/usr/bin/python3

import sys
import os
import argparse
import math

import tensorflow as tf

sys.path.append('../')
from dr_feature import DR

def define_model(kernel_size,strides,max_len):
    inputs = tf.keras.Input(shape=(max_len, 18))

    factor = 2
    conv1 = tf.keras.layers.Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv1)
    conv2 = tf.keras.layers.Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv2)
    conv3 = tf.keras.layers.Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv3)
    conv4 = tf.keras.layers.Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(drop4)

    conv5 = tf.keras.layers.Conv1D(1024*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv1D(1024*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling1D(size=2)(drop5))
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=2)
    conv6 = tf.keras.layers.Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.Conv1D(512*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = tf.keras.layers.Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling1D(size=2)(conv6))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=2)
    conv7 = tf.keras.layers.Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.Conv1D(256*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = tf.keras.layers.Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling1D(size=2)(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=2)
    conv8 = tf.keras.layers.Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.Conv1D(128*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = tf.keras.layers.Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling1D(size=2)(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=2)
    conv9 = tf.keras.layers.Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.Conv1D(64*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.Conv1D(18*factor, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.Conv1D(18, 1, activation='tanh')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['mse'])

    # Print summary of model
    model.summary()

    return model

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', help='size of kernel', type=int, required=False, default=24)
    parser.add_argument('--strides', help='size of stride', type=int, required=False, default=1)
    parser.add_argument('--batch-size', help='batch size', type=int, required=False, default=10)
    parser.add_argument('--epochs', help='number of epochs to train model', type=int, required=False, default=10)

    parser.add_argument('--train', help='training set files', required=True)
    parser.add_argument('--test', help='testing set files', required=True)

    parser.add_argument('--normalize', help='normalize features', required=False, default=None)
    parser.add_argument('--max-bb', help='max number of basic blocks to consider', type=int, required=False, default=20000)

    parser.add_argument('--model', help='model path', required=True)

    args = parser.parse_args()

    # Store arguments
    kernel_size = int(args.kernel)
    strides = int(args.strides)
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    trainFN = args.train
    testFN = args.test
    normalizeFN = args.normalize
    max_len = int(args.max_bb)
    model_path = args.model

    # Import dataset
    data = DR(trainFN,testFN,max_len,normalizeFN)

    # Define model
    model = define_model(kernel_size,strides,max_len)

    # Create callbacks
    callbacks_list = list()

    # Add learning rate scheduler
    base = tf.keras.backend.get_value(model.optimizer.lr)
    def schedule(epoch):
        return base / 10.0**(epoch//2)
    callbacks_list.append(tf.keras.callbacks.LearningRateScheduler(schedule))

    # Checkpoint the model and save the best
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
    callbacks_list.append(checkpoint)

    # Train model
    model.fit(
        data.generator('train',batch_size),
        steps_per_epoch=data.get_train_num()//batch_size,
        epochs=num_epochs,
        validation_data=data.generator('test',batch_size),
        callbacks=callbacks_list,
        validation_steps=int(math.ceil(data.get_test_num()/batch_size))
    )

if __name__ == '__main__':
    _main()
