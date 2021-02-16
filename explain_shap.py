import sys
import os
import argparse

import numpy as np

import time

from keras.models import load_model
from keras.preprocessing import image

import matplotlib.pyplot as plt

import shap

import pefile

from acfg import ACFG
from acfg_plus import ACFG_plus

# Implementation based on https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='dataset types help', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('acfg', help='basic block features of malware dataset')
    sp.set_defaults(cmd='acfg')
    sp.add_argument('--train', help='training set files', required=True)
    sp.add_argument('--test', help='testing set files', required=True)
    sp.add_argument('--valid', help='validation set files', required=True)
    sp.add_argument('--data', help='data set to explain [train, test, valid]', required=True)
    sp.add_argument('--model', help='model path', required=True)
    sp.add_argument('--map', help='class map path', required=True)
    sp.add_argument('--output', help='output path', required=True)
    sp.add_argument('--shuffle-bb', help='shuffle basic block ordering', required=False, default=False)
    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)
    sp.add_argument('--joint', help='joint classifer (adding benign class)', required=False, default=False)

    sp = subparsers.add_parser('acfg-plus', help='basic block features of malware dataset')
    sp.set_defaults(cmd='acfg_plus')
    sp.add_argument('--train', help='training set files', required=True)
    sp.add_argument('--test', help='testing set files', required=True)
    sp.add_argument('--valid', help='validation set files', required=True)
    sp.add_argument('--data', help='data set to explain [train, test, valid]', required=True)
    sp.add_argument('--model', help='model path', required=True)
    sp.add_argument('--map', help='class map path', required=True)
    sp.add_argument('--output', help='output path', required=True)
    sp.add_argument('--shuffle-bb', help='shuffle basic block ordering', required=False, default=False)
    sp.add_argument('--max-bb', help='max number of basic blocks to consider', required=False, default=20000)
    sp.add_argument('--joint', help='joint classifer (adding benign class)', required=False, default=False)
    sp.add_argument('--normalize', help='normalize data', required=False, default=False)

    args = parser.parse_args()

    # Store arguments
    dataset = args.cmd
    trainset = args.train
    testset = args.test
    validset = args.valid
    dataChoice = args.data
    model_path = args.model
    map_path = args.map
    output_path = args.output

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    sys.stdout.write('Joint classifier: {0}\n'.format(bool(args.joint)))

    max_len = None

    # If ACFG, change max_len. Based on results of ranked_number_of_basic_blocks.txt
    # Based on avg-ish number of basic blocks extracted from binaries
    if dataset == 'acfg':
        max_len = int(args.max_bb)
    elif dataset == 'acfg_plus':
        max_len = int(args.max_bb)

    # Load trained model
    model = load_model(model_path)

    # Convert image to input our model is expecting
    def convert_file(dataset,files):
        # Numpy array for predictions
        rv = np.array([])

        for fn,label in files:
            if dataset == 'acfg':
                # Only look at first max_len of data (and pad with empty feature vector)
                b = np.array([[0,0,0,0,0,0]]*max_len, dtype=np.float)
                #bytez = [eval(line)[1:] for line in open(fn,'r').read().split('\n')[:max_len] if len(line) > 0]
                bytez = [eval(line.replace('nan','0.0'))[1:] for line in open(fn,'r').read().split('\n')[:max_len] if len(line) > 0]
                b[:len(bytez)] = bytez

                if len(rv) == 0:
                    rv = [b]
                else:
                    rv = np.append(rv,[b],axis=0)

            elif dataset == 'acfg_plus':
                # Only look at first max_len of data (and pad with empty feature vector)
                b = np.array([[0]*18]*max_len, dtype=np.float)
                bytez = np.load(fn)

                # If nothing was loaded, ignore this sample
                if len(bytez) == 0:
                    sys.stderr.write('Error. Sample {0} has no data.\n'.format(fn))
                    continue

                bytez = bytez[:max_len]
                # First element is the entry point, so we should ignore this
                bytez = bytez[:,1:]
                b[:len(bytez)] = bytez

                # NOTE: normalize data
                b = b / data.maximum_val

                if len(rv) == 0:
                    rv = [b]
                else:
                    rv = np.append(rv,[b],axis=0)

        return rv

    # Import data
    if dataset == 'acfg':
        data = ACFG(trainset,testset,validset,max_len,map_path,bool(args.shuffle_bb),bool(args.joint))
    elif dataset == 'acfg_plus':
        data = ACFG_plus(trainset,testset,validset,max_len,map_path,bool(args.shuffle_bb),bool(args.joint),bool(args.normalize))

    # Create SHAP explainer based on background samples from each class
    # from the training set
#NOTE: can do either balanced or non-balanced (balanced takes longer)
#   X = convert_file(dataset,data.explain_generator_shap_balanced('train',2))
    X = convert_file(dataset,data.explain_generator_shap('train',10))

    start = time.time()

    explain = shap.GradientExplainer(model,np.asarray(X))

    sys.stdout.write('Initializing explainer took {0} seconds\n'.format(time.time()-start))

    # Explain images
    max_num = 0
    if dataChoice == 'train':
        max_num = data.get_train_num()
    elif dataChoice == 'test':
        max_num = data.get_test_num()
    elif dataChoice == 'valid':
        max_num = data.get_valid_num()
    elif dataChoice == 'all':
        max_num = data.get_train_num() + data.get_test_num()

    for e,t in enumerate(data.explain_generator_shap(dataChoice,max_num)):

        fn,l = t

        outFN = os.path.join(output_path,fn.split('/')[-1])

        if dataset == 'acfg':
            outFN_shap_val = outFN + '_shap_val.txt'
        elif dataset == 'acfg_plus':
            outFN_shap_val = outFN + '_shap_val.txt'

        # If shap value file already exists, don't overwrite it
#       if os.path.exists(outFN_shap_val):
#           continue

        if dataset == 'acfg':
            x = np.asarray(convert_file(dataset,[(fn,l)]))
        elif dataset == 'acfg_plus':
            x = np.asarray(convert_file(dataset,[(fn,l)]))
        else:
            x = np.array([convert_file(dataset,[(fn,l)])])

        start = time.time()

        #shap_values,indexes = explain.shap_values(x,ranked_outputs=data.get_class_count())
        shap_values,indexes = explain.shap_values(x,ranked_outputs=5)

        sys.stdout.write('Explaining {0}/{1} took {2} seconds\n'.format(e+1,max_num,time.time()-start))

        # If ACFG
        if (dataset == 'acfg') or (dataset == 'acfg_plus'):
            np.set_printoptions(linewidth=np.inf)

            # Output explanations
            with open(outFN_shap_val,'w') as fw:
                for e2,s in enumerate(shap_values):
                    # Print shap values
                    for v in s[0]:
                        fw.write('{0};'.format(v))

                    # Print class of explanation
                    fw.write('{0}'.format(indexes[0][e2]))
                    fw.write('\n')

        else:
            # Create dimensions for image
            h = 1024
            w = int(max_len / h)

            # From: https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
            # Convert image back to 3 channel
            img = np.stack((x,)*3, axis=-1)
            img = np.array([img])
            img = np.reshape(img,(1,h,w,3))

            # Reshape shap_values to match img shape (because now we've reshaped
            # our input to a proper image format)
            shap_values_img = list()
            for e2,s in enumerate(shap_values):
                if indexes[0][e2] == l:
                    s = np.stack((s,)*3, axis=-1)
                    shap_values_img.append(np.reshape(s,(1,h,w,3)))

            # Create labels for each image
            index_names = np.array([])
            for i in indexes[0]:
                if i == l:
                    name = list(data.label.keys())[list(data.label.values()).index(i)]
                    index_names = np.append(index_names,name)
            index_names = np.array([index_names])

            # Plot the explanations
            shap.image_plot(shap_values_img, img, index_names, fn=outFN_shap_img)

            # Output original image of binary
            img = img.astype(np.uint8)
            plt.imsave(outFN,img[0])

            # Output explanations
            with open(outFN_shap_val,'w') as fw:
                for e2,s in enumerate(shap_values):
                    for v in s[0]:
                        fw.write('{0},'.format(v))
                    fw.write('{0}'.format(indexes[0][e2]))
                    fw.write('\n')

    sys.stdout.write('\n')

if __name__ == '__main__':
    _main()
