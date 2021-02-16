import sys
import os
import argparse
import numpy as np

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--acfg', help='acfg features file', required=True)
    parser.add_argument('--shap-val', help='shap value file', required=True)
    parser.add_argument('--out', help='outputted numpy file', required=True)

    parser.add_argument('--normalize', type=bool, help='normalize shap values and avg. for all classes outputted', required=False, default=False)
    parser.add_argument('--absolute', type=bool, help='take absolute value of shap values', required=False, default=False)

    args = parser.parse_args()

    acfgFN = args.acfg
    shapFN = args.shap_val
    outFN = args.out

    normalize = False
    if args.normalize is not None:
        normalize = bool(args.normalize)
    absolute = False
    if args.absolute is not None:
        absolute = bool(args.absolute)

    acfg_bb = list()

    # Read ACFG features
    with open(acfgFN,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            acfg_bb.append(eval(line))

    # Create feature labels
    feat_label = ['number of instructions','number of transfer instructions','number of arithmetic instructions','number of call instructions','number of offspring','betweenness value']

    shap_vals = list()

    # Read shap values
    with open(shapFN,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            line = line.split(';')

            # Get shap values and class
            val = np.array([])
            for x in line[:-1]:
                tmp = list()
                for i in ' '.join(x.split()).split(' '):
                    i = i.strip('[]')
                    if i != '':
                        # If nan, set it to 0.0 (betweenness issues)
                        if i == 'nan':
                            i = 0.0

                        # If getting absolute shap value
                        if absolute is True:
                            tmp.append(abs(float(i)))
                        # Else get unaltered shap value
                        else:
                            tmp.append(float(i))

                if len(val) == 0:
                    val = np.asarray([tmp])
                else:
                    val = np.append(val,[tmp],axis=0)
            l = line[-1]

            # Append shap value and class to final array
            shap_vals.append((val,l))

    # If normalizing these values (each acfg feature value across each shap class)
    if normalize is True:
        tmp = np.array([])
        tmp_labels = list()

        rows = len(shap_vals[0][0])

        # Put all shap values into numpy array
        for val,l in shap_vals:
            if len(tmp) == 0:
                tmp = val
            else:
                tmp = np.vstack((tmp,val))

            tmp_labels.append(l)

        # Normalize numpy array
        tmp = tmp/np.linalg.norm(tmp, ord=np.inf, axis=0, keepdims=True)

        # Reconstruct shap list
        shap_vals = list()
        t = np.array([])

        for e,r in enumerate(tmp):
            # Add row
            if (e != 0) and (e % rows == 0):
                l = tmp_labels[int(e/rows)-1]
                shap_vals.append((t,l))

                t = np.array([])

            # Construct row
            if len(t) == 0:
                t = r
            else:
                t = np.vstack((t,r))

        e += 1
        l = tmp_labels[int(e/rows)-1]
        shap_vals.append((t,l))

    # For each shap value
    # NOTE : just get most popular class's shap values
    for val,l in shap_vals:
        # Get maximum feature's value per basic block
        val = np.max(val,axis=1)
        np.save(outFN,val)
        break

if __name__ == '__main__':
    _main()
