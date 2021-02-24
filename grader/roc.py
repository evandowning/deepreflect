#!/usr/bin/python3

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

sys.path.append('../')
from dr_feature import RoI

# Get FPR at TPR value
def get_fpr_at_tpr(target_tpr,tpr,fpr):
    return fpr[np.where(tpr >= target_tpr)[0][0]]
# Get TPR at FPR value
def get_tpr_at_fpr(target_fpr,tpr,fpr):
    return tpr[np.where(fpr >= target_fpr)[0][0]]
# Get threshold at TPR value
def get_thr_at_tpr(target_tpr,tpr,thr):
    return thr[np.where(tpr >= target_tpr)[0][0]]
# Get threshold at FPR value
def get_thr_at_fpr(target_fpr,fpr,thr):
    return thr[np.where(fpr >= target_fpr)[0][0]]

# Print TPs/FPs/FNs for threshold
def print_details(y,score,addr,thr,f_name):
    tp = list()
    fp = list()
    fn = list()
    tn = list()

    # Tally TPs, FPs, and FNs
    for e,s in enumerate(score):
        l = y[e]

        # TP
        if (l == 1.0) and (s >= thr):
            string = 'TP: Function Address: {0}  Score: {1}  Name: {2}\n'.format(hex(addr[e]),s,f_name[addr[e]])
            tp.append(string)
        # FP
        if (l == 0.0) and (s >= thr):
            string = 'FP: Function Address: {0}  Score: {1}\n'.format(hex(addr[e]),s)
            fp.append(string)
        # FN
        if (l == 1.0) and (s < thr):
            string = 'FN: Function Address: {0}  Score: {1}  Name: {2}\n'.format(hex(addr[e]),s,f_name[addr[e]])
            fn.append(string)
        # TN
        if (l == 0.0) and (s < thr):
            string = 'TN: Function Address: {0}  Score: {1}\n'.format(hex(addr[e]),s)
            tn.append(string)

    tpr = len(tp)/(len(tp)+len(fn))
    fpr = len(fp)/(len(fp)+len(tn))

    # Output TPs, FPs, and FNs
    sys.stdout.write('Total: {0} | TP: {1} | FP: {2} | FN: {3} | TN: {4}\n'.format(len(tp)+len(fp)+len(fn)+len(tn),len(tp),len(fp),len(fn),len(tn)))
    sys.stdout.write('TPR: {0} | FPR: {1}\n'.format(tpr*100,fpr*100))
    for string in tp:
        sys.stdout.write(string)
    for string in fp:
        sys.stdout.write(string)
    for string in fn:
        sys.stdout.write(string)

# Read ground-truth (expected) highlights of malicious functionalities
# These are function addresses
def get_gt(gtFN):
    gt_f_addr = set()
    gt_f_name = dict()

    with open(gtFN,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            f_addr,f = line.split(' ')
            f_addr = int(f_addr,16)

            # If f begins with "mal-", then it's a malicious functionality
            # If not, continue
            if 'mal-' != f[:len('mal-')]:
                continue
            f = f[len('mal-'):]

            # If address is 0x0, then it means I couldn't find the malicious
            # activity in the binary. could be reviewed later.
            if f_addr == 0x0:
                sys.stderr.write('NOTE: {0} hasn\'t been found in binary yet\n'.format(f))
                continue

            gt_f_addr.add(f_addr)
            gt_f_name[f_addr] = f

    return gt_f_addr,gt_f_name

# Read highlights
# These are basic block addresses
def get_highlights(highlight, all_f_addr):
    h_f_addr = dict()

    for bb in highlight:
        # If padding, skip
        if bb == -1:
            continue

        if bb not in all_f_addr.keys():
            sys.stderr.write('Note: {0} not in a function.\n'.format(hex(bb)))
            continue

        f_addr = all_f_addr[bb]

        h_f_addr[bb] = f_addr

    return h_f_addr

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mse', help='mse produced by autoencoder', required=True)
    parser.add_argument('--feature', help='features file', required=True)
    parser.add_argument('--bndb-func', help='bndb function file', required=True)

    parser.add_argument('--threshold', help='optional threshold to evaluate', default=None, required=False)

    parser.add_argument('--annotation', help='function annotations', required=True)
    parser.add_argument('--roc',help='roc curve output name', required=True)

    args = parser.parse_args()

    # Store arguments
    mseFN = args.mse
    featureFN = args.feature
    funcFN = args.bndb_func
    optionalThreshold = args.threshold
    annotationFN = args.annotation
    output = args.roc

    if args.threshold is not None:
        optionalThreshold = float(args.threshold)

    sample = [[mseFN,funcFN,featureFN]]

    # MSE threshold (-1 means it will highlight everything)
    threshold = -1

    # Load dataset
    data = RoI(sample,threshold,None)

    # Load MSE values for each basic block
    addr,mse = data.parse(mseFN,featureFN,threshold)

    # Get mapping between basic blocks and the functions they belong to
    bb_map,_ = data.get_mapping(funcFN)

    # Read in annotations
    gt_f_addr,gt_f_name = get_gt(annotationFN)

    # Aggregrated MSE values for each function
    mse_func = dict()

    # Calculate average MSE score per function
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
        mse_func[f_addr].append(m)

    # Ground-truth labels
    roc_y = list()
    # MSE for each function
    roc_score = list()
    # Addresses for each function
    roc_addr = list()

    # Generate datasets to pass to ROC
    for a,m in mse_func.items():
        if a in gt_f_addr:
            roc_y.append(1.0)
        else:
            roc_y.append(0.0)

        # Calculate average BB MSE values for this function
        roc_score.append(sum(m)/len(m))

        roc_addr.append(a)

    # Create ROC data
    fpr, tpr, thresholds = metrics.roc_curve(roc_y, roc_score)
    roc_auc = metrics.auc(fpr, tpr)

    # If user defines threshold to test at
    if optionalThreshold is not None:
        sys.stdout.write('Optional Threshold {0}\n'.format(optionalThreshold))
        print_details(roc_y,roc_score,roc_addr,optionalThreshold,gt_f_name)
        sys.stdout.write('\n')

    # Otherwise, print TPs/FPs/FNs for different thresholds
    else:
        # Output FPR/TPR/Thresholds
        sys.stdout.write('ROC Curve. AUC: {0}\n'.format(roc_auc))
        sys.stdout.write('FPR TRP Threshold\n')
        for i in range(len(fpr)):
            sys.stdout.write('{0} {1} {2}\n'.format(fpr[i],tpr[i],thresholds[i]))
        sys.stdout.write('\n')

        # Graph ROC curve
        plt.plot(fpr,tpr,'r--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Function ROC Curve. AUC = {0}'.format(round(roc_auc,4)))

        plt.savefig('{0}_func.png'.format(output))
        plt.clf()

        # Save y, score, and addr
        np.savez('{0}_func_data.npz'.format(output),
                 y=np.asarray(roc_y),
                 score=np.asarray(roc_score),
                 addr=np.asarray(roc_addr))

        thr = get_thr_at_tpr(0.8,tpr,thresholds)
        sys.stdout.write('Threshold - TPR at 80% (FPR {0}%): {1}\n'.format(get_fpr_at_tpr(0.8,tpr,fpr)*100,thr))
        print_details(roc_y,roc_score,roc_addr,thr,gt_f_name)
        sys.stdout.write('\n')

        thr = get_thr_at_fpr(0.05,fpr,thresholds)
        sys.stdout.write('Threshold - FPR at 5% (TPR {0}%): {1}\n'.format(get_tpr_at_fpr(0.05,tpr,fpr)*100,thr))
        print_details(roc_y,roc_score,roc_addr,thr,gt_f_name)
        sys.stdout.write('\n')

if __name__ == '__main__':
    _main()
