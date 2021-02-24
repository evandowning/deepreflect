#!/usr/bin/python3

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

from roc import get_thr_at_tpr, get_thr_at_fpr
from roc import get_fpr_at_tpr, get_tpr_at_fpr
from roc import print_details

def _main():
    if len(sys.argv) < 3:
        sys.stderr.write('usage: python roc_multi.py data1.npz data2.npz ... label1 label2 ... name out.png\n')
        sys.exit(2)

    # Get output filename
    outFN = sys.argv[-1]

    # Colors from LibreOffice
    color = ['004586','ff420e','ffd320','579d1c','7e0021','83caff','314004','aecf00']

    files = list()
    labels = list()

    # Get data files and labels
    for fn in sys.argv[1:-2]:
        ext='.npz'
        if fn[-len(ext):] == ext:
            files.append(fn)
        else:
            labels.append(fn)

    title_name = sys.argv[-2]

    # For each file
    for e,fn in enumerate(files):
        a = np.load(fn)
        roc_y = a['y']
        roc_score = a['score']
        roc_addr = a['addr']

        # Create ROC data
        fpr, tpr, thresholds = metrics.roc_curve(roc_y, roc_score)
        roc_auc = metrics.auc(fpr, tpr)

        label = labels[e]
        label = label.replace('_',' ')
        label += ' | AUC: {0}'.format(round(roc_auc,4))

        # Output FPR/TPR/Thresholds
        sys.stdout.write('ROC Curve. AUC: {0}\n'.format(roc_auc))
        sys.stdout.write('FPR TRP Threshold - {0}\n'.format(label))
        for i in range(len(fpr)):
            sys.stdout.write('{0} {1} {2}\n'.format(fpr[i],tpr[i],thresholds[i]))
        sys.stdout.write('\n')

        # Graph ROC curve
        plt.plot(fpr,tpr,color='#{0}'.format(color[e]),linestyle='--',label=label)

        # Construct some dummy dictionary
        gt_f_name = dict()
        for a in roc_addr:
            gt_f_name[a] = ''

        # Print TPs/FPs/FNs for different thresholds
        thr = get_thr_at_tpr(0.8,tpr,thresholds)
        sys.stdout.write('Threshold - TPR at 80% (FPR {0}%): {1}\n'.format(get_fpr_at_tpr(0.8,tpr,fpr)*100,thr))
        print_details(roc_y,roc_score,roc_addr,thr,gt_f_name)
        sys.stdout.write('\n')

        thr = get_thr_at_fpr(0.05,fpr,thresholds)
        sys.stdout.write('Threshold - FPR at 5% (TPR {0}%): {1}\n'.format(get_tpr_at_fpr(0.05,tpr,fpr)*100,thr))
        print_details(roc_y,roc_score,roc_addr,thr,gt_f_name)
        sys.stdout.write('\n')

    # Put a line at TPR 80%
    plt.axhline(y=0.8,color='black',linestyle='-')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('{0} ROC Curves'.format(title_name))
    plt.legend(loc='lower right')

    plt.savefig(outFN)
    plt.clf()

if __name__ == '__main__':
    _main()
