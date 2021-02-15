#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def get_fpr(target_tpr,tpr,fpr):
    return fpr[np.where(tpr>=target_tpr)[0][0]]
def get_thr_at_tpr(target_tpr,tpr,thr):
    return thr[np.where(tpr >= target_tpr)[0][0]]
def get_thr_at_fpr(target_fpr,fpr,thr):
    return thr[np.where(fpr >= target_fpr)[0][0]]

def _main():
    if len(sys.argv) < 3:
        sys.stderr.write('usage: python separate.py data1.npz data2.npz ... label1 label2 ... name out.png\n')
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
        y = a['y']
        score = a['score']
        functions = a['addr']

        # Create ROC data
        fpr, tpr, thresholds = metrics.roc_curve(y, score)
        roc_auc = metrics.auc(fpr, tpr)

        label = labels[e]
        label = label.replace('_',' ')
        label += ' | AUC: {0}'.format(round(roc_auc,4))

        # Output FPR/TPR/Thresholds
        sys.stdout.write('FPR TRP Threshold - {0}\n'.format(label))
        for i in range(len(fpr)):
            sys.stdout.write('{0} {1} {2}\n'.format(fpr[i],tpr[i],thresholds[i]))

        # Graph ROC curve
        plt.plot(fpr,tpr,color='#{0}'.format(color[e]),linestyle='--',label=label)

        t_fpr = get_thr_at_fpr(0.2,fpr,thresholds)
        t_tpr = get_thr_at_tpr(0.8,tpr,thresholds)

        sys.stdout.write('\n')
        sys.stdout.write('Threshold at FPR 0.2: {0}\n'.format(t_fpr))
        sys.stdout.write('Threshold at TPR 0.8: {0}\n'.format(t_tpr))

        # Print function RoIs given this threshold
        for e2,s in enumerate(score):
            l = y[e2]

            # TP
            if (l == 1.0) and (s >= t_tpr):
                sys.stdout.write('TP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FP
            if (l == 0.0) and (s >= t_tpr):
                sys.stdout.write('FP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FN
            if (l == 1.0) and (s < t_tpr):
                sys.stdout.write('FN: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
        sys.stdout.write('=============\n')

        # For clustering threshold used in paper
        t_fpr = 7.293461392658043e-06
        sys.stdout.write('Using threshold of 7.293461392658043e-06 (from paper - for DR only)\n')

        # Print function RoIs given this threshold
        for e2,s in enumerate(score):
            l = y[e2]

            # TP
            if (l == 1.0) and (s >= t_fpr):
                sys.stdout.write('TP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FP
            if (l == 0.0) and (s >= t_fpr):
                sys.stdout.write('FP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FN
            if (l == 1.0) and (s < t_fpr):
                sys.stdout.write('FN: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
        sys.stdout.write('=============\n')

        # For clustering threshold simulation (FPR 5%)
        t_fpr = get_thr_at_fpr(0.05,fpr,thresholds)
        sys.stdout.write('Threshold at FPR 0.05: {0}\n'.format(t_fpr))

        # Print function RoIs given this threshold
        for e2,s in enumerate(score):
            l = y[e2]

            # TP
            if (l == 1.0) and (s >= t_fpr):
                sys.stdout.write('TP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FP
            if (l == 0.0) and (s >= t_fpr):
                sys.stdout.write('FP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FN
            if (l == 1.0) and (s < t_fpr):
                sys.stdout.write('FN: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
        sys.stdout.write('=============\n')

        # For clustering threshold simulation (FPR 6%)
        t_fpr = get_thr_at_fpr(0.06,fpr,thresholds)
        sys.stdout.write('Threshold at FPR 0.06: {0}\n'.format(t_fpr))

        # Print function RoIs given this threshold
        for e2,s in enumerate(score):
            l = y[e2]

            # TP
            if (l == 1.0) and (s >= t_fpr):
                sys.stdout.write('TP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FP
            if (l == 0.0) and (s >= t_fpr):
                sys.stdout.write('FP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FN
            if (l == 1.0) and (s < t_fpr):
                sys.stdout.write('FN: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
        sys.stdout.write('=============\n')

        # For clustering threshold simulation (FPR 4%)
        t_fpr = get_thr_at_fpr(0.04,fpr,thresholds)
        sys.stdout.write('Threshold at FPR 0.04: {0}\n'.format(t_fpr))

        # Print function RoIs given this threshold
        for e2,s in enumerate(score):
            l = y[e2]

            # TP
            if (l == 1.0) and (s >= t_fpr):
                sys.stdout.write('TP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FP
            if (l == 0.0) and (s >= t_fpr):
                sys.stdout.write('FP: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
            # FN
            if (l == 1.0) and (s < t_fpr):
                sys.stdout.write('FN: Function Address: {0}  Score: {1}  Label: {2}\n'.format(hex(functions[e2]),s,y[e2]))
        sys.stdout.write('=============\n')

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
