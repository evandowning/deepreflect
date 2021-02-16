import sys
import os
import argparse
import json
import numpy as np

def get_data(capaFN, drFN):
    # Load CAPA results
    content = json.loads(open(capaFN,'r').read())

    capa_result = dict()

    for rule_name in content['rules']:

        # Skip rule if... (from: https://github.com/fireeye/capa/blob/8510f0465122ef11c1d259e47eadc0b0f6946f6c/capa/render/utils.py#L31)
        rule = content['rules'][rule_name]
        if rule["meta"].get("lib"):
            continue
        if rule["meta"].get("capa/subscope"):
            continue
        if rule["meta"].get("maec/analysis-conclusion"):
            continue
        if rule["meta"].get("maec/analysis-conclusion-ov"):
            continue
        if rule["meta"].get("maec/malware-category"):
            continue
        if rule["meta"].get("maec/malware-category-ov"):
            continue

        scope = content['rules'][rule_name]['meta']['scope']

        for addr in content['rules'][rule_name]['matches']:
            success = content['rules'][rule_name]['matches'][addr]['success']

            if success is True:
                capa_result[int(addr)] = '{0}: {1}'.format(rule_name,scope)

    # Load DeepReflect results
    deepreflect_result = np.load(drFN)
    dr_addr = deepreflect_result['addr']
    dr_y = deepreflect_result['y']


    addr = list()
    score = list()
    label = list()

    # For each DeepReflect address, determine if CAPA flagged it
    for e,a in enumerate(dr_addr):
        l = dr_y[e]

        # If address, in capa results, it means CAPA has flagged this function
        if a in capa_result.keys():
            s = 1.0
        else:
            s = 0.0

        addr.append(a)
        score.append(s)
        label.append(l)

    return addr,score,label

def _main():
    # Each argument comes in pairs [CAPA json, annotation]
    length = len(sys.argv) - 2
    if length % 2 != 0:
        sys.stderr.write('Error, arguments incorrect\n')
        sys.exit(2)

    # Last argument is output numpy file
    outFN = sys.argv[-1]

    addr = list()
    score = list()
    label = list()

    # For each file pair
    for i in range(1,length,2):
        capaFN = sys.argv[i]
        drFN = sys.argv[i+1]

        a,s,l = get_data(capaFN,drFN)
        addr.extend(a)
        score.extend(s)
        label.extend(l)

    # Output data file
    np.savez(outFN,
             y=np.asarray(label),
             score=np.asarray(score),
             addr=np.asarray(addr))

if __name__ == '__main__':
    _main()
