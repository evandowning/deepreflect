#!usr/bin/python3

import sys
import os
import argparse

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', help='number of people', type=int, required=True)
    parser.add_argument('--num', help='number of samples per person', type=int, required=True)
    parser.add_argument('--input', help='input file', required=True)
    parser.add_argument('--output', help='output file', required=True)

    args = parser.parse_args()

    # Store arguments
    split = int(args.split)
    num = int(args.num)
    inFN = args.input
    outFN = args.output

    sample = dict()

    flag = False

    # Read in cluster info
    with open(inFN,'r') as fr:
        for line in fr:
            line = line.strip('\n')

            # Get to part we care about
            if flag is False:
                if line != 'Cluster contents: filename address id probability':
                    continue
                else:
                    flag = True
                    continue

            line = line.split(' ')

            # Get cluster id, family, and sample
            family = line[0].split('/')[-2]
            h = line[0].split('/')[-1]
            addr = int(line[1],16)
            cid = int(line[2])
            prob = float(line[3])

            # If this was a noise cluster highlight, ignore it
            if cid == -1:
                continue

            if family not in sample:
                sample[family] = dict()
            if addr not in sample[family]:
                sample[family][addr] = list()

            # Store information
            sample[family][addr].append((h,cid,prob))

    # Divide work
    person = dict()
    for i in range(split):
        person[i] = list()

    # Get 'num' largest families based on number of highlighted addr
    fam_set = sorted(sample.keys(),key=lambda x:len(sample[x].keys()),reverse=True)[:num]

    # For each family
    for f in fam_set:

        # Get largest addresses based on number of highlights
        addr_set = sorted(sample[f].keys(),key=lambda x:len(sample[f][x]),reverse=True)

        e = 0

        # For each address
        for a in addr_set:
            entry = sample[f][a][0]

            # If confidence in this function belonging to its cluster is < 0.9,
            # search for a different function
            if entry[2] < 0.9:
                continue

            person[e].append((f,a,entry))
            e += 1

            if e == split:
                break

    # Output divided work
    with open(outFN,'w') as fw:
        fw.write('Person #\n')
        fw.write('    Family Address Sample ClusterID Probability\n')
        fw.write('\n')

        for k,v in person.items():
            fw.write('Person {0}\n'.format(k))

            for f,a,e in v:
                fw.write('    {0: <10} {1: <10} {2} {3: <7} {4}\n'.format(f,hex(a),e[0],e[1],e[2]))
            fw.write('\n')

if __name__ == '__main__':
    _main()
