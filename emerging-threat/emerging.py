#!usr/bin/python3

import sys
import os
import argparse
import time
from collections import Counter

def avg_cluster_fam(cluster,family):
    if len(cluster.keys()) == 0:
        return 0

    avg = 0

    for k,v in cluster.items():
        for f,addr,prob in v:
            fam = f.split('/')[-2]
            if fam == family:
                avg += 1

    avg /= len(cluster.keys())

    return avg

def avg_cluster(cluster):
    if len(cluster.keys()) == 0:
        return 0

    avg = 0

    for k,v in cluster.items():
        avg += len(v)

    avg /= len(cluster.keys())

    return avg

def parse_cluster(fn):
    cluster = dict()
    sample = dict()

    flag = False

    # Read in cluster info
    with open(fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')

            # Look for beginning of what we care about
            if flag is False:
                if line != 'Cluster contents: filename address id probability':
                    continue
                else:
                    flag = True
                    continue

            # End of what we care about
            if line == '':
                break

            f,addr,cid,prob = line.split(' ')

            # Skip noise clusters
            if cid == '-1':
                continue

            # Add cluster
            if cid not in cluster.keys():
                cluster[cid] = list()

            cluster[cid].append((f,addr,prob))

            # Add sample
            key = '{0}_{1}'.format(f,addr)
            sample[key] = cid

    return cluster,sample

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--minus', help='clustering stdout without family', required=True)
    parser.add_argument('--plus', help='clustering stdout with family', required=True)
    parser.add_argument('--family', help='family which is emerging', required=True)

    args = parser.parse_args()

    # Store arguments
    minusFN = args.minus
    plusFN = args.plus
    family = args.family

    sys.stdout.write('Extracting cluster contents (minus noise points)\n')

    # Parse cluster contents
    minusCluster,minusSample = parse_cluster(minusFN)
    plusCluster,plusSample = parse_cluster(plusFN)

    sys.stdout.write('Number of clusters before and after {0}: {1} -> {2}\n'.format(family,len(minusCluster.keys()),len(plusCluster.keys())))

    # Find emerging threat

    # Get cluster contents that family exists in (plus)
    plus = dict()
    for k,v in plusCluster.items():
        for f,addr,prob in v:
            fam = f.split('/')[-2]

            if fam == family:
                plus[k] = v[:]
                break

    sys.stdout.write('Number of clusters with {0} in them: {1}\n'.format(family, len(plus.keys())))

    # Get clusters with *only* family in them
    mixed = dict()
    exclusive = dict()
    for k,v in plus.items():
        for f,addr,prob in v:
            fam = f.split('/')[-2]

            if fam != family:
                break

        if fam != family:
            mixed[k] = v[:]
        else:
            exclusive[k] = v[:]

    sys.stdout.write('Number of clusters with only {0} samples in them and those with mixed contents: {1} and {2}\n'.format(family,len(exclusive.keys()),len(mixed.keys())))

    # Calculate various metrics of composition of clusters

    avg_exclusive = avg_cluster(exclusive)
    avg_mixed = avg_cluster(mixed)
    sys.stdout.write('Average size of those clusters ^: {0} and {1}\n'.format(avg_exclusive,avg_mixed))

    avg_exclusive = avg_cluster_fam(exclusive,family)
    avg_mixed = avg_cluster_fam(mixed,family)
    sys.stdout.write('Average number of {0} samples in them ^: {1} and {2}\n'.format(family,avg_exclusive,avg_mixed))
    sys.stdout.write('\n')

    # Determine if mixed clusters existed before (i.e., would have been labeled as a cluster singularly, before the existence of this family)
    tally = Counter()
    for k,v in mixed.items():
        c = set()

        # Get old clusters that these samples belong to
        for f,addr,prob in v:
            # Target family won't be in old clusters
            fam = f.split('/')[-2]
            if fam == family:
                continue

            key = '{0}_{1}'.format(f,addr)

            # Means that it's in a cluster *after inserting family*, but a noise point *before inserting faimly*
            if key not in minusSample.keys():
                tally[-1] += 1
                continue

            oldC = minusSample[key]
            c.add(oldC)

        tally[len(c)] += 1

    # Output interpretation of results
    sys.stdout.write('Number of unique Old clusters (KEY) that samples in New clusters (VALUE) reside in:\n')
    sys.stdout.write('-1: the number of samples that were clustered *after inserting {0}*, but were noise points *before inserting {0}*\n'.format(family))
    sys.stdout.write('0: means that there were new clusters which were made up entirely of old noise points\n')
    sys.stdout.write('1: the ideal (i.e., the new cluster contents precisely matched the old before and after adding in {0} samples)\n'.format(family))
    sys.stdout.write('Tally: Key Number-of-New-Clusters\n')

    for k,v in tally.most_common():
        sys.stdout.write('  {0} {1}\n'.format(k,v))

if __name__ == '__main__':
    _main()
