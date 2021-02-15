# Reproducing DeepReflect Experiments
This provides instructions on reproducing experiments and results from our paper.

## Checkout original code
```
$ git clone https://github.com/evandowning/deepreflect.git
$ cd deepreflect/
$ git checkout tags/v0.0.1
```

## Download dataset
```
$ wget <link>
$ <extract>
```

## Train baseline models
```
```

## Train DeepReflect and cluster RoIs
See [README.md](../README.md)

## Random cluster sample selection
```
(dr) $ python cluster_select.py --split 5 \
                                --num 10 \
                                --input pca_hdbscan_output.txt \
                                --output cluster_select_output.txt
```

## Identify singletons in clusters
```
(dr) $ time python find_singleton.py --input pca_hdbscan_output.txt > find_singleton_stdout.txt
```

## Emerging threats example
```
(dr) $ cd emerging-threat/
# I assume RoIs have already been extracted
(dr) $ cp -r ../autoencoder_roi/ .
(dr) $ time ./run.sh &> run_output.txt
```

## Sorting functions

## Graphs
  - Ground-truth ROC curves
  - Cluster diversity
  - Function highlight percentage

## Compile & evaluate obfuscated malware

## Compile & evaluate mimicry-like malware

## Code examples
  - Experiment 1:
  - Experiment 2:
  - Experiment 3:
  - Experiment 4:
  - Experiment 5:
  - Malware CFGs:
    - 
