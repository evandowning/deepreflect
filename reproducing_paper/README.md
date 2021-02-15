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
# "ACFG" are features inspired by "ACFG" features. Since they are not exactly ACFG features, we call them "ABB" features.
# "ACFG Plus" are features used by DeepReflect. See paper for details.
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
    ```
    (dr) $ cd ./grader/
    (dr) $ time ./rbot_final_corrected.sh &> rbot_final_stdout_stderr.txt
    (dr) $ time ./pegasus_final_corrected.sh &> pegasus_final_stdout_stderr.txt
    (dr) $ time ./carbanak_final_corrected.sh &> carbanak_final_stdout_stderr.txt
    (dr) $ time ./combined_final.sh &> combined_final_stdout_stderr.txt
    ```
  - Identify ideal threshold to use from ground-truth samples
    - See `combined_final_stdout_stderr.txt`
  - Cluster diversity
    ```
    (dr) $ python cluster_contents.py --input pca_hdbscan_output.txt \
                                      --output cluster_contents.png
    ```
  - Function highlight percentage
    ```
    (dr) $ time python function_coverage.py --functions /data/malicious_unipacker_bndb_function/ \
                                            --fn autoencoder_roi/train_fn.npy \
                                            --addr autoencoder_roi/train_addr.npy > function_coverage_stdout.txt
    ```
  - Distribution of cluster sizes
    ```
    (dr) $ python cluster_distribution.py --input pca_hdbscan_output.txt \
                                          --output cluster_distribution.png
    ```

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
