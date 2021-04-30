# Reproducing DeepReflect Experiments
This provides instructions on reproducing experiments and results from our paper.

## Checkout original code
```
$ git clone https://github.com/evandowning/deepreflect.git
$ cd deepreflect/
$ git checkout tags/v0.0.1
```

## Download & extract dataset
  - [DeepReflect Dataset](https://s3.console.aws.amazon.com/s3/buckets/deepreflect-usenix21-dataset)

## Train baseline models
```
# "ACFG" are features inspired by "ACFG" features. Since they are not exactly ACFG features, we call them "ABB" features in our paper, but for simplicity we call them "ACFG" features in our code. This will be corrected in a future version of our code.
# "ACFG Plus" are features used by DeepReflect. See paper for details.

# For autoencoders, see ../README.md

# For VGG19
$ time python model.py --shap True --kernel 4 --strides 1 acfg \
              --train ./models/malicious_plus_benign_joint/train.txt \
              --test ./models/malicious_plus_benign_joint/test.txt \
              --valid ./models/malicious_plus_benign_joint/valid.txt \
              --model ./models/malicious_plus_benign_joint/vgg19_half_joint.h5 \
              --map ./models/malicious_plus_benign_joint/final_map.txt \
              --vgg19-half True &> ./models/final_binaries_unipacker_bndb_acfg/vgg19_output.txt

# Get SHAP highlights
$ time python explain_shap.py acfg --train ./models/malicious_plus_benign_joint/train.txt \
              --test ./models/malicious_plus_benign_joint/test.txt \
              --valid ./models/malicious_plus_benign_joint/valid.txt \
              --data test \
              --model ./models/malicious_plus_benign_joint/vgg19_half_joint.h5 \
              --map ./models/malicious_plus_benign_joint/final_map.txt \
              --joint True \
              --output ./shap/ 2> error.txt
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

## Graphs
  - Ground-truth ROC curves
    ```
    (dr) $ time ./rbot.sh &> rbot_final_stdout_stderr.txt
    (dr) $ time ./pegasus.sh &> pegasus_final_stdout_stderr.txt
    (dr) $ time ./carbanak.sh &> carbanak_final_stdout_stderr.txt

    (dr) $ cd ./grader/
    (dr) $ time ./rbot.sh &> rbot_final_stdout_stderr.txt
    (dr) $ time ./pegasus.sh &> pegasus_final_stdout_stderr.txt
    (dr) $ time ./carbanak.sh &> carbanak_final_stdout_stderr.txt
    (dr) $ time ./combined.sh &> combined_final_stdout_stderr.txt
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

## Sorting functions
  - See `sorting/` folder in dataset

## Compile & evaluate obfuscated malware
  - See `malware-gt/` folder in dataset

## Compile & evaluate mimicry-like malware
  - See `malware-gt/` folder in dataset
