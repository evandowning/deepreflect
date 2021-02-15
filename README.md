# DeepReflect
This tool identifies functions within malware PE binaries which differ from benign PE binaries. That is, it detects malicious functions within malware binaries.

Additionally, it clusters extracted malicious functionalities from malware samples so the analyst can incrementally label each cluster (reviewing a small set of samples per cluster), and automatically identify known functions they have seen in the past. This allows the analyst to focus on new emerging malware behaviors in their malware feed.

For technical details, please see the paper cited below.

**Overview**:
  - Input: unpacked malware PE binary
  - Output: list of all functions in binary along with their reconstruction error values

**Usage**: Using ground-truth malware binaries, choose an error value threshold which gives the analyst their desired results (tune to favor increasing TPR or decreasing FPR).

## Citation
  - ```
    things and stuff
    ```
  - Paper: link to paper
  - [Reproducing paper experiments](reproducing_paper/README.md)

## Coming soon
  - Dockerfile
  - BinaryNinja plugin

## Setup
  - Requirements:
    - Tested on Debian 10 (Buster)
    - Python 3 (tested with Python 3.7.3) and pip
    - virtualenvwrapper (optional, but recommended)
    - BinaryNinja (used only to extract features from binaries)
  - Setup:
    ```
    $ git clone https://github.com/evandowning/deepreflect.git
    $ cd deepreflect/
    $ mkvirutalenv dr --python=python3
    (dr) $ pip install -r requirements.txt
    ```

## Usage
  - Obtain unpacked benign and malicious PE file datasets
    - I put benign unpacked binaries in `/data/benign_unipacker/` and malicious unpacked binaries in `/data/malicious_unipacker/` because I unpacked them via unipacker.
  - Extract BinaryNinja DB file
    ```
    (dr) $ find /data/benign_unipacker -type f > samples_benign_unipacker.txt
    (dr) $ ./write_commands_binja.sh samples_benign_unipacker.txt /data/benign_unipacker_bndb/ > commands_benign_unipacker_bndb.txt
    (dr) $ time parallel --memfree 2G --retries 10 -a commands_benign_unipacker_bndb.txt 2> error.txt > output.txt
    ```
  - Extracting features:
    - ```
      (dr) $ find /data/benign_unipacker_bndb -type f > samples_benign_unipacker_bndb.txt
      (dr) $ ./write_commands_acfg_plus_binja.sh samples_benign_unipacker_bndb.txt benign_unipacker_bndb_acfg_plus/ > commands_benign_unipacker_bndb_acfg_plus.txt
      (dr) $ time parallel --memfree 2G --retries 10 -a commands_benign_unipacker_bndb_acfg_plus.txt 2> error.txt > output.txt

      (dr) $ find benign_unipacker_bndb_acfg_plus -type f > samples_benign_unipacker_bndb_acfg_plus.txt
      (dr) $ ./write_commands_acfg_plus_feature.sh samples_benign_unipacker_bndb_acfg_plus.txt benign_unipacker_bndb_acfg_plus_feature/ > commands_benign_unipacker_bndb_acfg_plus_feature.txt
      (dr) $ time parallel --memfree 2G --retries 10 -a commands_benign_unipacker_bndb_acfg_plus_feature.txt 2> error.txt > output.txt
      ```
  - Train autoencoder:
    ```
    (dr) $ python split_final.py /data/benign_unipacker_bndb_acfg_plus_feature/ train.txt test.txt
    (dr) $ for fn in 'train.txt' 'test.txt' 'valid.txt'; do shuf $fn > tmp.txt; mv tmp.txt $fn; done

    # Check that benign samples use all features:
    (dr) $ time python feature_check.py train.txt test.txt valid.txt
    # Do the same for malicious samples as well
    (dr) $ find /data/malicious_unipacker_bndb_acfg_plus_feature/ -type f > malicious.txt
    (dr) $ time python feature_check.py malicious.txt valid.txt valid.txt

    # Train model
    (dr) $ time python autoencoder.py --kernel 24 --strides 1 --option 2 acfg_plus --train train.txt --test test.txt --valid valid.txt --model ./models/m2_normalize_24_12.h5 --map benign_map.txt --normalize True > output.txt
    ```
  - Extract reconstruction errors:
    ```
    (dr) $ time python autoencoder_eval_all.py acfg_plus --acfg-feature /data/malicious_unipacker_bndb_acfg_plus_feature/ \
                                                         --model ./models/autoencoder_benign_unipacker_plus/m2_normalize_24_12.h5 \
                                                         --normalize True \
                                                         --output /data/malicious_unipacker_bndb_acfg_plus_feature_error/ 2> autoencoder_eval_all_stderr.txt
    ```
  - Extract function-related data from BinaryNinja DB files
    ```
    (dr) $ find /data/malicious_unipacker_bndb/ -type f > samples_bndb.txt
    (dr) $ time ./write_commands_get_function.sh samples_bndb.txt /data/malicious_unipacker_bndb_function/ > commands_get_function.txt
    (dr) $ time parallel --memfree 4G --retries 10 -a commands_get_function.txt 2> parallel_get_function_stderr.txt > parallel_get_function_stdout.txt

    ```
  - Extract regions of interest:
    ```
    (dr) $ time python autoencoder_roi.py acfg_plus --data /data/malicious_unipacker_bndb_acfg_plus_feature_error/ \
                                                    --bndb-func /data/malicious_unipacker_bndb_function/ \
                                                    --acfg /data/malicious_unipacker_bndb_acfg_plus_feature/ \
                                                    --output ./autoencoder_roi/ \
                                                    --bb --avg --thresh 7.293461392658043e-06 > ./autoencoder_roi/stdout.txt 2> ./autoencoder_roi/stderr.txt
    ```
  - Cluster regions of interest:
    ```
    (dr) $ time python pca_hdbscan.py --x autoencoder_roi/x_train.npy \
                                      --fn autoencoder_roi/train_fn.npy \
                                      --addr autoencoder_roi/train_addr.npy > pca_hdbscan_output.txt
    ```

## FAQs
  - Why don't you release the binaries used to train and evaluate DeepReflect (other than ground-truth samples)?
    - We cannot release malware binaries because of our contractual agreement with those who provided them to us.
    - We cannot release benign binaries because of copyright rules.
    - We do, however, release our extracted features so you can still train your own models from scratch.
