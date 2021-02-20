# DeepReflect
This tool identifies functions within malware PE binaries which differ from benign PE binaries. That is, it detects malicious functions within malware binaries.

It clusters extracted malicious functionalities from malware samples so the analyst can incrementally label each cluster (reviewing a small set of samples per cluster), and automatically identify known functions they have seen in the past. This allows the analyst to focus on new emerging malware behaviors in their malware feed.

For technical details, please see the paper cited below.

**Overview**:
  - Input: unpacked malware PE binary
  - Middle: list of all basic blocks in binary along with their reconstruction error values
  - Output: choosing a threshold (based on average reconstruction error value per function), identifies regions of interest (RoI) (i.e., basic blocks above threshold), and clusters the averaged feature vectors of RoIs

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
    - BinaryNinja (used only to extract features and function information from binaries)
    - parallel (optional, but recommended)
  - Setup:
    ```
    $ git clone https://github.com/evandowning/deepreflect.git
    $ cd deepreflect/
    $ mkvirutalenv dr --python=python3
    (dr) $ pip install -r requirements.txt
    ```

## Usage
  - Obtain unpacked benign and malicious PE file datasets
    - Folder layout: `benign_unpacked/benign/<binary files>` and `malicious_unpacked/<family label>/<binary files>`
  - Extract binary features & data
    ```
    (dr) $ ./extract.sh benign_unpacked/
    (dr) $ ./extract.sh malicious_unpacked/
    ```
  - Train autoencoder:
    ```
    (dr) $ cd ./model/

    # Split & shuffle dataset
    (dr) $ python split.py benign_unpacked_bndb_raw_feature/ train.txt test.txt > split_stdout.txt
    (dr) $ for fn in 'train.txt' 'test.txt'; do shuf $fn > tmp.txt; mv tmp.txt $fn; done

    # Check that benign samples use all features:
    (dr) $ python feature_check.py train.txt
    (dr) $ python feature_check.py test.txt
    # Check that malicious samples use all features:
    (dr) $ find malicious_unpacked_bndb_raw_feature/ -type f > malicious.txt
    (dr) $ python feature_check.py malicious.txt

    # Get max values (for normalizing)
    (dr) $ python normalize.py  --train train.txt \
                                --test test.txt \
                                --output normalize.npy

    # Train model
    (dr) $ time python autoencoder.py   --train train.txt \
                                        --test test.txt \
                                        --normalize normalize.npy \
                                        --model dr.h5 > autoencoder_stdout.txt 2> autoencoder_stderr.txt
    ```
  - Determine desired threshold:
    ```
    ```
  - Cluster suspicious functions:
    - Extract reconstruction errors for each basic block:
      ```
      (dr) $ time python autoencoder_eval_all.py acfg_plus --acfg-feature /data/malicious_unipacker_bndb_acfg_plus_feature/ \
                                                           --model ./models/autoencoder_benign_unipacker_plus/m2_normalize_24_12.h5 \
                                                           --normalize True \
                                                           --output /data/malicious_unipacker_bndb_acfg_plus_feature_error/ 2> autoencoder_eval_all_stderr.txt
      ```
    - Extract RoI (basic blocks):
      ```
      (dr) $ time python autoencoder_roi.py acfg_plus --data /data/malicious_unipacker_bndb_acfg_plus_feature_error/ \
                                                      --bndb-func /data/malicious_unipacker_bndb_function/ \
                                                      --acfg /data/malicious_unipacker_bndb_acfg_plus_feature/ \
                                                      --output ./autoencoder_roi/ \
                                                      --bb --avg --thresh 7.293461392658043e-06 > ./autoencoder_roi/stdout.txt 2> ./autoencoder_roi/stderr.txt
      ```
    - Cluster functions containing RoI:
      ```
      (dr) $ time python pca_hdbscan.py --x autoencoder_roi/x_train.npy \
                                        --fn autoencoder_roi/train_fn.npy \
                                        --addr autoencoder_roi/train_addr.npy > pca_hdbscan_output.txt
      ```

## Grading
  - Every system will have FPs and FNs. Ours is no different. The following allows the user to identify FPs and FNs to grade this tool and continue improving it.

## FAQs
  - Why don't you release the binaries used to train and evaluate DeepReflect (other than ground-truth samples)?
    - We cannot release malware binaries because of our agreement with those who provided them to us.
      - If you're looking for malware binaries, you might consider the [SOREL dataset](https://github.com/sophos-ai/SOREL-20M)
    - We cannot release benign binaries because of copyright rules.
      - If you're looking for benign binaries, you might consider [crawling](https://github.com/evandowning/selenium-crawler) them on [CNET](https://download.cnet.com/windows/). Make sure to verify they're not malicious via [VirusTotal](https://www.virustotal.com/).
    - We do, however, release our extracted features so models can be trained from scratch.
