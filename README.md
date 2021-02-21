# DeepReflect
This tool identifies functions within malware PE binaries which differ from benign PE binaries. That is, it detects malicious functions within malware binaries.

It clusters extracted malicious functionalities from malware samples so the analyst can incrementally label each cluster (reviewing a small set of samples per cluster), and automatically identify known functions they have seen in the past. This allows the analyst to focus on new emerging malware behaviors in their malware feed.

For technical details, please see the paper cited below.

**Overview**:
  - Input: unpacked malware PE binary
  - Middle: list of all basic blocks in binary along with their reconstruction error values (MSE values)
  - Output: choosing a threshold (based on average MSE value per function), identifies regions of interest (RoI) (i.e., basic blocks above threshold), and clusters the averaged feature vectors of RoIs

**Usage**: Using ground-truth malware binaries, choose an MSE threshold which gives the analyst their desired results (tune to favor increasing TPR or decreasing FPR).

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
    (dr) $ cd ./autoencoder/

    # Split & shuffle benign dataset
    (dr) $ python split.py benign_unpacked_bndb_raw_feature/ train.txt test.txt > split_stdout.txt
    (dr) $ for fn in 'train.txt' 'test.txt'; do shuf $fn > tmp.txt; mv tmp.txt $fn; done

    # Check that benign samples use all features:
    (dr) $ python feature_check.py train.txt
    (dr) $ python feature_check.py test.txt
    # Check that malicious samples use all features:
    (dr) $ find malicious_unpacked_bndb_raw_feature/ -type f > malicious.txt
    (dr) $ python feature_check.py malicious.txt

    # Get max values (for normalizing)
    (dr) $ python normalize.py --train train.txt \
                               --test test.txt \
                               --output normalize.npy

    # Train model
    (dr) $ time python autoencoder.py --train train.txt \
                                      --test test.txt \
                                      --normalize normalize.npy \
                                      --model dr.h5 > autoencoder_stdout.txt 2> autoencoder_stderr.txt
    ```
  - Cluster suspicious functions:
    - Extract MSE values for each malware basic block:
      ```
      (dr) $ cd ./autoencoder/
      (dr) $ time python mse.py --feature malicious.txt \
                                --model dr.h5 \
                                --normalize normalize.npy \
                                --output malicious_unpacked_bndb_raw_feature_mse/ 2> mse_stderr.txt
      ```
    - Identify desired threshold. See [Grading](#grading).
    - Extract RoI (basic blocks):
      ```
      (dr) $ cd ./autoencoder/
      (dr) $ mkdir roi/
      (dr) $ time python roi.py --bndb-func malicious_unpacked_bndb_function/ \
                                --feature malicious_unpacked_bndb_raw_feature/ \
                                --mse malicious_unpacked_bndb_raw_feature_mse/ \
                                --output roi/ \
                                --bb --avg --thresh 7.293461392658043e-06 > roi/stdout.txt 2> roi/stderr.txt
      ```
    - Cluster functions containing RoI:
      ```
      (dr) $ cd ./cluster/
      (dr) $ time python pca_hdbscan.py --x ../autoencoder/roi/x.npy \
                                        --fn ../autoencoder/roi/fn.npy \
                                        --addr ../autoencoder/roi/addr.npy > pca_hdbscan_stdout.txt
      ```
    - Graph percentage of functions highlighted:
      ```
      (dr) $ cd ./cluster/
      (dr) $ time python function_coverage.py --functions malicious_unpacked_bndb_function/ \
                                              --fn ../autoencoder/roi/fn.npy \
                                              --addr ../autoencoder/roi/addr.npy \
                                              --output function_coverage.png > function_coverage_stdout.txt
      ```

## Grading
  - Graph ROC curves
    ```
    (dr) $ cd grader/
    (dr) $ ./roc.sh
    ```
    ![rbot](grader/roc_rbot.png)
    ![pegasus](grader/roc_pegasus.png)
    ![carbanak](grader/roc_carbanak.png)
    ![combined](grader/roc_combined.png)
  - Pick desired threshold
    ```
    $ grep 
    ```
  - Examine FPs & FNs due to threshold

## FAQs
  - Why don't you release the binaries used to train and evaluate DeepReflect (other than ground-truth samples)?
    - We cannot release malware binaries because of our agreement with those who provided them to us.
      - If you're looking for malware binaries, you might consider the [SOREL dataset](https://github.com/sophos-ai/SOREL-20M)
    - We cannot release benign binaries because of copyright rules.
      - If you're looking for benign binaries, you might consider [crawling](https://github.com/evandowning/selenium-crawler) them on [CNET](https://download.cnet.com/windows/). Make sure to verify they're not malicious via [VirusTotal](https://www.virustotal.com/).
    - We do, however, release our extracted features so models can be trained from scratch.
