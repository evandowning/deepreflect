# DeepReflect
This tool identifies functions within malware PE binaries which differ from benign PE binaries. That is, it detects malicious functions within malware binaries.

It clusters extracted malicious functionalities from malware samples so the analyst can incrementally label each cluster (reviewing a small set of samples per cluster), and automatically identify known functions they have seen in the past. This allows the analyst to focus on new emerging malware behaviors in their malware feed.

For technical details, please see the paper cited below.

**Overview**:
  - Input: unpacked malware PE binary
  - Middle: list of all basic blocks in binary along with their reconstruction error values (MSE values)
  - Output: choosing a threshold (based on average MSE value per function), identifies regions of interest (RoI) (i.e., basic blocks with MSE values above threshold), and clusters the averaged feature vectors of RoIs

**Usage**: Using ground-truth malware binaries, choose an MSE threshold which gives the analyst their desired results (tune to favor increasing TPR or decreasing FPR).

--------------------------------------------------------------------------------

## Setup
  - Requirements:
    - Tested on Debian 10 (Buster)
    - Python 3 (tested with Python 3.7.3) and pip
    - BinaryNinja 2.3 (used to extract features and function information from binaries)
    - PostgreSQL 11.10 (to store results)
    - virtualenvwrapper (optional, but recommended)
    - parallel (optional, but recommended)
  - Setup:
    ```
    $ git clone https://github.com/evandowning/deepreflect.git
    $ cd deepreflect/
    $ mkvirtualenv dr --python=python3
    (dr) $ pip install -r requirements.txt
    ```

## [BinaryNinja Plugin](./binaryninja_plugin/deepreflect/)

## Docker Container
BinaryNinja setup:
  * Zip Linux version of `binaryninja/` folder.
    * `$ 7z a binaryninja.7z binaryninja/`
  * Copy `binaryninja.7z` into `binja_setup/`.
  * Copy `license.dat` into `binja_setup/`.

Build `dr` container:
```
$ docker build -t dr .
```

Run `dr` container:
```
$ docker run --rm dr --help
```

[Docker README](README_Docker.md)

--------------------------------------------------------------------------------

## Usage
  - Obtain unpacked benign and malicious PE file datasets
    - Benign folder layout:    `/data/benign_unpacked/benign/<binary_files>`
    - Malicious folder layout: `/data/malicious_unpacked/<family_label>/<binary_files>`
  - Extract binary features & data
    ```
    (dr) $ ./extract.sh /data/benign_unpacked/
    (dr) $ ./extract.sh /data/malicious_unpacked/
    ```
  - Train autoencoder:
    ```
    (dr) $ cd ./autoencoder/

    # Split & shuffle benign dataset
    (dr) $ python split.py /data/benign_unpacked_bndb_raw_feature/ train.txt test.txt > split_stdout.txt
    (dr) $ for fn in 'train.txt' 'test.txt'; do shuf $fn > tmp.txt; mv tmp.txt $fn; done

    # Check that benign samples use all features:
    (dr) $ python feature_check.py train.txt
    (dr) $ python feature_check.py test.txt
    # Check that malicious samples use all features:
    (dr) $ find /data/malicious_unpacked_bndb_raw_feature/ -type f > malicious.txt
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
                                --output /data/malicious_unpacked_bndb_raw_feature_mse/ 2> mse_stderr.txt
      ```
    - Identify desired threshold. See [Grading](#grading).
    - Extract RoI (basic blocks) and output average RoI feature vectors for each function:
      ```
      (dr) $ cd ./autoencoder/
      (dr) $ mkdir roi/
      (dr) $ time python roi.py --bndb-func /data/malicious_unpacked_bndb_function/ \
                                --feature /data/malicious_unpacked_bndb_raw_feature/ \
                                --mse /data/malicious_unpacked_bndb_raw_feature_mse/ \
                                --normalize normalize.npy \
                                --output roi/ \
                                --bb --avg --thresh 7.293461392658043e-06 > roi/stdout.txt 2> roi/stderr.txt

      # Extract MSE values for each highlighted function (avg RoI MSE value)
      (dr) $ time python mse_func.py --bndb-func /data/malicious_unpacked_bndb_function/ \
                                     --feature /data/malicious_unpacked_bndb_raw_feature/ \
                                     --roiFN roi/fn.npy \
                                     --roiAddr roi/addr.npy \
                                     --thresh 7.293461392658043e-06 \
                                     --output roi/mse_func.npy > /dev/null
      ```
    - Create & initialize database:
      ```
      $ psql --list
      $ createdb dr
      $ psql -d dr -f ./db/create.sql

      # To export/import database to another database
      $ pg_dump -O dr -f export.sql
      $ psql -d dr -f export.sql

      # To clear database (i.e., drop tables)
      $ psql -d dr -f ./db/drop.sql
      # To remove database
      $ dropdb dr
      ```
    - Modify `./cluster/cluster.cfg` and `./binaryninja/deepreflect/db.cfg` files
    - Cluster functions containing RoI:
      ```
      (dr) $ cd ./cluster/
      (dr) $ time python pca_hdbscan.py --cfg cluster.cfg  > pca_hdbscan_stdout.txt
      ```
    - Graph percentage of functions highlighted:
      ```
      (dr) $ cd ./cluster/
      (dr) $ python function_coverage.py --functions /data/malicious_unpacked_bndb_function/ \
                                         --fn ../autoencoder/roi/fn.npy \
                                         --addr ../autoencoder/roi/addr.npy \
                                         --output function_coverage.png > function_coverage_stdout.txt
      ```

--------------------------------------------------------------------------------

## Grading
  - Here we provide real malware binaries compiled from source code which have been [open-sourced or leaked](https://thezoo.morirt.com/). **These are real malware. Do NOT execute these binaries. They should be used for educational purposes only.**
  - [Download](https://github.com/fireeye/capa/releases) CAPA release binary and move it to `grader/capa/capa`
  - Extract CAPA results
    ```
    (dr) $ cd grader/capa/
    (dr) $ ./output_data.sh
    ```
  - Graph ROC curves
    ```
    (dr) $ cd grader/
    (dr) $ unzip malware.zip # Password is "infected"
    (dr) $ time ./roc.sh &> roc_stdout_stderr.txt
    ```
    - [roc_rbot.png](grader/roc_rbot.png)
    - [roc_pegasus.png](grader/roc_pegasus.png)
    - [roc_carbanak.png](grader/roc_carbanak.png)
    - [roc_combined.png](grader/roc_combined.png)
  - Pick desired threshold
    ```
    (dr) $ vim roc_stdout_stderr.txt
    ```
  - Examine FPs & FNs due to chosen threshold
    ```
    (dr) $ ./examine.sh 9.053894787328584e-08 &> examine_stdout_stderr.txt
    (dr) $ vim examine_stdout_stderr.txt
    ```
  - Observe characteristics about benign and malicious functions
    ```
    (dr) $ ./info.sh > info_stdout.txt
    (dr) $ vim info_stdout.txt
    ```

## Post Processing
  - To continue improving results, we've added some post-processing steps to our tool.
  - Run post-processing
    ```
    (dr) $ cd post-processing/
    ```
    - Prioritize TPs over FPs
      - Sort functions by MSE value (calculated by avg RoI MSE value) to list TPs before FPs
        - Our intuition is that functions more unrecognizable by the autoencoder are more likely to be malicious.
      - Sort functions by number of basic blocks to list TPs before FPs
        - We observed that (on average) malicious functions from our ground-truth samples have more basic blocks than benign functions.
      - Sort functions by number of function callees (functions called by a function) to list TPs before FPs
        - We observed that (on average) malicious functions from our ground-truth samples have more function callees than benign functions.
      - Sort functions randomly to list TPs before FPs
        - This is a gut-check to make sure something naive won't work better
      - Sort functions by address to list TPs before FPs
        - It is *obviously* a poor assumption that malicious functionalities would appear in the binary in a specific order linearly.
        - This is a gut-check to make sure something naive won't work better
      - Grade each option from above
        ```
        # Run "roc.sh" above first

        (dr) $ ./grade_sort.sh 9.053894787328584e-08 > grade_sort_stdout.txt
        (dr) $ vim grade_sort_stdout.txt
        ```
    - Reduce FPs
      - Ignore functions which have few basic blocks
          - We observed that (on average) malicious functions from our ground-truth samples have more basic blocks than benign functions.
      - Ignore functions which have few internal/external function calls
          - We observed that (on average) malicious functions from our ground-truth samples have more callees than benign functions.
      - See above grader section for this option's results
    - Reduce FNs
      - Signature-based solutions can be used to identify *known* functionalities, and thus could catch FNs missed by DeepReflect.
        - If CAPA identifies a function, it's marked. Else it gets a score from DeepReflect.
      - See above grader section for this option's results

--------------------------------------------------------------------------------

## FAQs
  - Why don't you release the binaries used to train and evaluate DeepReflect (other than ground-truth samples)?
    - We cannot release malware binaries because of our agreement with those who provided them to us.
      - If you're looking for malware binaries, you might consider the [SOREL dataset](https://github.com/sophos-ai/SOREL-20M) or contacting [VirusTotal](https://www.virustotal.com/).
    - We cannot release benign binaries because of copyright rules.
      - If you're looking for benign binaries, you might consider [crawling](https://github.com/evandowning/selenium-crawler) them on [CNET](https://download.cnet.com/windows/). Make sure to verify they're not malicious via [VirusTotal](https://www.virustotal.com/).
    - We do, however, release our extracted features so models can be trained from scratch.
      - [DeepReflect Dataset](Dataset.md)

## Citing
  ```
  @inproceedings{downing_deepreflect_2021,
    title = {{DeepReflect}: {Discovering} {Malicious} {Functionality} through {Binary} {Reconstruction}},
    shorttitle = {{DeepReflect}},
    booktitle = {30th \$\{\${USENIX}\$\}\$ {Security} {Symposium} (\$\{\${USENIX}\$\}\$ {Security} 21)},
    author = {Downing, Evan and Mirsky, Yisroel and Park, Kyuhong and Lee, Wenke},
    year = {2021}
  }
  ```
  - [Paper](https://www.usenix.org/conference/usenixsecurity21/presentation/downing)
  - [Reproducing paper experiments](reproducing_paper/README.md)

