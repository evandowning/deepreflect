#!/bin/bash

# Experiment to simulate emerging threats

function cluster()
{
    family="$1"

    mkdir -p $family
    mkdir -p "${family}/autoencoder_roi/"

    # Separate dataset (one without the family, one with the family)
    echo "Creating datasets"
    time python separate.py --x "./autoencoder_roi/x_train.npy" \
                            --fn "./autoencoder_roi/train_fn.npy" \
                            --addr "./autoencoder_roi/train_addr.npy" \
                            --family "$family" \
                            --output "${family}/autoencoder_roi/"

    echo "===================="

    # Cluster samples
    echo "Cluster samples - family"
    time python ../pca_hdbscan.py --x "${family}/autoencoder_roi/x_train_minus.npy" \
                                  --fn "${family}/autoencoder_roi/train_fn_minus.npy" \
                                  --addr "${family}/autoencoder_roi/train_addr_minus.npy" > "${family}/pca_hdbscan_stdout_minus.txt"

    echo "===================="

    # Cluster samples + family
    echo "Cluster samples + family"
    time python ../pca_hdbscan.py --x "${family}/autoencoder_roi/x_train_plus.npy" \
                                  --fn "${family}/autoencoder_roi/train_fn_plus.npy" \
                                  --addr "${family}/autoencoder_roi/train_addr_plus.npy" > "${family}/pca_hdbscan_stdout_plus.txt"

    echo "===================="

    # Output sizes of clusters with the new family (label existing and new clusters)
    python emerging.py --minus "${family}/pca_hdbscan_stdout_minus.txt" \
                       --plus "${family}/pca_hdbscan_stdout_plus.txt" \
                       --family "$family"
}

# Identify family to withhold
echo "++++++++++++++++++++"
cluster "zbot"
echo "++++++++++++++++++++"
cluster "gandcrypt"
echo "++++++++++++++++++++"
cluster "cosmicduke"
echo "++++++++++++++++++++"
cluster "wannacry"
echo "++++++++++++++++++++"
