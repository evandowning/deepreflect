#!/bin/bash

shap()
{
    name="$1"

    # Extract ACFG features
    python acfg_preprocess.py binja --bndb "malware-gt/old/${name}.bndb" \
                                    --output "malware-gt-binja/acfg/${name}.txt"

    python acfg_feature_extraction.py --acfg "malware-gt-binja/acfg/" \
                                      --output "malware-gt-binja/acfg-feature/"

    echo "malware-gt-binja/acfg-feature/acfg/${name}.txt" > "malware-gt-binja/acfg-feature/acfg/${name}_path.txt"

    # Get SHAP values
    python explain_shap.py acfg --train "./models/final_plus_benign_joint_acfg_compare/train.txt" \
                                --test "malware-gt-binja/acfg-feature/acfg/${name}_path.txt" \
                                --valid "./models/final_plus_benign_joint_acfg_compare/valid.txt" \
                                --data "test" \
                                --model "./models/final_plus_benign_joint_acfg_compare/vgg19_half_4_2.h5" \
                                --map "./models/final_plus_benign_joint_acfg_compare/final_map.txt" \
                                --joint "True" \
                                --output "malware-gt-binja/acfg-shap/"

    # Parse SHAP values
    python parse_acfg_shap_all.py --acfg "malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                                  --shap-val "malware-gt-binja/acfg-shap/${name}.txt_shap_val.txt" \
                                  --out "malware-gt-binja/acfg-shap-eval/${name}.txt.npy"

}

acfg ()
{
    name="$1"

    # Extract ACFG features
    python acfg_preprocess.py binja --bndb "malware-gt/old/${name}.bndb" \
                                    --output "malware-gt-binja/acfg/${name}.txt"

    python acfg_feature_extraction.py --acfg "malware-gt-binja/acfg/" \
                                      --output "malware-gt-binja/acfg-feature/"

    # Extract Autoencoder values
    python autoencoder_eval.py acfg --model "./models/autoencoder_benign_unipacker/m2_normalize_24_12.h5" \
                                    --normalize "True" \
                                    --acfg-feature "malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                                    --output "malware-gt-binja/acfg-autoencoder/${name}.npy"
}

acfg_plus ()
{
    name="$1"

    # Extract ACFG plus features
    python acfg_plus_preprocess.py binja --bndb "malware-gt/old/${name}.bndb" \
                                         --output "malware-gt-binja/acfg-plus/${name}.txt"

    python acfg_plus_feature_extraction.py --acfg "malware-gt-binja/acfg-plus/${name}.txt" \
                                           --output "malware-gt-binja/acfg-plus-feature/"

    # Extract Autoencoder values
    python autoencoder_eval.py acfg_plus --model "./models/autoencoder_benign_unipacker_plus/m2_normalize_24_12.h5" \
                                         --normalize "True" \
                                         --acfg-feature "malware-gt-binja/acfg-plus-feature/acfg-plus/${name}.txt.npy" \
                                         --output "malware-gt-binja/acfg-plus-autoencoder/${name}.npy"
}

name="rbot"

echo "SHAP + ACFG"
# SHAP + ACFG
shap "$name"

echo "Autoencoder + ACFG"
# Autoencoder + ACFG
acfg "$name"

echo "Autoencoder + ACFG plus"
# Autoencoder + ACFG Plus
acfg_plus "$name"
