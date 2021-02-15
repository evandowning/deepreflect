#!/bin/bash

shap()
{
    fam="$1"
    name="$2"
    lower="${name,,}"

    echo "${name}: SHAP + ACFG"

    # Extract ACFG features
    python acfg_preprocess.py binja --bndb "malware-gt/new/pegasus/binres/${name}.bndb" \
                                    --output "malware-gt-binja/acfg/${fam}_${lower}.txt"

    python acfg_feature_extraction.py --acfg "malware-gt-binja/acfg/" \
                                      --output "malware-gt-binja/acfg-feature/"

    echo "malware-gt-binja/acfg-feature/acfg/${fam}_${lower}.txt" > "malware-gt-binja/acfg-feature/acfg/${fam}_${lower}_path.txt"

    # Get SHAP values
    python explain_shap.py acfg --train "./models/final_plus_benign_joint_acfg_compare/train.txt" \
                                --test "malware-gt-binja/acfg-feature/acfg/${fam}_${lower}_path.txt" \
                                --valid "./models/final_plus_benign_joint_acfg_compare/valid.txt" \
                                --data "test" \
                                --model "./models/final_plus_benign_joint_acfg_compare/vgg19_half_4_2.h5" \
                                --map "./models/final_plus_benign_joint_acfg_compare/final_map.txt" \
                                --joint "True" \
                                --output "malware-gt-binja/acfg-shap/"

    # Parse SHAP values
    python parse_acfg_shap_all.py --acfg "malware-gt-binja/acfg-feature/acfg/${fam}_${lower}.txt" \
                                  --shap-val "malware-gt-binja/acfg-shap/${fam}_${lower}.txt_shap_val.txt" \
                                  --out "malware-gt-binja/acfg-shap-eval/${fam}_${lower}.txt.npy"

}

acfg ()
{
    fam="$1"
    name="$2"
    lower="${name,,}"

    echo "${name}: Autoencoder + ACFG"

    # Extract ACFG features
    python acfg_preprocess.py binja --bndb "malware-gt/new/pegasus/binres/${name}.bndb" \
                                    --output "malware-gt-binja/acfg/${fam}_${lower}.txt"

    python acfg_feature_extraction.py --acfg "malware-gt-binja/acfg/" \
                                      --output "malware-gt-binja/acfg-feature/"

    # Extract Autoencoder values
    python autoencoder_eval.py acfg --model "./models/autoencoder_benign_unipacker/m2_normalize_24_12.h5" \
                                    --normalize "True" \
                                    --acfg-feature "malware-gt-binja/acfg-feature/acfg/${fam}_${lower}.txt" \
                                    --output "malware-gt-binja/acfg-autoencoder/${fam}_${lower}.npy"
}

acfg_plus ()
{
    fam="$1"
    name="$2"
    lower="${name,,}"

    echo "${name}: Autoencoder + ACFG plus"

    python acfg_plus_preprocess.py binja --bndb "malware-gt/new/pegasus/binres/${name}.bndb" \
                                         --output "malware-gt-binja/acfg-plus/${fam}_${lower}.txt"

    python acfg_plus_feature_extraction.py --acfg "malware-gt-binja/acfg-plus/${fam}_${lower}.txt" \
                                           --output "malware-gt-binja/acfg-plus-feature/"

    python autoencoder_eval.py acfg_plus --model "./models/autoencoder_benign_unipacker_plus/m2_normalize_24_12.h5" \
                                         --normalize "True" \
                                         --acfg-feature "malware-gt-binja/acfg-plus-feature/acfg-plus/${fam}_${lower}.txt.npy" \
                                         --output "malware-gt-binja/acfg-plus-autoencoder/${fam}_${lower}.npy"
}

run ()
{
    fam="$1"
    name="$2"

    shap "$fam" "$name"
    acfg "$fam" "$name"
    acfg_plus "$fam" "$name"
}

fam="pegasus"

name="idd"
run "$fam" "$name"

name="mod_CmdExec"
run "$fam" "$name"

name="mod_DomainReplication"
run "$fam" "$name"

name="mod_LogonPasswords"
run "$fam" "$name"

name="mod_NetworkConnectivity"
run "$fam" "$name"

name="rse"
run "$fam" "$name"
