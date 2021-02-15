#!/bin/bash

shap ()
{
    name="$1"

    echo "$name - SHAP - ACFG"

    python roc.py --mse "../malware-gt-binja/acfg-shap-eval/${name}.txt.npy" \
                  --acfg-feature "../malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                  --func "./rbot/${name}_function.txt" \
                  --bndb-func "./rbot/${name}_bndb_function.txt" \
                  --gt "./rbot/${name}_annotations.txt" \
                  --roc "./rbot/${name}_shap_acfg_roc"
}

roc ()
{
    name="$1"

    echo "$name - Autoencoder - ACFG"

    python roc.py --mse "../malware-gt-binja/acfg-autoencoder/${name}.npy" \
                  --acfg-feature "../malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                  --func "./rbot/${name}_function.txt" \
                  --bndb-func "./rbot/${name}_bndb_function.txt" \
                  --gt "./rbot/${name}_annotations.txt" \
                  --roc "./rbot/${name}_ae_acfg_roc"
}

roc_plus ()
{
    name="$1"

    echo "$name - Autoencoder - ACFG plus"

    python roc.py --mse "../malware-gt-binja/acfg-plus-autoencoder/${name}.npy" \
                  --acfg-plus-feature "../malware-gt-binja/acfg-plus-feature/acfg-plus/${name}.txt.npy" \
                  --func "./rbot/${name}_function.txt" \
                  --bndb-func "./rbot/${name}_bndb_function.txt" \
                  --gt "./rbot/${name}_annotations.txt" \
                  --roc "./rbot/${name}_ae_acfg_plus_roc"
}

capa()
{
    name="$1"

    echo "$name - CAPA"

    cd ./capa/
    ./output_data.sh
    cd ../
}

name="rbot"
shap "$name"
roc "$name"
roc_plus "$name"
capa "$name"

# For function average
echo "Function MSE calculation: Average of BB MSE values"
python separate.py "./rbot/rbot_ae_acfg_plus_roc_data_func_avg.npz" \
                   "./rbot/rbot_ae_acfg_roc_data_func_avg.npz" \
                   "./rbot/rbot_shap_acfg_roc_data_func_avg.npz" \
                   "./capa/rbot_capa_data_func.npz" \
                   "./functionsimsearch/rbot_data_func_avg.npz" \
                   "DeepReflect" \
                   "AE_ABB" \
                   "SHAP_ABB" \
                   "CAPA" \
                   "FunctionSimSearch" \
                   "Rbot" \
                   "./rbot/roc_avg_all.png"

