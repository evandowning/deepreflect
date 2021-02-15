#!/bin/bash

combine ()
{
    base="$1"
    output="$2"

    echo "Combine Func"

    python combine.py "${base}_bot_data_func.npz" \
                      "${base}_botcmd_data_func.npz" \
                      "${base}_downloader_data_func.npz" \
                      "${base}_autorunsidebar_data_func.npz" \
                      "${base}_cve2014-4113_data_func.npz" \
                      "${base}_rdpwrap_data_func.npz" \
                      "${output}_data_func.npz"

    python combine.py "${base}_bot_data_func_avg.npz" \
                      "${base}_botcmd_data_func_avg.npz" \
                      "${base}_downloader_data_func_avg.npz" \
                      "${base}_autorunsidebar_data_func_avg.npz" \
                      "${base}_cve2014-4113_data_func_avg.npz" \
                      "${base}_rdpwrap_data_func_avg.npz" \
                      "${output}_data_func_avg.npz"

    python combine.py "${base}_bot_data_func_avg_log.npz" \
                      "${base}_botcmd_data_func_avg_log.npz" \
                      "${base}_downloader_data_func_avg_log.npz" \
                      "${base}_autorunsidebar_data_func_avg_log.npz" \
                      "${base}_cve2014-4113_data_func_avg_log.npz" \
                      "${base}_rdpwrap_data_func_avg_log.npz" \
                      "${output}_data_func_avg_log.npz"
}

shap ()
{
    fam="$1"
    name="$2"

    echo "$name - SHAP - ACFG"

    python roc_corrected.py --mse "../malware-gt-binja/acfg-shap-eval/${name}.txt.npy" \
                  --acfg-feature "../malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                  --func "./carbanak/${name}_function.txt" \
                  --bndb-func "./carbanak/${name}_bndb_function.txt" \
                  --gt "./carbanak/${name}_annotation.txt" \
                  --roc "./carbanak/${fam}_shap_acfg_roc_${name}"
}


roc ()
{
    fam="$1"
    name="$2"

    echo "$name - Autoencoder - ACFG"

    python roc_corrected.py --mse "../malware-gt-binja/acfg-autoencoder/${name}.npy" \
                  --acfg-feature "../malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                  --func "./carbanak/${name}_function.txt" \
                  --bndb-func "./carbanak/${name}_bndb_function.txt" \
                  --gt "./carbanak/${name}_annotation.txt" \
                  --roc "./carbanak/${fam}_ae_acfg_roc_${name}"
}

roc_plus ()
{
    fam="$1"
    name="$2"

    echo "$name - Autoencoder - ACFG plus"

    python roc_corrected.py --mse "../malware-gt-binja/acfg-plus-autoencoder/${name}.npy" \
                  --acfg-plus-feature "../malware-gt-binja/acfg-plus-feature/acfg-plus/${name}.txt.npy" \
                  --func "./carbanak/${name}_function.txt" \
                  --bndb-func "./carbanak/${name}_bndb_function.txt" \
                  --gt "./carbanak/${name}_annotation.txt" \
                  --roc "./carbanak/${fam}_ae_acfg_plus_roc_${name}"
}

capa()
{
    name="$1"

    echo "$name - CAPA"

    cd ./capa/
    ./output_data.sh
    cd ../
}

fam="carbanak"

name="${fam}_bot"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_botcmd"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_downloader"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_autorunsidebar"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_cve2014-4113"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_rdpwrap"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

capa "$fam"

# Combine ROC data
base="./carbanak/carbanak_ae_acfg_roc_carbanak"
output="./carbanak/combined/ae_acfg"
combine "$base" "$output"

base="./carbanak/carbanak_ae_acfg_plus_roc_carbanak"
output="./carbanak/combined/ae_acfg_plus"
combine "$base" "$output"

base="./carbanak/carbanak_shap_acfg_roc_carbanak"
output="./carbanak/combined/shap_acfg"
combine "$base" "$output"

# For function average
echo "Separate: Average"
python separate.py "./carbanak/combined/ae_acfg_plus_data_func_avg.npz" \
                   "./carbanak/combined/ae_acfg_data_func_avg.npz" \
                   "./carbanak/combined/shap_acfg_data_func_avg.npz" \
                   "./capa/carbanak_capa_data_func.npz" \
                   "./functionsimsearch/carbanak_data_func_avg.npz" \
                   "DeepReflect" \
                   "AE_ABB" \
                   "SHAP_ABB" \
                   "CAPA" \
                   "FunctionSimSearch" \
                   "Carbanak" \
                   "./carbanak/combined/separate_avg.png"

