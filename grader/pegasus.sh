#!/bin/bash

combine ()
{
    base="$1"
    output="$2"

    echo "Combine Func"

    python combine.py "${base}_idd_data_func.npz" \
                      "${base}_mod_cmdexec_data_func.npz" \
                      "${base}_mod_domainreplication_data_func.npz" \
                      "${base}_mod_logonpasswords_data_func.npz" \
                      "${base}_mod_networkconnectivity_data_func.npz" \
                      "${base}_rse_data_func.npz" \
                      "${output}_data_func.npz"

    python combine.py "${base}_idd_data_func_avg.npz" \
                      "${base}_mod_cmdexec_data_func_avg.npz" \
                      "${base}_mod_domainreplication_data_func_avg.npz" \
                      "${base}_mod_logonpasswords_data_func_avg.npz" \
                      "${base}_mod_networkconnectivity_data_func_avg.npz" \
                      "${base}_rse_data_func_avg.npz" \
                      "${output}_data_func_avg.npz"

    python combine.py "${base}_idd_data_func_avg_log.npz" \
                      "${base}_mod_cmdexec_data_func_avg_log.npz" \
                      "${base}_mod_domainreplication_data_func_avg_log.npz" \
                      "${base}_mod_logonpasswords_data_func_avg_log.npz" \
                      "${base}_mod_networkconnectivity_data_func_avg_log.npz" \
                      "${base}_rse_data_func_avg_log.npz" \
                      "${output}_data_func_avg_log.npz"
}

shap ()
{
    fam="$1"
    name="$2"

    echo "$name - SHAP - ACFG"

    python roc.py --mse "../malware-gt-binja/acfg-shap-eval/${name}.txt.npy" \
                  --acfg-feature "../malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                  --func "./pegasus/${name}_function.txt" \
                  --bndb-func "./pegasus/${name}_bndb_function.txt" \
                  --gt "./pegasus/${name}_annotation.txt" \
                  --roc "./pegasus/${fam}_shap_acfg_roc_${name}"
}


roc ()
{
    fam="$1"
    name="$2"

    echo "$name - Autoencoder - ACFG"

    python roc.py --mse "../malware-gt-binja/acfg-autoencoder/${name}.npy" \
                  --acfg-feature "../malware-gt-binja/acfg-feature/acfg/${name}.txt" \
                  --func "./pegasus/${name}_function.txt" \
                  --bndb-func "./pegasus/${name}_bndb_function.txt" \
                  --gt "./pegasus/${name}_annotation.txt" \
                  --roc "./pegasus/${fam}_ae_acfg_roc_${name}"
}

roc_plus ()
{
    fam="$1"
    name="$2"

    echo "$name - Autoencoder - ACFG plus"

    python roc.py --mse "../malware-gt-binja/acfg-plus-autoencoder/${name}.npy" \
                  --acfg-plus-feature "../malware-gt-binja/acfg-plus-feature/acfg-plus/${name}.txt.npy" \
                  --func "./pegasus/${name}_function.txt" \
                  --bndb-func "./pegasus/${name}_bndb_function.txt" \
                  --gt "./pegasus/${name}_annotation.txt" \
                  --roc "./pegasus/${fam}_ae_acfg_plus_roc_${name}"
}

capa()
{
    name="$1"

    echo "$name - CAPA"

    cd ./capa/
    ./output_data.sh
    cd ../
}

fam="pegasus"

name="${fam}_idd"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_mod_cmdexec"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_mod_domainreplication"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_mod_logonpasswords"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_mod_networkconnectivity"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

name="${fam}_rse"
shap "$fam" "$name"
roc "$fam" "$name"
roc_plus "$fam" "$name"

capa "$fam"

# Combine ROC data
base="./pegasus/pegasus_ae_acfg_roc_pegasus"
output="./pegasus/combined/ae_acfg"
combine "$base" "$output"

base="./pegasus/pegasus_ae_acfg_plus_roc_pegasus"
output="./pegasus/combined/ae_acfg_plus"
combine "$base" "$output"

base="./pegasus/pegasus_shap_acfg_roc_pegasus"
output="./pegasus/combined/shap_acfg"
combine "$base" "$output"

# For function average
echo "Separate: Average"
python separate.py "./pegasus/combined/ae_acfg_plus_data_func_avg.npz" \
                   "./pegasus/combined/ae_acfg_data_func_avg.npz" \
                   "./pegasus/combined/shap_acfg_data_func_avg.npz" \
                   "./capa/pegasus_capa_data_func.npz" \
                   "./functionsimsearch/pegasus_data_func_avg.npz" \
                   "DeepReflect" \
                   "AE_ABB" \
                   "SHAP_ABB" \
                   "CAPA" \
                   "FunctionSimSearch" \
                   "Pegasus" \
                   "./pegasus/combined/separate_avg.png"

