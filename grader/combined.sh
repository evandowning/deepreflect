#!/bin/bash

echo "Combining all"

pegasus_base="./pegasus/pegasus_ae_acfg_plus_roc_pegasus"
carbanak_base="./carbanak/carbanak_ae_acfg_plus_roc_carbanak"
rbot_base="./rbot/"
output="./combine"

# Function average
python combine.py "${pegasus_base}_idd_data_func_avg.npz" \
                  "${pegasus_base}_mod_cmdexec_data_func_avg.npz" \
                  "${pegasus_base}_mod_domainreplication_data_func_avg.npz" \
                  "${pegasus_base}_mod_logonpasswords_data_func_avg.npz" \
                  "${pegasus_base}_mod_networkconnectivity_data_func_avg.npz" \
                  "${pegasus_base}_rse_data_func_avg.npz" \
                  "${carbanak_base}_bot_data_func_avg.npz" \
                  "${carbanak_base}_botcmd_data_func_avg.npz" \
                  "${carbanak_base}_downloader_data_func_avg.npz" \
                  "${carbanak_base}_autorunsidebar_data_func_avg.npz" \
                  "${carbanak_base}_cve2014-4113_data_func_avg.npz" \
                  "${carbanak_base}_rdpwrap_data_func_avg.npz" \
                  "${rbot_base}/rbot_ae_acfg_plus_roc_data_func_avg.npz" \
                  "${output}_func_avg"

python separate.py "${output}_func_avg.npz" \
                   "Function_Average" \
                   "All_Ground-Truth" \
                   "${output}_func_avg.png"
