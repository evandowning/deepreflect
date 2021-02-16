#!/bin/bash

function capa() {
    target="${1}/${2}"
    output="${3}_${2}.json"

    # Extract CAPA data
    ./capa -j "$target" > "$output"
}

# Rbot
echo "Rbot"
root="../../malware-gt/old/"
family="rbot"
name="rbot.exe"
capa $root $name $family
python output_data.py "${family}_${name}.json" "../rbot_final_corrected/rbot_ae_acfg_plus_roc_data_func.npz" \
                      "${family}_capa_data_func.npz"

# Pegasus
echo "Pegasus"
root="../../malware-gt/new/pegasus/binres"
family="pegasus"
capa $root "idd.x32" $family
capa $root "mod_CmdExec.x32" $family
capa $root "mod_DomainReplication.x32" $family
capa $root "mod_LogonPasswords.x32" $family
capa $root "mod_NetworkConnectivity.x32" $family
capa $root "rse.x32" $family
python output_data.py "${family}_idd.x32.json" "../pegasus_final/pegasus_ae_acfg_plus_roc_pegasus_idd_data_func.npz" \
                      "${family}_mod_CmdExec.x32.json" "../pegasus_final/pegasus_ae_acfg_plus_roc_pegasus_mod_cmdexec_data_func.npz" \
                      "${family}_mod_DomainReplication.x32.json" "../pegasus_final/pegasus_ae_acfg_plus_roc_pegasus_mod_domainreplication_data_func.npz" \
                      "${family}_mod_LogonPasswords.x32.json" "../pegasus_final/pegasus_ae_acfg_plus_roc_pegasus_mod_logonpasswords_data_func.npz" \
                      "${family}_mod_NetworkConnectivity.x32.json" "../pegasus_final/pegasus_ae_acfg_plus_roc_pegasus_mod_networkconnectivity_data_func.npz" \
                      "${family}_rse.x32.json" "../pegasus_final/pegasus_ae_acfg_plus_roc_pegasus_rse_data_func.npz" \
                      "${family}_capa_data_func.npz"

# Carbanak
echo "Carbanak"
root="../../malware-gt/new/carbanak/bin/Release"
family="carbanak"
capa $root "bot.exe" $family
capa $root "botcmd.exe" $family
capa $root "downloader.exe" $family
root2="../../malware-gt/new/carbanak/bin/Release simple/plugins/"
capa "$root2" "AutorunSidebar.dll" $family
capa "$root2" "cve2014-4113.dll" $family
capa "$root2" "rdpwrap.dll" $family
python output_data.py "${family}_bot.exe.json" "../carbanak_final/carbanak_ae_acfg_plus_roc_carbanak_bot_data_func.npz" \
                      "${family}_botcmd.exe.json" "../carbanak_final/carbanak_ae_acfg_plus_roc_carbanak_botcmd_data_func.npz" \
                      "${family}_downloader.exe.json" "../carbanak_final/carbanak_ae_acfg_plus_roc_carbanak_downloader_data_func.npz" \
                      "${family}_AutorunSidebar.dll.json" "../carbanak_final/carbanak_ae_acfg_plus_roc_carbanak_autorunsidebar_data_func.npz" \
                      "${family}_cve2014-4113.dll.json" "../carbanak_final/carbanak_ae_acfg_plus_roc_carbanak_cve2014-4113_data_func.npz" \
                      "${family}_rdpwrap.dll.json" "../carbanak_final/carbanak_ae_acfg_plus_roc_carbanak_rdpwrap_data_func.npz" \
                      "${family}_capa_data_func.npz"

