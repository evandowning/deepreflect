#!/bin/bash

function capa() {
    target="${1}"
    output="${2}.json"

    # Extract CAPA data
    ./capa -j "$target" > "$output"
}

# Rbot
echo "Rbot"
root="../malware/"
family="rbot"
base="${root}/${family}/"
mkdir -p "${family}"
capa "${base}/rbot.exe" "${family}/rbot"

# Pegasus
echo "Pegasus"
root="../malware/"
family="pegasus"
base="${root}/${family}/"
mkdir -p "${family}"
capa "${root}/${family}/idd.x32" "${family}/idd"
capa "${root}/${family}/mod_CmdExec.x32" "${family}/mod_CmdExec"
capa "${root}/${family}/mod_DomainReplication.x32" "${family}/mod_DomainReplication"
capa "${root}/${family}/mod_LogonPasswords.x32" "${family}/mod_LogonPasswords"
capa "${root}/${family}/mod_NetworkConnectivity.x32" "${family}/mod_NetworkConnectivity"
capa "${root}/${family}/rse.x32" "${family}/rse"

# Carbanak
echo "Carbanak"
root="../malware/"
family="carbanak"
base="${root}/${family}/"
mkdir -p "${family}"
capa "${root}/${family}/bot.exe" "${family}/bot"
capa "${root}/${family}/botcmd.exe" "${family}/botcmd"
capa "${root}/${family}/downloader.exe" "${family}/downloader"
capa "${root}/${family}/AutorunSidebar.dll" "${family}/AutorunSidebar"
capa "${root}/${family}/cve2014-4113.dll" "${family}/cve2014-4113"
capa "${root}/${family}/rdpwrap.dll" "${family}/rdpwrap"
