#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "usage: ./examine.sh <threshold>"
    exit 2
fi

roc()
{
    family="$1"
    name="$2"

    root=`pwd`
    root_input="${root}/malware/${family}/"

    root_output="${root_input}/output"
    mkdir -p "${root_output}"

    base="${root_output}/${name: 0:-4}"

    feature="${base}_feature.npy"

    function="${base}_function.txt"
    mse="${base}_mse"
    annotation="${root_input}/${name: 0:-4}_annotation.txt"
    roc_name="${root}/examine/${name: 0:-4}_roc"

    python roc.py --mse "${mse}/output/${name: 0:-4}_feature.npy" \
                  --feature "${feature}" \
                  --bndb-func "${function}" \
                  --annotation "${annotation}" \
                  --threshold "${threshold}" \
                  --roc "${roc_name}"
}

rbot()
{
    family="rbot"
    echo "${family}"

    name="rbot.exe"
    roc "${family}" "${name}"

    echo "========================="
}
pegasus()
{
    family="pegasus"
    echo "${family}"

    name="idd.x32"
    roc "${family}" "${name}"

    name="mod_CmdExec.x32"
    roc "${family}" "${name}"

    name="mod_DomainReplication.x32"
    roc "${family}" "${name}"

    name="mod_LogonPasswords.x32"
    roc "${family}" "${name}"

    name="mod_NetworkConnectivity.x32"
    roc "${family}" "${name}"

    name="rse.x32"
    roc "${family}" "${name}"

    echo "========================="
}
carbanak()
{
    family="carbanak"
    echo "${family}"

    name="AutorunSidebar.dll"
    roc "${family}" "${name}"

    name="bot.exe"
    roc "${family}" "${name}"

    name="botcmd.exe"
    roc "${family}" "${name}"

    name="cve2014-4113.dll"
    roc "${family}" "${name}"

    name="downloader.exe"
    roc "${family}" "${name}"

    name="rdpwrap.dll"
    roc "${family}" "${name}"

    echo "========================="
}

threshold="$1"

rbot
pegasus
carbanak
