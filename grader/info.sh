#!/bin/bash

if (( $# != 0 )); then
    >&2 echo "usage: ./info.sh"
    exit 2
fi

info()
{
    family="$1"
    name="$2"

    root=`pwd`
    root_input="${root}/malware/${family}/"

    root_output="${root_input}/output"
    mkdir -p "${root_output}"

    base="${root_output}/${name: 0:-4}"
    bndb="${base}.bndb"
    annotation="${root_input}/${name: 0:-4}_annotation.txt"

    python info.py --bndb "${bndb}" \
                   --annotation "${annotation}"
}

rbot()
{
    family="rbot"
    echo "${family}"

    name="rbot.exe"
    info "${family}" "${name}"

    echo "========================="
}
pegasus()
{
    family="pegasus"
    echo "${family}"

    name="idd.x32"
    info "${family}" "${name}"

    name="mod_CmdExec.x32"
    info "${family}" "${name}"

    name="mod_DomainReplication.x32"
    info "${family}" "${name}"

    name="mod_LogonPasswords.x32"
    info "${family}" "${name}"

    name="mod_NetworkConnectivity.x32"
    info "${family}" "${name}"

    name="rse.x32"
    info "${family}" "${name}"

    echo "========================="
}
carbanak()
{
    family="carbanak"
    echo "${family}"

    name="AutorunSidebar.dll"
    info "${family}" "${name}"

    name="bot.exe"
    info "${family}" "${name}"

    name="botcmd.exe"
    info "${family}" "${name}"

    name="cve2014-4113.dll"
    info "${family}" "${name}"

    name="downloader.exe"
    info "${family}" "${name}"

    name="rdpwrap.dll"
    info "${family}" "${name}"

    echo "========================="
}

threshold="$1"

rbot
pegasus
carbanak
