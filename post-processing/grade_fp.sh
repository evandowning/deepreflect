#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "usage: ./grade_fp.sh <threshold>"
    exit 2
fi

grade()
{
    family="$1"
    args=("$@")
    num=$(($# - 1))

    root="../grader/"
    root_input="${root}/malware/${family}/"

    root_output="${root_input}/output"
    mkdir -p "${root_output}"

    # Construct parameters
    mseParam=""
    featureParam=""
    funcParam=""
    annotationParam=""
    for i in `seq $num`; do
        name="${args[$i]}"
        base="${root_output}/${name: 0:-4}"

        feature="${base}_feature.npy"

        function="${base}_function.txt"
        mse="${base}_mse"
        annotation="${root_input}/${name: 0:-4}_annotation.txt"


        mseParam="${mseParam}${mse}/output/${name: 0:-4}_feature.npy "
        featureParam="${featureParam}${feature} "
        funcParam="${funcParam}${function} "
        annotationParam="${annotationParam}${annotation} "
    done

    # Sort randomly
    echo "Sort functions randomly"
    python sort.py --mse "${mseParam}" \
                   --feature "${featureParam}" \
                   --bndb-func "${funcParam}" \
                   --annotation "${annotationParam}" \
                   --threshold "${threshold}" \
                   --sort-random
    echo ""

    # Sort by function address value
    echo "Sort functions by function address"
    python sort.py --mse "${mseParam}" \
                   --feature "${featureParam}" \
                   --bndb-func "${funcParam}" \
                   --annotation "${annotationParam}" \
                   --threshold "${threshold}" \
                   --sort-addr
    echo ""

    # Sort by number of basic blocks
    echo "Sort functions by number of basic blocks"
    python sort.py --mse "${mseParam}" \
                   --feature "${featureParam}" \
                   --bndb-func "${funcParam}" \
                   --annotation "${annotationParam}" \
                   --threshold "${threshold}" \
                   --sort-bb
    echo ""

    # Sort by MSE values
    echo "Sort functions by MSE value"
    python sort.py --mse "${mseParam}" \
                   --feature "${featureParam}" \
                   --bndb-func "${funcParam}" \
                   --annotation "${annotationParam}" \
                   --threshold "${threshold}" \
                   --sort-mse
    echo ""
}

rbot()
{
    family="rbot"
    echo "${family}"

    name="rbot.exe"
    grade "${family}" "${name}"

    echo "========================="
}
pegasus()
{
    family="pegasus"
    echo "${family}"

    grade "${family}" \
            "idd.x32" \
            "mod_CmdExec.x32"\
            "mod_DomainReplication.x32"\
            "mod_LogonPasswords.x32" \
            "mod_NetworkConnectivity.x32" \
            "rse.x32"

    echo "========================="
}
carbanak()
{
    family="carbanak"
    echo "${family}"

    grade "${family}" \
            "AutorunSidebar.dll"\
            "bot.exe" \
            "botcmd.exe" \
            "cve2014-4113.dll" \
            "downloader.exe" \
            "rdpwrap.dll"

    echo "========================="
}

threshold="$1"


rbot
pegasus
carbanak
