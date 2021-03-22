#!/bin/bash

if (( $# != 2 )); then
    >&2 echo "usage: ./write_commands_raw.sh bndb/ output/"
    exit 2
fi

sample=$1
output=$2

# Get bndb files
for path in `find ${sample} -type f`;
do
    # Get hash value and family name of binary
    sha=$(echo ${path} | rev | cut -d '/' -f 1 | rev)
    sha=${sha:0:-5} # remove ".bndb"
    family=$(echo ${path} | rev | cut -d '/' -f 2 | rev)

    # Extract raw features
    echo "python extract_raw.py binja --bndb ${path} --output ${output}/${family}/${sha}.txt"
done
