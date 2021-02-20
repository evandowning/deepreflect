#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: ./write_commands_bndb.sh binaries/ output/"
    exit 2
fi

sample=$1
output=$2

# Get binaries
for path in `find ${sample} -type f`;
do
    # Get hash value and family name of binary
    sha=$(echo ${path} | rev | cut -d '/' -f 1 | rev)
    family=$(echo ${path} | rev | cut -d '/' -f 2 | rev)

    # Extract BinaryNinja DB file
    echo "python binja.py --exe ${path} --output ${output}/${family}/${sha}.bndb"
done
