#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: ./write_commands_function.sh bndb/ output/"
    exit 2
fi

sample=$1
output=$2

# Read in file
for path in `find ${sample} -type f`;
do
    # Get hash value and family name of binary
    sha=$(echo ${path} | rev | cut -d '/' -f 1 | rev)
    sha=${sha:0:-5}
    family=$(echo ${path} | rev | cut -d '/' -f 2 | rev)

    # Extract function data
    echo "python extract_function.py --bndb ${path} --output ${output}/${family}/${sha}.txt"
done
