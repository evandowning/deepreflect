#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "usage: ./write_commands_features.sh raw/ output/"
    exit 2
fi

sample=$1
output=$2

# Read in file
for path in `find ${sample} -type f`;
do
    # Get hash value and family name of binary
    sha=$(echo ${path} | rev | cut -d '/' -f 1 | rev)
    sha=${sha:0:-4} # remove ".txt"
    family=$(echo ${path} | rev | cut -d '/' -f 2 | rev)

    # Extract features
    echo "python extract_features.py --raw ${path} --output ${output}/${family}/${sha}.npy"
done
