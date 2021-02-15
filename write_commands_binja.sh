#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: ./write_commands_binja.sh samples.txt output/"
    exit 2
fi

sampleFN=$1
output=$2

mkdir -p "$output"

# Read in file
while read line;
do
    sha=$(echo "$line" | rev | cut -d '/' -f 1 | rev)
    family=$(echo "$line" | rev | cut -d '/' -f 2 | rev)

    mkdir -p "${output}/${family}/"

    # Run trace
    echo "python binja.py --exe $line --output ${output}/${family}/${sha}.bndb"
done < "$sampleFN"
