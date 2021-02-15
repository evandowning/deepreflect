#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: ./write_commands.sh samples.txt output/"
    exit 2
fi

sampleFN=$1
output=$2

mkdir -p "$output"

# Read in file
while read line;
do
    sha=$(echo "$line" | rev | cut -d '/' -f 1 | rev)
    sha=${sha:0:-5}
    family=$(echo "$line" | rev | cut -d '/' -f 2 | rev)

    mkdir -p "${output}/${family}/"

    # Run trace
    echo "python get_function.py acfg_plus --bndb $line --output ${output}/${family}/${sha}.txt"
done < "$sampleFN"
