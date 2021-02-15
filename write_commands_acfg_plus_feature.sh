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
    # Run trace
    echo "python acfg_plus_feature_extraction.py --acfg $line --output ${output}/"
done < "$sampleFN"
