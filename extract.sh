#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "usage: ./extract.sh binaries/"
    exit 2
fi

start_print()
{
    title="DeepReflect: Data Extraction"
    time="UNIX epoch: `date +%s`"

    echo -e "\e[0;31m[$time]\e[0m \e[0;34m[$1]\e[0m"
}

end_print()
{
    time="UNIX epoch: `date +%s`"
    echo -e "\e[0;31m[$time]\e[0m \e[0;32m[Done]\e[0m"
}

# Extracts BinaryNinja DB files
extract_bndb()
{
    binaries="$1"

    dirname=`dirname ${binaries}`
    basename=`basename ${binaries}`

    cd extract/

    ./write_commands_bndb.sh ${binaries} ${dirname}/${basename}_bndb/ > commands_${basename}_bndb.txt
    parallel --memfree 4G --retries 10 -a commands_${basename}_bndb.txt > parallel_bndb_${basename}_stdout.txt 2> parallel_bndb_${basename}_stderr.txt

    cd ../

    echo "${dirname}/${basename}_bndb/"
}

# Extracts raw features
extract_raw()
{
    bndb="$1"

    dirname=`dirname ${bndb}`
    basename=`basename ${bndb}`

    cd extract/

    ./write_commands_raw.sh ${bndb} ${dirname}/${basename}_raw/ > commands_${basename}_raw.txt
    parallel --memfree 4G --retries 10 -a commands_${basename}_raw.txt > parallel_raw_${basename}_stdout.txt 2> parallel_raw_${basename}_stderr.txt

    cd ../

    echo "${dirname}/${basename}_raw/"
}

# Extracts final features
extract_features()
{
    raw="$1"

    dirname=`dirname ${raw}`
    basename=`basename ${raw}`

    cd extract/

    ./write_commands_features.sh ${raw} ${dirname}/${basename}_features/ > commands_${basename}_features.txt
    parallel --memfree 4G --retries 10 -a commands_${basename}_features.txt > parallel_features_${basename}_stdout.txt 2> parallel_features_${basename}_stderr.txt

    cd ../

    echo "${dirname}/${basename}_features/"
}

# Extracts function/basic block information
extract_function()
{
    bndb="$1"

    dirname=`dirname ${bndb}`
    basename=`basename ${bndb}`

    cd extract/

    ./write_commands_function.sh ${bndb} ${dirname}/${basename}_function/ > commands_${basename}_function.txt
    parallel --memfree 4G --retries 10 -a commands_${basename}_function.txt > parallel_function_${basename}_stdout.txt 2> parallel_function_${basename}_stderr.txt

    cd ../

    echo "${dirname}/${basename}_function/"
}

root=`pwd`
binaries="$1"

# Extract BinaryNinja DB files
start_print "Extracting BinaryNinja DB files"
output_bndb=$(extract_bndb ${binaries})
end_print

# Extract raw features
start_print "Extracting raw features"
output_raw=$(extract_raw ${output_bndb})
end_print

# Extract features
start_print "Extracting features"
output_features=$(extract_features ${output_raw})
end_print

# Extract function-related data for malicious files
start_print "Extracting function information"
output_function=$(extract_function ${output_bndb})
end_print

