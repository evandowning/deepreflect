#!/bin/bash

# Extract BinaryNinja folder
7z x binaryninja.7z

# Install BinaryNinja Python library
cd binaryninja/scripts/
./linux-setup.sh

# Copy license file to location
cd ../../
cp license.dat ~/.binaryninja/
