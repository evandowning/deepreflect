#!/bin/bash

# Graph each malware sample's ROC curve
./rbot.sh &> rbot_stdout_stderr.txt
./pegasus.sh &> pegasus_stdout_stderr.txt
./carbanak.sh &> carbanak_stdout_stderr.txt

# Graph combined ROC curve
python combine.py "./malware/rbot/output/rbot_roc_func_data.npz" \
                  "./malware/pegasus/output/combined_roc_func_data.npz" \
                  "./malware/carbanak/output/combined_roc_func_data.npz" \
                  "combined_roc_func_data.npz"

python roc_multi.py "combined_roc_func_data.npz" \
                    "DeepReflect" \
                    "Combined" \
                    "roc_combined.png"
