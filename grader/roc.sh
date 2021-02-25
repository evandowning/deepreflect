#!/bin/bash

# Graph each malware sample's ROC curve
./rbot.sh &> rbot_stdout_stderr.txt
cp "./malware/rbot/output/combined_roc.png" "./roc_rbot.png"

./pegasus.sh &> pegasus_stdout_stderr.txt
cp "./malware/pegasus/output/combined_roc.png" "./roc_pegasus.png"

./carbanak.sh &> carbanak_stdout_stderr.txt
cp "./malware/carbanak/output/combined_roc.png" "./roc_carbanak.png"

# Graph combined ROC curve
python combine.py "./malware/rbot/output/rbot_roc_func_data.npz" \
                  "./malware/pegasus/output/combined_roc_func_data.npz" \
                  "./malware/carbanak/output/combined_roc_func_data.npz" \
                  "combined_roc_func_data.npz"

python roc_multi.py "combined_roc_func_data.npz" \
                    "DeepReflect" \
                    "Combined" \
                    "roc_combined.png"
