#!/bin/bash
# doit_01_04.sh
# Run spatial bin pre-assignment (01_04) for a single sample.
#
# Usage: bash doit_01_04.sh <sid>
#   sid : sample ID

sid=$1
echo "[$(date)] Starting 01_04 spatial bin pre-assignment for: $sid"
nohup python -u 01_04_spatial_bin_preassignment.py "$sid" > "../logs/01_04_${sid}.log" 2>&1
echo "[$(date)] Done: $sid"
