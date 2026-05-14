#!/bin/bash
# doit_01_03.sh
# Run luminal mask area estimation (01_03) for a single sample.
#
# Usage: bash doit_01_03.sh <sid>
#   sid : sample ID (e.g., s1835007)

sid=$1
echo "[$(date)] Starting 01_03 luminal mask area estimation for: $sid"
nohup python -u 01_03_estimate_luminal_mask_area.py "$sid" > "../logs/01_03_${sid}.log" 2>&1
echo "[$(date)] Done: $sid"
