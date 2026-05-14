#!/bin/bash
# doit_01_01.sh
# Run acini segmentation (01_01) for a single sample.
#
# Usage: bash doit_01_01.sh <sid>
#   sid : sample ID (e.g., slide1_1622753)

sid=$1
echo "[$(date)] Starting 01_01 acini segmentation for: $sid"
nohup python -u 01_01_acini_segmentation.py "$sid" > "../logs/01_01_${sid}.log" 2>&1
echo "[$(date)] Done: $sid"
