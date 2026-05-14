#!/bin/bash
# doit_01_05.sh
# Run pixel overlap resolution and rescaling (01_05) for a single sample
# and one distance bin. Run once per bin name per sample.
#
# Usage: bash doit_01_05.sh <sid> <colnm>
#   sid   : sample ID
#   colnm : bin name, one of: acini_dist4 acini_dist8 acini_dist12 acini_dist16 acini_dist20

sid=$1
colnm=$2
echo "[$(date)] Starting 01_05 pixel reassignment for: $sid / $colnm"
nohup python -u 01_05_run_reassign_pixel_rescale_expand.py "$sid" "$colnm" \
    > "../logs/01_05_${sid}_${colnm}.log" 2>&1
echo "[$(date)] Done: $sid / $colnm"
