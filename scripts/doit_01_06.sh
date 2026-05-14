#!/bin/bash
# doit_01_06.sh
# Map resolved pixel masks back to cells in AnnData (01_06) for a single sample.
#
# Usage: bash doit_01_06.sh <sid>
#   sid : sample ID

sid=$1
coords_pth="../results/dfs_newlyassigned_rescaled2x"
echo "[$(date)] Starting 01_06 mask-to-cell mapping for: $sid"
nohup python -u 01_06_map_mask_to_cell.py "$sid" "$coords_pth" \
    > "../logs/01_06_${sid}.log" 2>&1
echo "[$(date)] Done: $sid"
