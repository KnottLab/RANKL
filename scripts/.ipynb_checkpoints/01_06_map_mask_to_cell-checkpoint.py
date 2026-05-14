"""
01_06_map_mask_to_cell.py
==========================
Map resolved periductal/intralobular spatial bin pixel masks back to individual 
cells in the AnnData object.

For each distance bin (mask_ncell4, mask_ncell8, …, mask_ncell20) and each
acinus, the script:
  1. Converts the deduplicated pixel coordinates from 01_05 into a binary image.
  2. For each cell whose centroid falls within the image bounding box, reads the
     mask value at its pixel location to determine bin membership.
  3. Applies a consistency check: a cell is only assigned to an acinus if it has
     not already been assigned to a different acinus in another distance bin,
     preventing conflicting assignments.
  4. Saves the per-cell bin assignments as a CSV for later consolidation in 01_07.

Usage
-----
  python 01_06_map_mask_to_cell.py <sid> <coords_pth>

  sid        : str
      Sample ID matching the Batch column in adata_acini.h5ad.
  coords_pth : str
      Path to the folder containing the resolved coordinate CSVs from 01_05,
      e.g. '../results/dfs_newlyassigned_rescaled2x/'.

Inputs
------
  ../data/adata_acini.h5ad
      AnnData with acini assignments (obs: y_centroid_px, x_centroid_px).
  {coords_pth}/coords_{sid}_mask_ncell{N}.csv
      Deduplicated pixel coordinates for each distance bin from 01_05.

Outputs
-------
  {coords_pth}/adata_cols_expand_{sid}.csv
      Per-cell distance bin assignments (columns: mask_ncell4 … mask_ncell20).
      Values are acinus IDs or '-1' (unassigned).

Parameters (set in the script body)
--------------------------------------
  pix_size = 0.2125   µm per pixel (level-0 resolution)
"""

import numpy as np
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import glob
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ── Helper functions ───────────────────────────────────────────────────────────

def coords2img(points, pspan, extra_d=0):
    """
    Convert a DataFrame of (x, y) pixel coordinates into a binary image array.

    Parameters
    ----------
    points : pd.DataFrame
        Must have columns 'x' and 'y'.
    pspan : int
        Margin added to bounding box.
    extra_d : int
        Additional zero-padding (rarely needed here).

    Returns
    -------
    img : 2-D np.ndarray
        Binary image (1 = pixel present, 0 = absent).
    xmin, xmax, ymin, ymax : int
        Bounding-box corners in original pixel coordinates.
    """
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0)[['x', 'y']].round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0)[['x', 'y']].round()]

    points_use = points.copy()
    points_use['x'] = points['x'] - xmin
    points_use['y'] = points['y'] - ymin

    img = np.zeros((points_use['y'].max()+1, points_use['x'].max()+1))
    img[points_use['y'], points_use['x']] = 1

    if extra_d != 0:
        p, q = img.shape
        out  = np.zeros((2*extra_d + p, 2*extra_d + q), dtype=img.dtype)
        out[extra_d:p+extra_d, extra_d:q+extra_d] = img
        img  = out

    return img, xmin, xmax, ymin, ymax


# ── Command-line arguments ─────────────────────────────────────────────────────
sid        = sys.argv[1]
coords_pth = sys.argv[2]

# ── Parameters ─────────────────────────────────────────────────────────────────
pix_size = 0.2125          # µm per pixel (level-0 resolution)
pix_area = pix_size ** 2   # µm² per pixel

# ── Load data ──────────────────────────────────────────────────────────────────
adata = sc.read('../data/adata_acini.h5ad')
adata_sub = adata[adata.obs.Batch == sid].copy()

# Cell centroid pixel coordinates (used for mask look-up)
df_coords_cell = adata_sub.obs[['y_centroid_px', 'x_centroid_px']].copy()


# ── Map each distance bin mask to cells ────────────────────────────────────────
# Iterate over distance bins in 4-cell increments (4, 8, 12, 16, 20)
for d_ncell in range(4, 21, 4):
    new_colnm = f'mask_ncell{d_ncell}'
    fn_coords = f'{coords_pth}/coords_{sid}_{new_colnm}.csv'

    if not os.path.isfile(fn_coords):
        print(f"  [{new_colnm}] coordinate file not found; skipping.")
        continue

    df_coords_bin = pd.read_csv(fn_coords, index_col=0)
    # Remove any remaining duplicate pixels (safety check)
    df_coords_bin = df_coords_bin[~df_coords_bin[['x', 'y']].duplicated()]

    adata_sub.obs[new_colnm] = '-1'  # initialise; -1 = unassigned

    for aid in df_coords_bin.aid.unique():
        # Convert this acinus's bin pixels to a binary image
        points = df_coords_bin.query('aid == @aid')[['x', 'y']]
        points['mask'] = 1
        img, xmin, xmax, ymin, ymax = coords2img(points, pspan=0)

        # Restrict cell look-up to cells within the image bounding box
        df_sub = df_coords_cell.query(
            'x_centroid_px >= @xmin & x_centroid_px <= @xmax & y_centroid_px >= @ymin & y_centroid_px <= @ymax'
        )
        if df_sub.shape[0] == 0:
            continue

        # Shift cell coordinates to image-local origin and look up mask value
        df_sub = df_sub.copy()
        df_sub['x_centroid_px'] -= xmin
        df_sub['y_centroid_px'] -= ymin
        df_sub = np.floor(df_sub).astype('int')
        df_sub['mask'] = img[tuple(np.array(df_sub.T))]
        cids = df_sub.query('mask == 1').index

        # Consistency check: only assign cells that agree across all filled bins.
        # A cell is excluded if its acinus ID is already different in another bin,
        # preventing a cell from being counted in two different acini's rings.
        cols    = [a for a in adata_sub.obs.columns if 'mask_' in a]
        tmp     = adata_sub.obs.loc[cids, cols].astype('str')
        cids_no = tmp[~tmp.eq(tmp.iloc[:, 0], axis=0).all(1)].index
        cids    = [a for a in cids if a not in cids_no]

        if len(cids) > 0:
            adata_sub.obs.loc[cids, new_colnm] = aid

    print(f"  [{sid}] {new_colnm} mapped.")


# ── Save per-cell bin assignments ──────────────────────────────────────────────
savecols = [a for a in adata_sub.obs.columns if 'mask_ncell' in a]
adata_sub.obs[savecols].to_csv(
    f'{coords_pth}/adata_cols_expand_{sid}.csv'
)

print(f"[{sid}] Saved to {coords_pth}/adata_cols_expand_{sid}.csv")
print('done')
