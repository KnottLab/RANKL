"""
01_04_spatial_bin_preassignment.py
====================================
Pre-assign pixels to periductal spatial bins by expanding each acinus's luminal
mask in 4-cell-diameter increments, creating concentric distance rings ("spatial bins").

Bins are defined as donut-shaped regions around each acinus:
  acini_afterdil : 1 cell dilation beyond the luminal mask (outer layer of acinus, a1)
  acini_dist4    : 0–4 cell-diameter ring (periglandular, p0)
  acini_dist8    : 4–8 cell-diameter ring (periglandular, p1)
  acini_dist12   : 8–12 cell-diameter ring (intralobular, i0)
  acini_dist16   : 12–16 cell-diameter ring (intralobular, i1)
  acini_dist20   : 16–20 cell-diameter ring (intralobular, i2)

Note: At this stage, rings from different acini may overlap.
Overlapping pixels are reassigned to the nearest acinus in script 01_05.

Usage
-----
  python 01_04_spatial_bin_preassignment.py <sid>

  sid : str
      Sample ID matching the Batch column in adata_acini.h5ad.

Inputs
------
  ../data/adata_acini.h5ad
      AnnData with acini assignments from 01_02.
  ../results/mask_coords_area/filled_lummask_coords_{sid}.csv
      Gap-filled luminal mask pixel coordinates from 01_03.

Outputs
-------
  ../results/mask_coords_area/lummask_coords_expanded_{sid}.csv
      All pixel coordinates for every distance bin (columns: x, y, aid, mask).
  ../results/mask_coords_area/lummask_area_expanded_{sid}.csv
      Pixel area (µm²) per bin per acinus, before overlap resolution.

Parameters (set in the script body)
--------------------------------------
  pix_size  = 0.2125   µm per pixel (level-0 resolution)
  pspan     = 100      pixel margin added around the bounding box
  extra_d   = 2000     padding added before dilation (prevents edge wrap)
  d_cell    = 4.372    µm, mean epithelial cell diameter
  dinit     = ceil(2 * d_cell / pix_size)   dilation kernel side length
"""

import numpy as np
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import glob
from pathlib import Path
import sys
from scipy import ndimage


# ── Command-line argument ──────────────────────────────────────────────────────
sid = sys.argv[1]

# ── Parameters ─────────────────────────────────────────────────────────────────
pix_size = 0.2125              # µm per pixel (level-0 resolution)
pix_area = pix_size ** 2       # µm² per pixel
pspan    = 100                 # bounding-box margin (pixels)
extra_d  = 2000                # padding for dilation (pixels)
d_cell   = 4.371621916951665   # mean epithelial cell diameter (µm)
dinit    = int(np.ceil(2 * d_cell / pix_size))  # kernel side length (pixels)

# Ordered bin names; each represents a concentric ring around the acinus
colnms = ['acini_luminal', 'acini_afterdil',
          'acini_dist4', 'acini_dist8', 
          'acini_dist12', 'acini_dist16', 'acini_dist20']


# ── Helper functions ───────────────────────────────────────────────────────────

def img_extraspace(a, d_buff):
    """
    Embed image `a` in a zero-padded array with `d_buff` pixels on each side.
    Prevents boundary artefacts during morphological dilation.
    """
    p, q = a.shape
    out = np.zeros((2*d_buff + p, 2*d_buff + q), dtype=a.dtype)
    out[d_buff:p + d_buff, d_buff:q + d_buff] = a
    return out


def coords2img(points, pspan, extra_d=0):
    """
    Convert a DataFrame of (x, y) pixel coordinates into a binary image array.

    Parameters
    ----------
    points : pd.DataFrame
        Must have columns 'x' and 'y'.
    pspan : int
        Margin added to bounding box in all directions.
    extra_d : int
        Additional zero-padding added after image creation.

    Returns
    -------
    img : 2-D np.ndarray
        Binary image (1 = pixel present, 0 = absent).
    xmin, ymin : int
        Bounding-box origin in original pixel coordinates.
    """
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0).round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0).round()]

    points_use = points.copy()
    points_use['x'] = points['x'] - xmin
    points_use['y'] = points['y'] - ymin

    img = np.zeros((points_use['y'].max()+1, points_use['x'].max()+1))
    img[points_use['y'], points_use['x']] = 1

    if extra_d != 0:
        img = img_extraspace(img, extra_d)

    return img, xmin, ymin


# ── Load data ──────────────────────────────────────────────────────────────────
adata = sc.read('../data/adata_acini.h5ad')
df_coords_er_all = pd.read_csv(
    f'../results/mask_coords_area/filled_lummask_coords_{sid}.csv',
    index_col=0
)


# ── Main loop: build concentric distance rings per acinus ──────────────────────
df_area      = []
df_coords_all = []

for aid in df_coords_er_all.aid.unique():
    # Convert luminal mask coordinates to binary image
    points = df_coords_er_all.query('aid == @aid')[['x', 'y']]
    img, xmin, ymin = coords2img(points, pspan, extra_d)

    pre_mask = None

    for i in range(1, len(colnms)):
        colnm = colnms[i]

        # Determine dilation extent in pixels for this bin:
        #   acini_afterdil → 1 cell-diameter beyond luminal mask
        #   acini_distN    → N+1 cell-diameters (cumulative)
        if colnm == 'acini_afterdil':
            n_cell = 1
        else:
            n_cell = int(colnm.split('dist')[-1]) + 1

        dilated_mask = ndimage.binary_dilation(img, iterations=dinit * n_cell)

        # Donut = current dilation minus previous dilation (or luminal mask)
        if colnm == 'acini_afterdil':
            donut_mask = dilated_mask.astype('int') - img
        else:
            donut_mask = dilated_mask.astype('int') - pre_mask

        pre_mask = dilated_mask.astype('int')

        # Convert donut pixels back to original coordinate space
        coords = np.argwhere(donut_mask)
        df_coords = pd.DataFrame(coords, columns=['y', 'x'])
        df_coords['x']    = df_coords['x'] + xmin - extra_d
        df_coords['y']    = df_coords['y'] + ymin - extra_d
        df_coords['aid']  = aid
        df_coords['mask'] = colnm

        df_coords_all = (
            df_coords.copy() if len(df_coords_all) == 0
            else pd.concat([df_coords_all, df_coords])
        )

        # Record area for this bin
        area_um2 = np.sum(donut_mask) * pix_area
        if len(df_area) == 0:
            df_area = pd.DataFrame({aid: area_um2}, index=[colnm])
        else:
            try:
                df_area.loc[colnm, aid] = area_um2
            except KeyError:
                df_area[aid] = 0.0
                df_area.loc[colnm, aid] = area_um2

    print(f"  Acinus {aid} done")


# ── Save outputs ───────────────────────────────────────────────────────────────
df_coords_all.to_csv(
    f'../results/mask_coords_area/lummask_coords_expanded_{sid}.csv'
)
df_area.to_csv(
    f'../results/mask_coords_area/lummask_area_expanded_{sid}.csv'
)

print(f"[{sid}] Outputs saved.")
print('done')
