"""
01_03_estimate_luminal_mask_area.py
====================================
Estimate the luminal mask area for each segmented acinus using transcript
density as a proxy for the glandular lumen boundary.

For each acinus identified in 01_01:
  1. Retrieve the xy coordinates of all transcripts belonging to cells in
     that acinus from the Baysor segmentation output.
  2. Compute a 2-D Gaussian KDE (kernel density estimate) over those
     transcript positions to produce a density image.
  3. Threshold at the 90th percentile of non-zero density values to define
     the luminal mask.
  4. Fill gaps in the mask via morphological dilation followed by erosion.
  5. Record the luminal area (µm²) and save pixel coordinates.

Usage
-----
  python 01_03_estimate_luminal_mask_area.py <sid>

  sid : str
      Sample ID matching the Batch column in adata_acini.h5ad.

Inputs
------
  ../data/adata_acini.h5ad
      AnnData with acini assignments from 01_02 (obs columns acini_luminal,
      acini_afterdil, etc.).
  ../data/imgs_crop_level0/{sid}_index.csv
      Pixel-index file with xmin/ymin offsets for coordinate alignment.
  ../data/baysor_xen/{sid}/segmentation.csv
      Baysor transcript-level output with columns: cell, x, y.

Outputs
-------
  ../results/mask_coords_area/filled_lummask_coords_{sid}.csv
      Pixel coordinates (x, y) of the gap-filled luminal mask per acinus.
  ../results/mask_coords_area/lummask_area_{sid}.csv
      Luminal area (µm²) per acinus ID.

Parameters (set in the script body)
--------------------------------------
  pix_size  = 0.2125   µm per pixel (level-0 resolution)
  extra_d   = 1000     pixels of padding added around each mask before
                       morphological filling (prevents edge artifacts)
  dinit     = 50       structural element side length for dilation/erosion
"""

import numpy as np
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import tifffile as tff
from skimage import morphology

import glob
from pathlib import Path
import sys
import scipy
from scipy import ndimage


# ── Command-line argument ──────────────────────────────────────────────────────
sid = sys.argv[1]

# ── Parameters ─────────────────────────────────────────────────────────────────
pix_size = 0.2125              # µm per pixel (level-0 resolution)
pix_area = pix_size ** 2       # µm² per pixel
extra_d  = 1000                # padding pixels for morphological filling
dinit    = 50                  # dilation/erosion kernel size
plotit   = False

# ── Output path ────────────────────────────────────────────────────────────────
savepth = '../results/mask_coords_area/'
Path(savepth).mkdir(parents=True, exist_ok=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def compute_density_image(points, pspan=10, bwvalue=0.1):
    """
    Compute a 2-D Gaussian KDE density image from transcript xy coordinates.

    Parameters
    ----------
    points : pd.DataFrame
        Columns: x_centroid_px, y_centroid_px.
    pspan : int
        Extra pixel margin beyond the coordinate bounding box.
    bwvalue : float
        Bandwidth factor passed to scipy.stats.gaussian_kde (bw_method).

    Returns
    -------
    points_use : pd.DataFrame
        Coordinates shifted to image-local origin.
    density_image : 2-D np.ndarray
        KDE density values on the pixel grid.
    xmin, xmax, ymin, ymax : int
        Bounding box in original pixel coordinates.
    """
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0).round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0).round()]

    points_use = points.copy()
    points_use['x_centroid_px'] = points['x_centroid_px'] - xmin
    points_use['y_centroid_px'] = points['y_centroid_px'] - ymin

    xsize = xmax - xmin + 1
    ysize = ymax - ymin + 1
    X, Y  = np.mgrid[0:xsize, 0:ysize]
    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = scipy.stats.gaussian_kde(points_use.transpose(), bw_method=bwvalue)
    density_image = kernel(positions).T.reshape((xsize, ysize)).T

    return points_use, density_image, xmin, xmax, ymin, ymax


def img_extraspace(a, d_buff):
    """
    Embed image `a` in a zero-padded array with `d_buff` pixels on each side.

    Parameters
    ----------
    a : 2-D np.ndarray
    d_buff : int
        Padding width in pixels.

    Returns
    -------
    out : 2-D np.ndarray
        Padded image with shape (H + 2*d_buff, W + 2*d_buff).
    """
    p, q = a.shape
    out = np.zeros((2*d_buff + p, 2*d_buff + q), dtype=a.dtype)
    out[d_buff:p + d_buff, d_buff:q + d_buff] = a
    return out


# ── Load data ──────────────────────────────────────────────────────────────────
adata = sc.read('../data/adata_acini.h5ad')

# Pixel index file provides xmin/ymin offsets for coordinate alignment
df_idx = pd.read_csv(f'../data/imgs_crop_level0/{sid}_index.csv', index_col=0)

# Baysor transcript-level output; filter out unassigned transcripts
fn_baysor = f'../data/baysor_xen/{sid}/segmentation.csv'
transcripts_df = pd.read_csv(fn_baysor, usecols=["cell", "x", "y"])
transcripts_df = transcripts_df[~transcripts_df.cell.isna()].copy()

# Convert transcript µm coordinates to pixel coordinates (aligned to cropped image)
transcripts_df['x_centroid_px'] = (
    transcripts_df['x'] / pix_size - df_idx.idx_px.xmin
)
transcripts_df['y_centroid_px'] = (
    transcripts_df['y'] / pix_size - df_idx.idx_px.ymin
)
transcripts_df['cid'] = transcripts_df.cell.str.split('-', n=1, expand=True)[1]

# Subset to cells assigned to any acinus in this sample
adata_sub = adata[
    (adata.obs.Batch == sid) & (adata.obs.acini_luminal != '-1')
].copy()

kernel = np.ones((dinit, dinit))  # structuring element for morphological ops


# ── Main loop: per-acinus luminal mask estimation ──────────────────────────────
df_coords_all    = []   # raw KDE-thresholded mask coordinates
df_coords_er_all = []   # gap-filled (dilation+erosion) mask coordinates
df_area          = []   # luminal area (µm²) per acinus

for aid in adata_sub.obs.acini_luminal.unique():
    # Retrieve transcript coordinates for cells in this acinus
    cids = (
        adata_sub[adata_sub.obs.acini_luminal == aid]
        .obs.index.str.replace('cell_', '').str.replace('-{}'.format(sid), '')
    )
    points = transcripts_df[transcripts_df.cid.isin(cids)][
        ['x_centroid_px', 'y_centroid_px']
    ].copy()

    # Step 1: KDE density image
    points_use, density_image, xmin, xmax, ymin, ymax = compute_density_image(
        points, pspan=100
    )

    # Step 2: Threshold at 90th percentile of non-zero density → luminal mask
    lum_mask = density_image > np.percentile(density_image[density_image != 0], 90)

    # Step 3: Fill gaps via binary dilation + erosion (10 iterations each)
    #         Padding prevents edge artefacts during morphological operations.
    lum_mask_padded = img_extraspace(lum_mask, extra_d)
    dilated_mask    = ndimage.binary_dilation(lum_mask_padded, kernel, iterations=10)
    eroded_mask     = ndimage.binary_erosion(dilated_mask, kernel, iterations=10)

    # Step 4: Save gap-filled coordinates (shifted back to original pixel space)
    coords_er = np.argwhere(eroded_mask)
    df_coords_er = pd.DataFrame(coords_er, columns=['y', 'x'])
    df_coords_er['x'] = df_coords_er['x'] + xmin - extra_d
    df_coords_er['y'] = df_coords_er['y'] + ymin - extra_d
    df_coords_er['aid'] = aid

    df_coords_er_all = (
        df_coords_er.copy() if len(df_coords_er_all) == 0
        else pd.concat([df_coords_er_all, df_coords_er])
    )

    # Step 5: Save raw mask coordinates and luminal area
    coords = np.argwhere(lum_mask)
    df_coords = pd.DataFrame(coords, columns=['y', 'x'])
    df_coords['x'] = df_coords['x'] + xmin
    df_coords['y'] = df_coords['y'] + ymin
    df_coords['aid'] = aid

    df_coords_all = (
        df_coords.copy() if len(df_coords_all) == 0
        else pd.concat([df_coords_all, df_coords])
    )

    area_um2 = np.sum(lum_mask) * pix_area
    if len(df_area) == 0:
        df_area = pd.DataFrame({aid: area_um2}, index=['area'])
    else:
        df_area[aid] = area_um2

    print(f"  Acinus {aid}: {area_um2:.1f} µm²")


# ── Save outputs ───────────────────────────────────────────────────────────────
df_coords_er_all.to_csv(f'{savepth}/filled_lummask_coords_{sid}.csv')
df_area.to_csv(f'{savepth}/lummask_area_{sid}.csv')

print(f"[{sid}] Outputs saved to {savepth}")
print('done')
