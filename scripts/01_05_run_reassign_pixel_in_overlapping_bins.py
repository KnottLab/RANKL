"""
01_05_run_reassign_pixel_in_overlapping_bins.py
==========================================
Resolve pixel ownership conflicts in overlapping periductal/intralobular distance bins.

When two acini are close together, the expanded distance rings produced in 01_04 overlap. 
This script:
  1. Identifies pixels claimed by more than one acinus in a given bin.
  2. Reassigns each disputed pixel to the nearest acinus (Euclidean distance
     to the nearest boundary point of the luminal mask).
  3. Up-scales the result back to the original resolution after processing.

To reduce memory usage, coordinate arrays are scaled down by `gamma` before
duplicate detection and nearest-neighbour lookup, then rescaled back.

Usage
-----
  python 01_05_run_reassign_pixel_in_overlapping_bins.py <sid> <colnm>

  sid   : str
      Sample ID matching the Batch column in adata_acini.h5ad.
  colnm : str
      Bin name to process, e.g. 'acini_dist4', 'acini_dist8', …, 'acini_dist20'.

Inputs
------
  ../results/mask_coords_area/lummask_coords_expanded_{sid}.csv
      Pre-assigned bin coordinates from 01_04 (may contain duplicates).
  ../results/mask_coords_area/filled_lummask_coords_{sid}.csv
      Gap-filled luminal mask coordinates (used to find nearest acinus boundary).

Outputs
-------
  ../results/dfs_newlyassigned_rescaled2x/coords_{sid}_mask_ncell{N}.csv
      Deduplicated pixel coordinates (x, y, aid) for the requested bin.
  ../results/dfs_newlyassigned_rescaled2x/area_{sid}_mask_ncell{N}.csv
      Pixel area (µm²) per acinus ID for the requested bin.

Parameters (set in the script body)
--------------------------------------
  pix_size = 0.2125   µm per pixel (level-0 resolution)
  gamma    = 0.5      down-scaling factor for duplicate resolution step
"""

import numpy as np
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

from skimage import segmentation, transform
from scipy import spatial

import re
import glob
from pathlib import Path
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ── Helper functions ───────────────────────────────────────────────────────────

def img_extraspace(a, d_buff):
    """Embed image `a` in a zero-padded array with `d_buff` pixels on each side."""
    p, q = a.shape
    out = np.zeros((2*d_buff + p, 2*d_buff + q), dtype=a.dtype)
    out[d_buff:p + d_buff, d_buff:q + d_buff] = a
    return out


def coords2img(points, pspan, extra_d=0, rescaling=0, xy=[0, 0]):
    """
    Convert a DataFrame of (x, y, aid) pixel coordinates into a labelled image.

    Each pixel is assigned the integer index of its acinus (aid), so the
    image encodes which acinus owns each pixel.  Optionally rescales the image
    by `rescaling` for faster processing.

    Parameters
    ----------
    points : pd.DataFrame
        Must have columns 'x', 'y', and 'aid'.
    pspan : int
        Margin added to bounding box.
    extra_d : int
        Additional zero-padding.
    rescaling : float
        If > 0, rescale image by this factor (e.g. 0.5 = half resolution).
    xy : [xmin, ymin]
        Override bounding-box origin (use when consistent origin is needed).

    Returns
    -------
    img : 2-D np.ndarray
        Integer-labelled image (0 = empty, N = acinus index N).
    xmin, xmax, ymin, ymax : int
        Bounding-box corners.
    dict_aid2int : dict
        Mapping from acinus ID (str) to integer label in the image.
    """
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0)[['x', 'y']].round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0)[['x', 'y']].round()]

    if xy != [0, 0]:
        xmin, ymin = xy
    if rescaling < 1:
        xmax, xmin, ymax, ymin = [a - a % 10 for a in [xmax, xmin, ymax, ymin]]

    points_use = points.copy()
    points_use['x'] = points['x'] - xmin
    points_use['y'] = points['y'] - ymin

    dict_aid2int = {a: i+1 for i, a in enumerate(points.aid.unique())}

    img = np.zeros((points_use['y'].max()+1, points_use['x'].max()+1))
    img[points_use['y'], points_use['x']] = points_use.aid.map(dict_aid2int)

    if extra_d != 0:
        img = img_extraspace(img, extra_d)
    if rescaling != 0:
        img = transform.rescale(img, rescaling, anti_aliasing=False, order=0)
        xmin, xmax, ymin, ymax = [int(float(a)*rescaling) for a in [xmin, xmax, ymin, ymax]]

    return img, xmin, xmax, ymin, ymax, dict_aid2int


def rescale_coords(df_coords, rescaling=0.1, byaid=False, xy=[0, 0]):
    """
    Rescale pixel coordinate DataFrame by `rescaling` factor.

    When `byaid=True`, rescales each acinus independently to avoid label
    collisions in the labelled image.

    Returns
    -------
    df_coords_rescaled : pd.DataFrame
        Rescaled coordinates (x, y, aid).
    img : 2-D np.ndarray
        Labelled image at rescaled resolution.
    xmin, ymin : int
        Bounding-box origin at rescaled resolution.
    """
    if byaid:
        df_coords_rescaled = []
        for a in df_coords.aid.unique():
            tmp = df_coords.query('aid == @a')
            img, xmin, xmax, ymin, ymax, _ = coords2img(tmp, 0, rescaling=rescaling)
            coords = np.argwhere(img)
            tmp_df = pd.DataFrame(coords, columns=['y', 'x'])
            tmp_df['aid'] = a
            tmp_df['x'] += xmin
            tmp_df['y'] += ymin
            df_coords_rescaled = (
                tmp_df.copy() if len(df_coords_rescaled) == 0
                else pd.concat([df_coords_rescaled, tmp_df])
            )
        return df_coords_rescaled, None, None, None
    else:
        img, xmin, xmax, ymin, ymax, dict_aid2int = coords2img(
            df_coords, 0, rescaling=rescaling, xy=xy
        )
        dict_int2aid = dict(map(reversed, dict_aid2int.items()))
        coords = np.argwhere(img)
        df_out = pd.DataFrame(coords, columns=['y', 'x'])
        df_out['aid'] = img[df_out['y'], df_out['x']].astype('int').map(dict_int2aid)
        df_out['x'] += xmin
        df_out['y'] += ymin
        return df_out, img, xmin, ymin


def progressbar(current_value, total_value, bar_lengh=30,
                progress_char='■', binname=''):
    """Print an in-place progress bar."""
    percentage = int((current_value / total_value) * 100)
    progress   = int((bar_lengh * current_value) / total_value)
    loadbar    = "Progress {}: [{:{len}}]{}%".format(
        binname, progress * progress_char, percentage, len=bar_lengh
    )
    print(loadbar, end='\r')


# ── Command-line arguments ─────────────────────────────────────────────────────
sid   = sys.argv[1]
colnm = sys.argv[2]

# ── Parameters ─────────────────────────────────────────────────────────────────
pix_size = 0.2125          # µm per pixel (level-0 resolution)
pix_area = pix_size ** 2   # µm² per pixel
gamma    = 0.5             # down-scaling factor for duplicate resolution

# ── Output path ────────────────────────────────────────────────────────────────
savepth = '../results/dfs_newlyassigned_rescaled{}x/'.format(int(1 / gamma))
Path(savepth).mkdir(parents=True, exist_ok=True)


# ── Load pre-assigned bin and luminal mask coordinates ─────────────────────────
fn_mask = f'../results/mask_coords_area/lummask_coords_expanded_{sid}.csv'
fn_fill = f'../results/mask_coords_area/filled_lummask_coords_{sid}.csv'

if not os.path.isfile(fn_fill):
    print(f"[{sid}] Luminal mask file not found; skipping.")
    sys.exit(0)

df_coords_all = pd.read_csv(fn_mask, index_col=0)
df_coords_lum = pd.read_csv(fn_fill, index_col=0)
print(f"[{sid}] Files loaded.")

# ── Prepare pixel key (unique x_y string) for duplicate detection ──────────────
df_coords_lum['key'] = df_coords_lum['x'].astype('str') + '_' + df_coords_lum['y'].astype('str')
df_coords_all['key'] = df_coords_all['x'].astype('str') + '_' + df_coords_all['y'].astype('str')

# Remove pixels that overlap with the luminal mask (handled separately)
df_coords_all_nolum = df_coords_all[~df_coords_all.key.isin(df_coords_lum.key)].copy()

# ── Identify pixels in this bin that do not belong to earlier bins ─────────────
colnms     = df_coords_all['mask'].unique()
j          = np.where(colnms == colnm)[0][0]
colnms_pre = colnms[0:j]

d_ncell    = int(re.findall(r'\d+', colnm)[0])
new_colnm  = 'mask_ncell{}'.format(int(d_ncell))

key_bins_pre     = df_coords_all_nolum[df_coords_all_nolum['mask'].isin(colnms_pre)]['key'].unique()
df_coords_bin    = df_coords_all_nolum.query('mask == @colnm')[['x', 'y', 'aid', 'key']]
df_coords_bin_nopre = df_coords_bin[~df_coords_bin.key.isin(key_bins_pre)].copy()

# ── Find nearest acinus boundary for disputed pixels ──────────────────────────
# Scale down luminal mask and find boundaries for nearest-neighbour lookup.
df_coords_lum_rescaled, img_lum, xmin, ymin = rescale_coords(
    df_coords_lum, rescaling=gamma
)

# Boundary pixels of each acinus in the luminal mask
img_b  = segmentation.find_boundaries(img_lum, mode='inner')
coords = np.argwhere(img_b)
df_coords_boundary = pd.DataFrame(coords, columns=['y', 'x'])
df_coords_boundary['x'] += xmin
df_coords_boundary['y'] += ymin
df_coords_boundary = df_coords_boundary.merge(df_coords_lum_rescaled, how='left')

# Scale down bin pixels and identify duplicates (pixels claimed by ≥2 acini)
df_coords_bin_rescaled, img_bin, xmin_bin, ymin_bin = rescale_coords(
    df_coords_bin_nopre, rescaling=gamma, byaid=True
)
xmin_bin, ymin_bin = [int(a) for a in df_coords_bin_nopre.min(axis=0)[['x', 'y']].round()]
df_coords_bin_rescaled['key'] = (
    df_coords_bin_rescaled['x'].astype('str') + '_' +
    df_coords_bin_rescaled['y'].astype('str')
)

df_coords_bin_dup_new = []
df_coords_bin_dup     = df_coords_bin_rescaled[df_coords_bin_rescaled[['key']].duplicated()]
df_coords_bin_dup_all = df_coords_bin_rescaled[
    df_coords_bin_rescaled[['key']].duplicated(keep=False)
]

# ── Resolve duplicates: assign each to the nearest acinus boundary ─────────────
i, n_iter = 1, len(df_coords_bin_dup.key.unique())
for xy_key in df_coords_bin_dup.key.unique():
    tmp   = df_coords_bin_dup_all.query('key == @xy_key')
    aids  = tmp.aid.tolist()
    x, y  = tmp.iloc[0][['x', 'y']]

    # Nearest-neighbour distance to each candidate acinus's boundary
    tmp_coords = df_coords_boundary[df_coords_boundary.aid.isin(aids)]
    xy_array   = np.array(tmp_coords[['x', 'y']])
    id_min     = np.argmin(spatial.distance.cdist(xy_array, [[x, y]]), axis=0)

    tmp_df = pd.DataFrame(
        {'x': x, 'y': y, 'key': xy_key,
         'aid': tmp_coords.iloc[id_min].aid.values[0]},
        index=[0]
    )
    df_coords_bin_dup_new = (
        tmp_df.copy() if len(df_coords_bin_dup_new) == 0
        else pd.concat([df_coords_bin_dup_new, tmp_df])
    )

    if (i % 1000 == 0) or (i == n_iter):
        progressbar(i, n_iter, binname=new_colnm)
    i += 1

# ── Combine resolved duplicates with non-duplicated pixels ────────────────────
df_coords_bin_nodup = df_coords_bin_rescaled[
    ~df_coords_bin_rescaled[['key']].duplicated(keep=False)
].copy()
df_coords_bin_new = pd.concat(
    [df_coords_bin_nodup[['x', 'y', 'key', 'aid']], df_coords_bin_dup_new]
)

# ── Rescale back to original resolution ───────────────────────────────────────
xmin_bin = int(float(int(df_coords_bin_nopre.x.min() - df_coords_bin_nopre.x.min() % 10)) * gamma)
ymin_bin = int(float(int(df_coords_bin_nopre.y.min() - df_coords_bin_nopre.y.min() % 10)) * gamma)

df_coords_bin_final, img_rere, xmin2, ymin2 = rescale_coords(
    df_coords_bin_new.dropna(), rescaling=1/gamma, xy=[xmin_bin, ymin_bin]
)
df_coords_bin_final.dropna(inplace=True)

# ── Save outputs ───────────────────────────────────────────────────────────────
fnout   = f'{savepth}/coords_{sid}_{new_colnm}.csv'
fnout_a = f'{savepth}/area_{sid}_{new_colnm}.csv'

df_coords_bin_final.to_csv(fnout)

df_area = pd.DataFrame(df_coords_bin_final.groupby('aid').size() * pix_area)
df_area.columns = [new_colnm]
df_area.to_csv(fnout_a)

print(f"\n[{sid}] Saved {fnout}")
print('done')
