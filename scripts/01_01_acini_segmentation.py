"""
01_01_acini_segmentation.py
============================
Segment mammary acini (glandular units) from Xenium spatial transcriptomics data
using DBSCAN clustering on epithelial cells identified by DAPI signal and marker
expression (EPCAM+, ACTA2-).

Pipeline for each sample (sid):
  1. Load DAPI image and dilate high-intensity pixels to create a nuclear mask
  2. Map the DAPI mask to cells in the AnnData object
  3. Identify epithelial cells: DAPI+ and EPCAM+ and ACTA2-
  4. Cluster epithelial cells into acini with DBSCAN
  5. Dilate each DBSCAN cluster by n_cell_dil cell-diameters to fill gaps
  6. Save per-sample acini assignments to CSV

Usage
-----
  python 01_01_acini_segmentation.py <sid>

  sid : str
      Sample ID matching the Batch column in adata.h5ad and the filename of
      the cropped DAPI tif in data/imgs_crop_level0/.

Inputs
------
  ../data/adata.h5ad
      Cell-level AnnData with obs columns including:
        Batch, x_centroid, y_centroid, x_centroid_px, y_centroid_px
  ../data/imgs_crop_level0/{sid}.tif
      Cropped DAPI image at level-0 resolution (0.2125 µm/pixel).

Outputs
-------
  ../results/dbscan_epi_dapi_cell/{sid}.csv
      Per-cell assignments:
        dapi_mask    : 1 = in DAPI-bright region, -1 = background
        ACTA2_mask   : 1 = ACTA2 high (myoepithelial), 0 = otherwise
        EPCAM_mask   : 1 = EPCAM high (luminal epithelial), 0 = otherwise
        acini_luminal  : DBSCAN cluster label (-1 = noise/unassigned)
        acini_afterdil : cluster label after 1-cell-diameter dilation
  ../results/dbscan_epi_dapi_cell/figures/
      QC plots at each processing step.

Parameters (set in the script body)
--------------------------------------
  pix_size   = 0.2125   µm per pixel at level-0 resolution
  d_cell     = 4.372    µm, mean epithelial cell diameter (estimated from our Xenium data)
  n_cell     = 6        DBSCAN epsilon in cell-diameter units (this was optimized to return the best output.)
  n_cell_dil = 1        dilation radius in cell-diameter units
"""

import numpy as np
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import tifffile as tff

import glob
from pathlib import Path
import sys

from sklearn.cluster import DBSCAN


# ── Command-line argument ──────────────────────────────────────────────────────
sid = sys.argv[1]

# ── Parameters ─────────────────────────────────────────────────────────────────
pix_size   = 0.2125              # µm per pixel (level-0 resolution)
d_cell     = 4.371621916951665   # mean epithelial cell diameter (µm)
n_cell     = 6                   # DBSCAN epsilon = d_cell * n_cell (µm)
n_cell_dil = 1                   # post-clustering dilation radius (cell diameters)
saveplot   = True

# ── Output paths ───────────────────────────────────────────────────────────────
savepth = '../results/dbscan_epi_dapi_cell/'
Path(savepth).mkdir(parents=True, exist_ok=True)
Path(savepth, 'figures').mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
adata = sc.read('../data/adata.h5ad')
adata_sub = adata[adata.obs.Batch == sid].copy()

fn_img = glob.glob(f'../data/imgs_crop_level0/{sid}.tif')[0]
image  = tff.imread(fn_img)


# ── Step 1: Build dilated DAPI nuclear mask ────────────────────────────────────
# Identify bright DAPI pixels (>99.5th percentile) and dilate by 1 cell-diameter
# to create a continuous nuclear mask. Used to filter cells co-localizing with
# dense nuclear regions (i.e., real tissue vs. background).

p99 = np.percentile(image, 99.5)
y, x = np.where(image > p99)
value = image[image > p99]
df_img = pd.DataFrame({'x': x, 'y': y, 'value': value})

ymax = np.ceil(adata_sub.obs.y_centroid_px.max()).astype('int')
xmax = np.ceil(adata_sub.obs.x_centroid_px.max()).astype('int')

img_new = np.ones((ymax + 1, xmax + 1)) * -1
span    = int(np.ceil(1 * d_cell / pix_size))

for i in range(df_img.shape[0]):
    xl = span if df_img.x[i] >= span else 0
    xr = span if df_img.x[i] + span < img_new.shape[1] else 0
    yl = span if df_img.y[i] >= span else 0
    yr = span if df_img.y[i] + span < img_new.shape[0] else 0
    img_new[df_img.y[i]-yl:df_img.y[i]+yr, df_img.x[i]-xl:df_img.x[i]+xr] = 1

if saveplot:
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.imshow(img_new, cmap='gray')
    plt.savefig(f'{savepth}/figures/img_dapi_p995_dilated_{sid}.pdf')
    plt.close()


# ── Step 2: Map DAPI mask to cells ────────────────────────────────────────────
# For each cell, check the surrounding 2-cell-diameter neighborhood.
# Assign dapi_mask=1 if any DAPI-bright pixels are found; -1 otherwise.
# Acini with fewer than 5 cells passing this filter are excluded (noise).

span = int(np.ceil(2 * d_cell / pix_size))
adata_sub.obs['dapi_mask'] = -1

for i in range(adata_sub.shape[0]):
    xid = int(np.floor(adata_sub.obs.x_centroid_px.iloc[i]))
    yid = int(np.floor(adata_sub.obs.y_centroid_px.iloc[i]))
    xl  = span if xid >= span else 0
    xr  = span if xid + span < img_new.shape[1] else 0
    yl  = span if yid >= span else 0
    yr  = span if yid + span < img_new.shape[0] else 0
    tmp_array = img_new[yid-yl:yid+yr, xid-xl:xid+xr]
    if not (tmp_array == 0).all():
        uniq, counts = np.unique(tmp_array[tmp_array.nonzero()], return_counts=True)
        adata_sub.obs['dapi_mask'].iloc[i] = uniq[np.argmax(counts)]

# Remove acini clusters represented by fewer than 5 cells
tmp_db = adata_sub.obs.dapi_mask.value_counts()
list_lowcell = tmp_db[tmp_db < 5].index
adata_sub.obs.loc[adata_sub.obs['dapi_mask'].isin(list_lowcell), 'dapi_mask'] = -1

if saveplot:
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.imshow(image, cmap='gray')
    sns.scatterplot(x='x_centroid_px', y='y_centroid_px',
                    data=adata_sub.obs[adata_sub.obs.dapi_mask == 1],
                    hue='dapi_mask', s=2, alpha=0.9)
    plt.axis('scaled')
    plt.savefig(f'{savepth}/figures/img_dapi_p995_dilated_cell_{sid}.pdf')
    plt.close()


# ── Step 3: Identify epithelial cells (EPCAM+, ACTA2-) ────────────────────────
# Use population-level percentile thresholds to define marker-positive cells:
#   ACTA2 > 95th percentile → myoepithelial (excluded)
#   EPCAM > 75th percentile → luminal epithelial (kept)

p95_acta2 = np.percentile(adata[:, 'ACTA2'].X.toarray(), 95)
p75_epcam = np.percentile(adata[:, 'EPCAM'].X.toarray(), 75)

adata_sub.obs['ACTA2_mask'] = (adata_sub[:, 'ACTA2'].X.toarray().ravel() > p95_acta2).astype(int)
adata_sub.obs['EPCAM_mask'] = (adata_sub[:, 'EPCAM'].X.toarray().ravel() > p75_epcam).astype(int)

# Keep only DAPI+ EPCAM+ ACTA2- cells for DBSCAN
adata_mask = adata_sub[
    (adata_sub.obs.ACTA2_mask != 1) &
    (adata_sub.obs.EPCAM_mask == 1) &
    (adata_sub.obs.dapi_mask == 1)
]

if saveplot:
    plt.rcParams["figure.figsize"] = (10, 5)
    sns.set(rc={'axes.facecolor': 'black', 'figure.facecolor': 'black', 'axes.grid': False})
    sns.scatterplot(x='x_centroid', y='y_centroid', data=adata_mask.obs,
                    hue='EPCAM_mask', s=5, alpha=0.9, palette=["white"])
    plt.gca().get_legend().remove()
    plt.axis('scaled')
    plt.savefig(f'{savepth}/figures/spatial_dapi_epi_mask_{sid}.pdf')
    plt.close()


# ── Step 4: DBSCAN clustering to identify acini ───────────────────────────────
# epsilon = d_cell * n_cell (µm), min_samples=5 cells per cluster.
# Each cluster corresponds to one acinus (glandular unit).

adata_sub_epi = adata_mask.copy()
db = DBSCAN(eps=d_cell * n_cell, min_samples=5).fit(
    adata_sub_epi.obs[['x_centroid', 'y_centroid']]
)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_    = list(labels).count(-1)
print(f"[{sid}] DBSCAN: {n_clusters_} acini, {n_noise_} noise cells")

adata_sub_epi.obs['dbscan_cluster'] = pd.Categorical(labels)

if saveplot:
    plt.rcParams["figure.figsize"] = (10, 5)
    sns.set(rc={'axes.facecolor': 'black', 'figure.facecolor': 'black', 'axes.grid': False})
    sns.scatterplot(x='x_centroid', y='y_centroid',
                    data=adata_sub_epi.obs[adata_sub_epi.obs.dbscan_cluster != -1],
                    hue='dbscan_cluster', s=5, alpha=0.9, palette="Set1")
    plt.gca().get_legend().remove()
    plt.axis('scaled')
    plt.savefig(f'{savepth}/figures/spatial_dbscan_orig_{sid}.pdf')
    plt.close()

# Transfer DBSCAN labels back to the full sample AnnData
adata_sub.obs['acini_luminal'] = -1
adata_sub.obs.loc[adata_sub_epi.obs.index, 'acini_luminal'] = (
    adata_sub_epi.obs['dbscan_cluster'].astype('int')
)


# ── Step 5: Dilate clusters to fill gaps (optional) ───────────────────────────
# Expand each DBSCAN cluster outward by n_cell_dil cell-diameters.
# This assigns nearby unassigned cells to their nearest cluster,
# filling small gaps at cluster boundaries.

if n_cell_dil > 0:
    adata_sub.obs['acini_afterdil'] = -1
    adata_sub.obs.loc[adata_sub_epi.obs.index, 'acini_afterdil'] = (
        adata_sub_epi.obs['dbscan_cluster'].astype('int')
    )

    dbs_clusters = adata_sub.obs['acini_afterdil'].unique()[1:]
    cells_notasg = adata_sub.obs.loc[
        adata_sub.obs['acini_afterdil'] == -1, ['x_centroid', 'y_centroid']
    ]

    for cl in dbs_clusters:
        tmp = adata_sub.obs.loc[
            adata_sub.obs['acini_afterdil'] == cl, ['x_centroid', 'y_centroid']
        ]
        for cell in tmp.index:
            df_dist = (
                (cells_notasg.x_centroid - tmp.loc[cell].x_centroid) ** 2 +
                (cells_notasg.y_centroid - tmp.loc[cell].y_centroid) ** 2
            ).pow(0.5)
            tmp_notasg = df_dist[df_dist < n_cell_dil * d_cell]
            if len(tmp_notasg) != 0:
                adata_sub.obs.loc[tmp_notasg.index, 'acini_afterdil'] = cl

    if saveplot:
        groups = adata_sub.obs.groupby('acini_afterdil')
        plt.rcParams["figure.figsize"] = (10, 10)
        fig, ax = plt.subplots()
        for name, group in groups:
            color = 'k' if name == -1 else None
            marker_size = 0.5 if name == -1 else 1
            ax.plot(group.x_centroid, group.y_centroid,
                    marker='.', linestyle='', markersize=marker_size,
                    color=color, label=name)
        plt.axis('scaled')
        plt.savefig(f'{savepth}/figures/spatial_dbscan_dilated_{sid}.pdf')
        plt.close()


# ── Save outputs ───────────────────────────────────────────────────────────────
adata_sub.obs[['dapi_mask', 'ACTA2_mask', 'EPCAM_mask',
               'acini_luminal', 'acini_afterdil']].to_csv(
    f'{savepth}/{sid}.csv'
)
print(f"[{sid}] Saved to {savepth}/{sid}.csv")
print('all done')
