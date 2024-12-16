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


# sid; sample id. This script runs per sample. 
sid = sys.argv[1]

# save plot if True
saveplot = True

""" Set parameters """ 
# pix_size; pixel size per um. This depends on the cropped tif image resolution. 
# In this study, we used the highest resolution (level 0), thus pixel size is 0.2125 um. 
pix_size = 0.2125

# d_cell; one-cell distance (um). This was estimated by average diameter of epithelial cells in our dataset. 
d_cell = 4.371621916951665

# n_cell; number of cells to define the maximum distance between two points to be considered neighbors (epsilon in DBSCAN) 
# This was optimized to give the best output.
n_cell = 6

# n_cell_dil; number of cells to dilate from defined cluster. 
n_cell_dil = 1


""" Set output path """
savepth = '../results/dbscan_epi_dapi_cell/'
# create directory if it does not exist
p = Path(savepth)
p.mkdir(parents=True, exist_ok=True)
p = Path(savepth, 'figures')
p.mkdir(parents=True, exist_ok=True)


""" Read anndata """
f_adata = '../data/adata.h5ad'
adata = sc.read(f_adata)

# subset anndata for a given sample
adata_sub = adata[adata.obs.Batch == sid].copy()

""" Read croped dapi image """
fn_img = glob.glob('../data/imgs_crop_level0/{}.tif'.format(sid))[0]
image = tff.imread(fn_img)


""" Create dapi image mask """
p99 = np.percentile(image, 99.5)
y,x = np.where(image > p99)
value = image[image > p99]
df_img = pd.DataFrame({'x':x, 'y':y, 'value' :value})

ymax = np.ceil(adata_sub.obs.y_centroid_px.max()).astype('int')
xmax = np.ceil(adata_sub.obs.x_centroid_px.max()).astype('int')

img_new = np.ones((ymax + 1, xmax + 1))
img_new = img_new * -1

span = int(np.ceil(1*d_cell/pix_size))

# manual dilation
for i in range(df_img.shape[0]):
    xl = span; xr = span; yl = span; yr = span
    if df_img.x[i] < span:
        xl = 0
    if df_img.x[i] + span < img_new.shape[1]:
        xr = 0
    if df_img.y[i] < span:
        yl = 0
    if df_img.y[i] + span < img_new.shape[0]:
        yr = 0
    img_new[df_img.y[i]-yl:df_img.y[i]+yr, df_img.x[i]-xl:df_img.x[i]+xr] =  1

if saveplot:
    plt.rcParams["figure.figsize"] = (10,10)
    plt.imshow(img_new, cmap='gray')
    plt.savefig('{}figures/img_dapi_p995_dilated_{}.pdf'.format(savepth, sid))
    plt.close()


""" Map dapi mask to cell in anndata """

span = int(np.ceil(2*d_cell/pix_size))

adata_sub.obs['dapi_mask'] = -1
for i in range(adata_sub.shape[0]):
    xid = np.floor(adata_sub.obs.x_centroid_px.iloc[i]).astype('int')
    yid = np.floor(adata_sub.obs.y_centroid_px.iloc[i]).astype('int')
    xl = span; xr = span; yl = span; yr = span
    if xid < span:
        xl = 0
    if xid + span < img_new.shape[1]:
        xr = 0
    if yid < span:
        yl = 0
    if yid + span < img_new.shape[0]:
        yr = 0
    tmp_array = img_new[yid-yl:yid+yr,xid-xl:xid+xr]
    if (tmp_array == 0).all():
        adata_sub.obs['dapi_mask'].iloc[i] = -1
    else: 
        uniq, counts = np.unique(tmp_array[tmp_array.nonzero()], return_counts=True)
        adata_sub.obs['dapi_mask'].iloc[i] = uniq[np.argmax(counts)]

tmp_db = adata_sub.obs.dapi_mask.value_counts() 
list_lowcell = tmp_db[tmp_db < 5].index
adata_sub.obs.loc[adata_sub.obs['dapi_mask'].isin(list_lowcell),'dapi_mask'] = -1


if saveplot:
    plt.rcParams["figure.figsize"] = (20,10)
    plt.imshow(image, cmap='gray')
    fig = sns.scatterplot(
            x = 'x_centroid_px', y = 'y_centroid_px',
            data = adata_sub.obs[adata_sub.obs.dapi_mask == 1],
            hue = 'dapi_mask', 
            s = 2, alpha = 0.9
        )
    plt.axis('scaled')
    plt.savefig('{}figures/img_dapi_p995_dilated_cell_{}.pdf'.format(savepth, sid))
    plt.close()


""" Create EPCAM and ACTA2 mask based on gene expression """
p95_acta2 = np.percentile(adata[:,'ACTA2'].X.toarray(), 95)
p95_acta2

p75_epcam = np.percentile(adata[:,'EPCAM'].X.toarray(), 75)
p75_epcam

adata_sub.obs['ACTA2_mask'] = 0
cids = adata_sub[adata_sub[:,'ACTA2'].X > p95_acta2].obs.index
adata_sub.obs.loc[cids, 'ACTA2_mask'] = 1

adata_sub.obs['EPCAM_mask'] = 0
cids = adata_sub[adata_sub[:,'EPCAM'].X > p75_epcam].obs.index
adata_sub.obs.loc[cids, 'EPCAM_mask'] = 1


""" Filter out cells based on the masks """
adata_mask = adata_sub[(adata_sub.obs.ACTA2_mask != 1) & (adata_sub.obs.EPCAM_mask == 1) & (adata_sub.obs.dapi_mask == 1)]

if saveplot:
    plt.rcParams["figure.figsize"] = (10,5)
    sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False})
    # sns.set_palette(cmap2(np.linspace(0,1,cmap2.N)))
    # plt.imshow(image, cmap='gray')
    fig = sns.scatterplot(
            x = 'x_centroid', y = 'y_centroid',
            data = adata_mask.obs,
            hue = 'EPCAM_mask', 
            s = 5, alpha = 0.9, palette=["white"]
        )
    try:
        fig.legend_.remove()
    except:
        pass
    plt.axis('scaled')
    plt.savefig('{}figures/spatial_dapi_epi_mask_{}.pdf'.format(savepth, sid))
    plt.close()


""" Clustering using DBSCAN """
adata_sub_epi = adata_mask.copy()
db = DBSCAN(eps=d_cell*n_cell, min_samples=5).fit(adata_sub_epi.obs[['x_centroid', 'y_centroid']])
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


adata_sub_epi.obs['dbscan_cluster'] = labels
adata_sub_epi.obs['dbscan_cluster'] = adata_sub_epi.obs['dbscan_cluster'].astype('category')


# Plot
if saveplot:
    plt.rcParams["figure.figsize"] = (10,5)
    sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False})
    # sns.set_palette(cmap2(np.linspace(0,1,cmap2.N)))
    # plt.imshow(image, cmap='gray')
    fig = sns.scatterplot(
            x = 'x_centroid', y = 'y_centroid',
            data = adata_sub_epi.obs[adata_sub_epi.obs.dbscan_cluster != -1],
            hue = 'dbscan_cluster', 
            s = 5, alpha = 0.9, palette="Set1"
        )
    try:
        fig.legend_.remove()
    except:
        pass
    plt.axis('scaled')
    plt.savefig('{}figures/spatial_dbscan_orig_{}.pdf'.format(savepth, sid))
    plt.close()



adata_sub.obs['acini_luminal'] = -1
adata_sub.obs.loc[adata_sub_epi.obs.index, 'acini_luminal'] = adata_sub_epi.obs['dbscan_cluster'].astype('int')


""" fill gaps (dilation) """
if n_cell_dil > 0:
    adata_sub.obs['acini_afterdil'] = -1
    adata_sub.obs.loc[adata_sub_epi.obs.index, 'acini_afterdil'] = adata_sub_epi.obs['dbscan_cluster'].astype('int')
    
    dbs_clusters = adata_sub.obs['acini_afterdil'].unique()[1:]
    
    cells_notasg = adata_sub.obs.loc[adata_sub.obs['acini_afterdil'] == -1,['x_centroid', 'y_centroid']]
    
    for cl in dbs_clusters:
        tmp = adata_sub.obs.loc[adata_sub.obs['acini_afterdil'] == cl, ['x_centroid', 'y_centroid']]
        for cell in tmp.index:
            df_dist = ((cells_notasg.x_centroid - tmp.loc[cell].x_centroid) ** 2 + (cells_notasg.y_centroid - tmp.loc[cell].y_centroid)**2).pow(.5) 
            tmp_notasg = df_dist[df_dist < n_cell_dil*d_cell]
            
            if len(tmp_notasg) != 0:
                adata_sub.obs.loc[tmp_notasg.index, 'acini_afterdil'] = cl
                
        
    if saveplot:
        groups = adata_sub.obs.groupby('acini_afterdil')
        plt.rcParams["figure.figsize"] = (10,10)
        # Plot
        fig, ax = plt.subplots()
        # ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            if name == -1:
                ax.plot(group.x_centroid, group.y_centroid, 
                    marker='.', linestyle='', markersize=0.5, color='k',
                    label=name)
            else:
                ax.plot(group.x_centroid, group.y_centroid, 
                        marker='o', linestyle='', markersize=1,
                        label=name)
        # ax.legend()
        plt.axis('scaled')
        
        plt.savefig('{}figures/spatial_dbscan_dilated_{}.pdf'.format(savepth, sid))
        plt.close()


adata_sub.obs[['dapi_mask', 'ACTA2_mask', 'EPCAM_mask', 'acini_luminal', 'acini_afterdil']].to_csv('{}/{}.csv'.format(savepth, sid))

print('all done')




