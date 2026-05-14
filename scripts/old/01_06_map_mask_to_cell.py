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


def coords2img(points, pspan, extra_d=0):
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0)[['x','y']].round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0)[['x','y']].round()]
    
    points_use = points.copy()
    points_use['x'] = points['x']-xmin
    points_use['y'] = points['y']-ymin
    xsize = (xmax-xmin+1)
    ysize = (ymax-ymin+1)
    
    X, Y = np.mgrid[0:xsize,0:ysize]
    
    img = np.zeros((points_use['y'].max()+1,points_use['x'].max()+1))
    img[points_use['y'], points_use['x']] = 1

    if extra_d != 0:
        img = img_extraspace(img, extra_d)

    return img, xmin, xmax, ymin, ymax


def img_extraspace(a,d_buff):
    p,q = a.shape
    out = np.zeros((2*d_buff + p,2*d_buff + q),dtype=a.dtype)
    out[d_buff:p + d_buff,d_buff:q + d_buff] = a
    return out

# sid: sample id. This script runs per sample per acini mask 
# coords_pth: folder path for reassinged pixel coordinates
sid = sys.argv[1]
coords_pth = sys.argv[2]


""" Set parameters """ 
# pix_size; pixel size per um. This depends on the cropped tif image resolution. 
# In this study, we used the highest resolution (level 0), thus pixel size is 0.2125 um. 
pix_size = 0.2125
pix_area = pix_size ** 2 # area per pixel

# extra_d; pixels for extra space
extra_d = 0



""" Read anndata """
adata = sc.read('../data/adata_acini.h5ad')

# subset anndata for a given sample
adata_sub = adata[adata.obs.Batch == sid].copy()

# cell xy coordinates
df_coords_cell = adata_sub.obs[['y_centroid_px','x_centroid_px']].copy()

for d_ncell in range(4, 21, 4):
    new_colnm = 'mask_ncell{}'.format(d_ncell)
    fn_coords = '{}/coords_{}_{}.csv'.format(coords_pth, sid, new_colnm) 

    if os.path.isfile(fn_coords):
        df_coords_bin = pd.read_csv(fn_coords, index_col=0)
        df_coords_bin = df_coords_bin[~df_coords_bin[['x','y']].duplicated()]
        
        adata_sub.obs[new_colnm] = '-1'
    
        for aid in df_coords_bin.aid.unique():
            points = df_coords_bin.query('aid == @aid')[['x','y']]
            points['mask'] = 1
            
            img, xmin, xmax, ymin, ymax = coords2img(points, pspan=0, extra_d=extra_d)
            
            df_coords_cell_tmp = df_coords_cell.query('x_centroid_px >= @xmin & x_centroid_px <= @xmax & y_centroid_px >= @ymin & y_centroid_px <= @ymax')
            if df_coords_cell_tmp.shape[0] != 0:
                df_coords_cell_tmp.loc[:,'x_centroid_px'] = df_coords_cell_tmp['x_centroid_px'] - xmin + extra_d
                df_coords_cell_tmp.loc[:,'y_centroid_px'] = df_coords_cell_tmp['y_centroid_px'] - ymin + extra_d
                df_coords_cell_tmp = np.floor(df_coords_cell_tmp).astype('int') 
                
                df_coords_cell_tmp['mask'] = img[tuple(np.array(df_coords_cell_tmp.T))]
                cids = df_coords_cell_tmp.query('mask == 1').index
                
                cols = [a for a in adata_sub.obs.columns if 'mask_' in a]
                tmp = adata_sub.obs.loc[cids,cols] 
                tmp = tmp.astype('str')
                cids_no = tmp[~tmp.eq(tmp.iloc[:, 0], axis=0).all(1)].index
                cids = [a for a in cids if a not in cids_no]

                if len(cids) > 0:
                    adata_sub.obs.loc[cids, new_colnm] = aid
        

        print(new_colnm)


savecols = [a for a in adata_sub.obs.columns if 'mask_ncell' in a]
# save masks mapped to cells for a given sample
adata_sub.obs[savecols].to_csv('{}/adata_cols_expand_{}.csv'.format(coords_pth, sid))




print('done')



