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

""" This script performs pre-assignment of spatial bin by expanding bin with 4-cell distance. 
At this point, we don't care overlapping area from multiple acini. This will be handled next step. """

def img_extraspace(a,d_buff):
    """ return image with extra space """
    # a; image
    # d_buff; pixels for extra area at each side
    p,q = a.shape
    out = np.zeros((2*d_buff + p,2*d_buff + q),dtype=a.dtype)
    out[d_buff:p + d_buff,d_buff:q + d_buff] = a
    return out

def coords2img(points, pspan, extra_d=0):
    """ transform xy coordinates to image array """
    # points; xy coordinates of transcripts to compute density
    # pspan; pixels for extra area
    # extra_d; if you need additional extra space, pixels for extra area at each side
    
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0).round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0).round()]
    
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

    return img, xmin, ymin

def pix2realloc(df_coords, xmin, ymin, extra_d=0):
    """ shift coordinates to align with the expanded image """
    df_coords['x_centroid_px'] = df_coords['x_centroid_px'] - xmin + extra_d
    df_coords['y_centroid_px'] = df_coords['y_centroid_px'] - ymin + extra_d
    return df_coords
    
# sid; sample id. This script runs per sample. 
sid = sys.argv[1]


""" Set parameters """ 
# pix_size; pixel size per um. This depends on the cropped tif image resolution. 
# In this study, we used the highest resolution (level 0), thus pixel size is 0.2125 um. 
pix_size = 0.2125
pix_area = pix_size ** 2 # area per pixel

# pspan; pixels for extra area
pspan = 100

# extra_d; pixels for extra space
extra_d = 2000

# d_cell; one-cell distance (um). This was estimated by average diameter of epithelial cells in our dataset. 
d_cell = 4.371621916951665

# define kernel for dilation of density mask. We expanded mask at every 4-cell diameter so it's 4*d_cell/pix_size 
dinit = int(np.ceil(2*d_cell/pix_size))


colnms = ['acini_luminal', 'acini_afterdil', 
          'acini_dist4', 'acini_dist8', 'acini_dist12', 'acini_dist16', 'acini_dist20']



""" Read files """
# read anndata
f_adata = '../data/adata_acini.h5ad'
# '/common/yangy4/xenium/tunji_rankl/adatas/adata_merged_all_scanvi_scrublet_mapcoords2_rankgene_acinimask_dist_ncell20_clean.h5ad'
adata = sc.read(f_adata)

# for other acini mask, use df_coords_er_all
df_coords_er_all = pd.read_csv('../results/mask_coords_area/filled_lummask_coords_{}.csv'.format(sid), index_col=0)


""" assign spatial bin by expanding mask from individual acini"""
df_area = []
df_coords_all = []
for aid in df_coords_er_all.aid.unique():
    points = df_coords_er_all.query('aid == @aid')[['x','y']]
    img, xmin, ymin = coords2img(points, pspan, extra_d)

    for i in range(1, len(colnms)):
        # colnm_pre = colnms[i-1]
        colnm = colnms[i]
        # print(colnm)
    
        if colnm == 'acini_afterdil':
            n_cell = 1
        else:
            n_cell = int(colnm.split('dist')[-1]) + 1
            
        dilated_mask = ndimage.binary_dilation(img, iterations=dinit*(n_cell))
    
        if colnm == 'acini_afterdil':
            donut_mask = dilated_mask.astype('int') - img
        else:
            donut_mask = dilated_mask.astype('int') - pre_mask
    
        pre_mask = dilated_mask.astype('int')
        
        # plt.rcParams["figure.figsize"] = (4,4)
        # plt.imshow(donut_mask, cmap='gray')
        # plt.show()
        
        coords = np.argwhere(donut_mask)
        df_coords = pd.DataFrame(coords)
        df_coords.columns = ['y', 'x']
        df_coords['x'] = df_coords['x'] + xmin - extra_d
        df_coords['y'] = df_coords['y'] + ymin - extra_d

        df_coords['aid'] = aid
        df_coords['mask'] = colnm

        if len(df_coords_all) == 0:
            df_coords_all = df_coords.copy()
        else:
            df_coords_all = pd.concat([df_coords_all, df_coords])

               
        if len(df_area) == 0:
            df_area = pd.DataFrame({aid: np.sum(donut_mask) * pix_area},index=[colnm])
        else:
            try:
                df_area.loc[colnm, aid] = np.sum(donut_mask) * pix_area
            except:
                df_area[colnm] = 0.
                df_area.loc[colnm, aid] = np.sum(donut_mask) * pix_area
    print(aid)

df_coords_all.to_csv('../results/mask_coords_area/lummask_coords_expanded_{}.csv'.format(sid))
df_area.to_csv('../results/mask_coords_area/lummask_area_expanded_{}.csv'.format(sid))

print('done')
