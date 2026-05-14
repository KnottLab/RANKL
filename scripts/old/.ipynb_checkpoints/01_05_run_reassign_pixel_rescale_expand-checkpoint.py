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


def coords2img(points, pspan, extra_d=0, rescaling=0, xy=[0,0]):
    """ transform xy coordinates to image array """
    # points; xy coordinates of transcripts to compute density
    # pspan; pixels for extra area
    # extra_d; if you need additional extra space, pixels for extra area at each side
    # rescaling: rescale factor to reduce image size for rapid processing 
    # xy: list of xmin, ymin, e.g., [xmin, ymin]

    xmax, ymax = [int(a) + pspan for a in points.max(axis=0)[['x','y']].round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0)[['x','y']].round()]

    if xy != [0,0]:
        xmin, ymin = xy

    if rescaling < 1:
        xmax, xmin, ymax, ymin = [a - a%10 for a in [xmax, xmin, ymax, ymin]]
    # print(xmax, xmin, ymax, ymin)
    
    points_use = points.copy()
    points_use['x'] = points['x']-xmin
    points_use['y'] = points['y']-ymin
    xsize = (xmax-xmin+1)
    ysize = (ymax-ymin+1)
    
    X, Y = np.mgrid[0:xsize,0:ysize]

    dict_aid2int = {a:i+1 for i, a in enumerate(points.aid.unique())}
    
    img = np.zeros((points_use['y'].max()+1,points_use['x'].max()+1))
    img[points_use['y'], points_use['x']] = points_use.aid.map(dict_aid2int)

    if extra_d != 0:
        img = img_extraspace(img, extra_d)

    if rescaling != 0:
        img = transform.rescale(img, rescaling, anti_aliasing=False, order=0)
        xmin, xmax, ymin, ymax = [int(float(a)*rescaling) for a in [xmin, xmax, ymin, ymax]]
        

    return img, xmin, xmax, ymin, ymax, dict_aid2int

def rescale_coords(df_coords, rescaling=0.1, byaid=False, xy=[0,0]):
    """ rescale xy coordinates """
    # df_coords: xy coordinates
    # rescaling: rescaing factor
    # byaid: if True, rescale per acini
    # xy: list of xmin, ymin, e.g., [xmin, ymin]
    
    if byaid:
        df_coords_rescaled = []
        for a in df_coords.aid.unique():
            tmp = df_coords.query('aid == @a')
            
            # luminal mask coords to img
            img, xmin, xmax, ymin, ymax, dict_aid2int = coords2img(tmp, 0, rescaling=rescaling)
        
            coords = np.argwhere(img)
            df_coords_rescaled_tmp = pd.DataFrame(coords)
            df_coords_rescaled_tmp.columns = ['y', 'x']
            df_coords_rescaled_tmp['aid'] = a
            
            df_coords_rescaled_tmp['x'] = df_coords_rescaled_tmp['x'] + xmin
            df_coords_rescaled_tmp['y'] = df_coords_rescaled_tmp['y'] + ymin 

            if len(df_coords_rescaled) == 0:
                df_coords_rescaled = df_coords_rescaled_tmp.copy()
            else:
                df_coords_rescaled = pd.concat([df_coords_rescaled, df_coords_rescaled_tmp])
    else:    
        # luminal mask coords to img
        img, xmin, xmax, ymin, ymax, dict_aid2int = coords2img(df_coords, 0, rescaling=rescaling, xy=xy)
    
        dict_int2aid = dict(map(reversed, dict_aid2int.items()))
    
        coords = np.argwhere(img)
        df_coords_rescaled = pd.DataFrame(coords)
        df_coords_rescaled.columns = ['y', 'x']
        df_coords_rescaled['aid'] = img[df_coords_rescaled['y'], df_coords_rescaled['x']].astype('int')
        df_coords_rescaled['aid'] = df_coords_rescaled.aid.map(dict_int2aid)
        
        df_coords_rescaled['x'] = df_coords_rescaled['x'] + xmin
        df_coords_rescaled['y'] = df_coords_rescaled['y'] + ymin 

    return df_coords_rescaled, img, xmin, ymin


def img_extraspace(a,d_buff):
    """ return image with extra space """
    # a; image
    # d_buff; pixels for extra area at each side
    
    p,q = a.shape
    out = np.zeros((2*d_buff + p,2*d_buff + q),dtype=a.dtype)
    out[d_buff:p + d_buff,d_buff:q + d_buff] = a
    return out

def progressbar(current_value, total_value, bar_lengh=30, progress_char='■', binname=''): 
    """ show progress bar to track the status """
    percentage = int((current_value/total_value)*100)                                                # Percent Completed Calculation 
    progress = int((bar_lengh * current_value ) / total_value)                                       # Progress Done Calculation 
    loadbar = "Progress {}: [{:{len}}]{}%".format(binname, progress*progress_char, percentage, len = bar_lengh)  # Progress Bar String
    print(loadbar, end='\r')                                                                         # Progress Bar Output


""" The pre-assigned pixels for each acini can be overlapped each other, 
    so this script performs reassignment for those mask. 
    First, it finds overlapping region, 
    then each pixel in overlapping region is reassigned to the nearest acini.  """

    
# sid: sample id. This script runs per sample per acini mask 
# colnm: acini mask name e.g., acini_luminal, acini_dist4, etc. 
sid = sys.argv[1]
colnm = sys.argv[2]


""" Set parameters """ 
# pix_size; pixel size per um. This depends on the cropped tif image resolution. 
# In this study, we used the highest resolution (level 0), thus pixel size is 0.2125 um. 
pix_size = 0.2125
pix_area = pix_size ** 2 # area per pixel

# extra_d; pixels for extra space
extra_d = 0

# rescaling factor
gamma = 0.5

# save all coordinates in luminal mask
""" Set output path """
savepth =  '../results/dfs_newlyassigned_rescaled{}x/'.format(int(1/gamma))
# create directory if it does not exist
p = Path(savepth)
p.mkdir(parents=True, exist_ok=True)

   


plotit = False

# Read pre-assigned masks
fn_mask = '../results/mask_coords_area/lummask_coords_expanded_{}.csv'.format(sid) # masks for spatial bins
fn_fill = '../results/mask_coords_area/filled_lummask_coords_{}.csv'.format(sid) # acini mask

if os.path.isfile(fn_fill):
    # Read pre-assigned masks
    df_coords_all = pd.read_csv(fn_mask, index_col=0)
    df_coords_lum = pd.read_csv(fn_fill, index_col=0)

    print('all files are loaded.')

    # set key for each pixel 
    df_coords_lum['key'] = df_coords_lum['x'].astype('str') + '_' + df_coords_lum['y'].astype('str')
    df_coords_all['key'] = df_coords_all['x'].astype('str') + '_' + df_coords_all['y'].astype('str')

    # mask without any pixels luminal mask
    df_coords_all_nolum = df_coords_all[~df_coords_all.key.isin(df_coords_lum.key)].copy()

    # list unique mask names
    colnms = df_coords_all['mask'].unique()

    # set previous mask name 
    j = np.where(colnms == colnm)[0][0]
    colnms_pre = colnms[0:j]
    
    d_ncell = int(re.findall('\d+', colnm)[0])
    new_colnm = 'mask_ncell{}'.format(int(d_ncell))
    
    key_bins_pre = df_coords_all_nolum[df_coords_all_nolum['mask'].isin(colnms_pre)]['key'].unique()
    df_coords_bin = df_coords_all_nolum.query('mask == @colnm')[['x','y','aid','key']]
    
    # mask without any pixels in previous mask
    df_coords_bin_nopre = df_coords_bin[~df_coords_bin.key.isin(key_bins_pre)].copy()
    
    # luminal mask coords to img
    df_coords_lum_rescaled, img_lum, xmin, ymin = rescale_coords(df_coords_lum, rescaling=gamma)
    
    # find boundary of luminal masks:
    img_b = segmentation.find_boundaries(img_lum, mode='inner')
    
    # boundary img to xy coordinates
    coords = np.argwhere(img_b)
    df_coords = pd.DataFrame(coords)
    df_coords.columns = ['y', 'x']
    
    df_coords['x'] = df_coords['x'] + xmin
    df_coords['y'] = df_coords['y'] + ymin 
    
    # label 'aid' transfer to df_coords from df_coords_lum
    df_coords = df_coords.merge(df_coords_lum_rescaled, how='left')

    
    df_coords_bin_rescaled, img_bin, xmin_bin, ymin_bin = rescale_coords(df_coords_bin_nopre, rescaling=gamma, byaid=True)
    xmin_bin, ymin_bin = [int(a) for a in df_coords_bin_nopre.min(axis=0)[['x','y']].round()]
    df_coords_bin_rescaled['key'] = df_coords_bin_rescaled['x'].astype('str') + '_' + df_coords_bin_rescaled['y'].astype('str')
    
    df_coords_bin_dup_new = []
    # unique duplicated coordinates 
    df_coords_bin_dup = df_coords_bin_rescaled[df_coords_bin_rescaled[['key']].duplicated()]
    # any duplicated coordinates
    df_coords_bin_dup_all = df_coords_bin_rescaled[df_coords_bin_rescaled[['key']].duplicated(keep=False)]

    i = 1
    n_iter = len(df_coords_bin_dup.key.unique())
    for xy in df_coords_bin_dup.key.unique():
        tmp = df_coords_bin_dup_all.query('key == @xy')
        aids = tmp.aid.tolist()
        x, y = tmp.iloc[0][['x','y']]
        
        tmp_coords = df_coords[df_coords.aid.isin(aids)]
        xy_array = np.array(tmp_coords[['x','y']])
    
        id_min = np.argmin(spatial.distance.cdist(xy_array, [[x,y]]), axis=0)
    
        tmp_df = pd.DataFrame({'x':x, 'y':y, 'key': xy, 'aid': tmp_coords.iloc[id_min].aid.values[0]}, index=[0])
        if len(df_coords_bin_dup_new) == 0:
            df_coords_bin_dup_new = tmp_df
        else:
            df_coords_bin_dup_new = pd.concat([df_coords_bin_dup_new, tmp_df])
    
        if (i % 1000 == 0) or (i == n_iter):
            progressbar(i,n_iter, binname=new_colnm)
        i = i+1



    # no duplicated coordinates
    df_coords_bin_nodup =  df_coords_bin_rescaled[~df_coords_bin_rescaled[['key']].duplicated(keep=False)].copy()

    # combine no duplicated coordinates with newly assigned duplicated coordinates
    df_coords_bin_new = pd.concat([df_coords_bin_nodup[['x','y','key','aid']], df_coords_bin_dup_new])

    xmin_bin, ymin_bin = [int(a) for a in df_coords_bin_nopre.min(axis=0)[['x','y']].round()]
    xmin_bin, ymin_bin = [a - a%10 for a in [xmin_bin, ymin_bin]]
    xmin_bin, ymin_bin  = [int(float(a)*gamma) for a in [xmin_bin, ymin_bin]]

    df_coords_bin_new_rere, img_rere, xmin2, ymin2 = rescale_coords(df_coords_bin_new.dropna(), rescaling=1/gamma, xy=[xmin_bin, ymin_bin])
    df_coords_bin_new_rere.dropna(inplace=True)
    fnout = '{}/coords_{}_{}.csv'.format(savepth, sid, new_colnm)    
    df_coords_bin_new_rere.to_csv(fnout)

    df_area = pd.DataFrame(df_coords_bin_new_rere.groupby('aid').size()*pix_area)
    df_area.columns = [new_colnm]
    fnout_a = '{}/area_{}_{}.csv'.format(savepth, sid, new_colnm)    
    df_area.to_csv(fnout_a)


    if plotit:
        fig = sns.scatterplot(
            x = 'x', y = 'y',
            data = df_coords_bin_nopre,
            # hue = '',
            s = .1, alpha = 0.2, linewidth=0
        )
        
        fig = sns.scatterplot(
            x = 'x', y = 'y',
            data = df_coords_bin_new_rere,
            hue = 'aid',
            s = .1, alpha = 0.2, linewidth=0
        )
        plt.axis('scaled')
        plt.legend('')
        plt.show()

    print('\n')


print('done')



