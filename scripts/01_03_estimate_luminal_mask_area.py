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

""" This script estimate mask area for each segmented acini based on transcripts densitiy at acini """

# sid; sample id. This script runs per sample. 
sid = sys.argv[1]

# plot if True
plotit = False

def compute_density_image(points, pspan=10, bwvalue=0.1):
    """ compute transcripts density """
    # points; xy coordinates of transcripts to compute density
    # pspan; pixels for extra area
    # bwvalue; kde.factor in gaussian_kde function
    xmax, ymax = [int(a) + pspan for a in points.max(axis=0).round()]
    xmin, ymin = [int(a) - pspan for a in points.min(axis=0).round()]

    points_use = points.copy()
    points_use['x_centroid_px'] = points['x_centroid_px']-xmin
    points_use['y_centroid_px'] = points['y_centroid_px']-ymin
    xsize = (xmax-xmin+1)
    ysize = (ymax-ymin+1)

    X, Y = np.mgrid[0:xsize,0:ysize]

    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = scipy.stats.gaussian_kde(points_use.transpose(), bw_method=bwvalue)
    density_image = kernel(positions).T.reshape((xsize,ysize)).T 

    plt.rcParams["figure.figsize"] = (6,6)
    plt.imshow(density_image, cmap='gray')    
    return points_use, density_image, xmin, xmax, ymin, ymax

def img_extraspace(a,d_buff):
    """ return image with extra space """
    # a; image
    # d_buff; pixels for extra area at each side
    p,q = a.shape
    out = np.zeros((2*d_buff + p,2*d_buff + q),dtype=a.dtype)
    out[d_buff:p + d_buff,d_buff:q + d_buff] = a
    return out


# save all coordinates in luminal mask
""" Set output path """
savepth = '../results/mask_coords_area/'
# create directory if it does not exist
p = Path(savepth)
p.mkdir(parents=True, exist_ok=True)


""" Set parameters """ 
# pix_size; pixel size per um. This depends on the cropped tif image resolution. 
# In this study, we used the highest resolution (level 0), thus pixel size is 0.2125 um. 
pix_size = 0.2125
pix_area = pix_size ** 2 # area per pixel

# extra_d; pixels for extra space
extra_d = 1000

# define kernel for dilation of density mask
dinit = 50
kernel = np.ones((dinit,dinit))

""" Read files """
# read anndata
f_adata = '../data/adata_acini.h5ad'
# '/common/yangy4/xenium/tunji_rankl/adatas/adata_merged_all_scanvi_scrublet_mapcoords2_rankgene_acinimask_dist_ncell20_clean.h5ad'
adata = sc.read(f_adata)

# read pixel index file
df_idx = pd.read_csv('../data/imgs_crop_level0/{}_index.csv'.format(sid), index_col=0)

# read transcripts file
fn_baysor = '../data/baysor_xen/{}/segmentation.csv'.format(sid)
transcripts_df = pd.read_csv(fn_baysor,
                             usecols=["cell",
                                      "x","y"])
transcripts_df = transcripts_df[~transcripts_df.cell.isna()].copy()
transcripts_df['x_centroid_px'] = transcripts_df['x']/pix_size - df_idx.idx_px.xmin
transcripts_df['y_centroid_px'] = transcripts_df['y']/pix_size - df_idx.idx_px.ymin
transcripts_df['cid'] = transcripts_df.cell.str.split('-', n=1, expand=True)[1]

if plotit:
    fn_img = glob.glob('imgs_crop_level0/{}.tif'.format(sid))[0]
    image = tff.imread(fn_img)

# subset anndata for cells in acini for a given sample
adata_sub = adata[(adata.obs.Batch == sid) & (adata.obs.acini_luminal != '-1')].copy()

# save all coordinates that in acini mask
df_coords_all = []
df_coords_er_all = []
df_area = []
for aid in adata_sub.obs.acini_luminal.unique():
    cids = adata_sub[adata_sub.obs.acini_luminal == aid].obs.index.str.replace('cell_','').str.replace('-{}'.format(sid),'')
    points = transcripts_df[transcripts_df.cid.isin(cids)][['x_centroid_px','y_centroid_px']].copy()
    points_use, density_image, xmin, xmax, ymin, ymax = compute_density_image(points, pspan=100)

    lum_mask = density_image > np.percentile(density_image[density_image != 0], 90)

    lum_mask_tmp = img_extraspace(lum_mask,extra_d)

    # fill gaps by dilation followed by erosion with 10 iterations
    dilated_mask = ndimage.binary_dilation(lum_mask_tmp, kernel, iterations=10)
    eroded_mask = ndimage.binary_erosion(dilated_mask, kernel, iterations=10)
    
    if plotit:
        plt.imshow(lum_mask, cmap='gray')
        fig = sns.scatterplot(
                x = 'x_centroid_px', y = 'y_centroid_px',
                data = points_use,\
                s = 9, alpha = 0.9, palette="black"
            )

    coords_er = np.argwhere(eroded_mask)
    df_coords_er = pd.DataFrame(coords_er)
    df_coords_er.columns = ['y', 'x']
    df_coords_er['x'] = df_coords_er['x'] + xmin  - extra_d
    df_coords_er['y'] = df_coords_er['y'] + ymin  - extra_d

    df_coords_er['aid'] = aid
    if len(df_coords_er_all) == 0:
        df_coords_er_all = df_coords_er.copy()
    else:
        df_coords_er_all = pd.concat([df_coords_er_all, df_coords_er])

    coords = np.argwhere(lum_mask)
    df_coords = pd.DataFrame(coords)
    df_coords.columns = ['y', 'x']
    df_coords['x'] = df_coords['x'] + xmin
    df_coords['y'] = df_coords['y'] + ymin
    
    if plotit:
        plt.imshow(image, cmap='gray')
        fig = sns.scatterplot(
                x = 'x', y = 'y',
                data = df_coords,
                # hue = 'dbscan_cluster', 
                s = 9, alpha = 0.9, palette="black"
            )
        plt.xlim(xmin-100,xmax+100)
        plt.ylim(ymax+100,ymin-100)

    df_coords['aid'] = aid
    if len(df_coords_all) == 0:
        df_coords_all = df_coords.copy()
    else:
        df_coords_all = pd.concat([df_coords_all, df_coords])

    

    if len(df_area) == 0:
        df_area = pd.DataFrame({aid: np.sum(lum_mask) * pix_area},index=['area'])
    else:
        df_area[aid] = np.sum(lum_mask) * pix_area



# df_coords_all.to_csv('../results/mask_coords_area/lummask_coords_{}.csv'.format(sid))
# save all coordinates in luminal mask after dilation/erosion (gaps were filled)
df_coords_er_all.to_csv('../results/mask_coords_area/filled_lummask_coords_{}.csv'.format(sid))
# save area for each acini
df_area.to_csv('../results/mask_coords_area/lummask_area_{}.csv'.format(sid))

print('done')
