# RANKL — Supplementary Analysis Scripts

Supplementary scripts for the manuscript:

> **Breast Tissue Changes following RANKL Inhibition with Denosumab in Healthy Premenopausal Women with Dense Breasts**

Example data and result files are available in a separate repository:
https://doi.org/10.5281/zenodo.14511008

---

## Repository Structure

```
scripts/
├── 01_*   Xenium spatial transcriptomics: acini segmentation, spatial binning, area estimation
├── 02_*   H&E slide annotation and alignment to spatial metabolomics
├── 03_*   MALDI-MSI spatial metabolomics: QC, differential analysis, pathway enrichment
├── 04_*   BODIPY/Perilipin adipocyte staining: lipid droplet segmentation and quantification
├── 05_*   Multiplex immunofluorescence (Lunaphore): pixel-level marker binarization, ECM analysis
└── doit_* Shell wrappers to run parallelisable scripts per sample
```

---

## 01 — Xenium Acini Segmentation, Spatial Binning, and Area Estimation

These scripts identify mammary acini (glandular units) from Xenium spatial
transcriptomics data, assign cells to concentric periductal distance bins
around each acinus, and estimate the tissue area of each bin.

Scripts 01_01 and 01_03–01_06 are run **per sample** and are parallelisable
via the `doit_*.sh` wrappers. Scripts 01_02 and 01_07 consolidate all
per-sample results into a single AnnData object.

### Files

| File | Type | Description |
|------|------|-------------|
| `01_01_acini_segmentation.py` | Python script | Segment acini using DBSCAN on DAPI+ EPCAM+ ACTA2− epithelial cells; save per-cell cluster assignments |
| `01_02_transfer_acini_to_anndata.ipynb` | Notebook | Merge per-sample acini CSVs into the AnnData object |
| `01_03_estimate_luminal_mask_area.py` | Python script | Estimate luminal mask area per acinus using Gaussian KDE of transcript density |
| `01_04_spatial_bin_preassignment.py` | Python script | Expand luminal masks into concentric distance rings (pre-assignment; may overlap) |
| `01_05_run_reassign_pixel_in_overlapping_bins.py` | Python script | Resolve overlap between rings from adjacent acini; reassign each pixel to the nearest acinus |
| `01_06_map_mask_to_cell.py` | Python script | Map resolved pixel masks to cell centroids in the AnnData object |
| `01_07_consolidate_masks.ipynb` | Notebook | Merge per-sample bin assignments and area estimates; save final AnnData and area table |
| `doit_01_01.sh` | Shell | Run `01_01` for one sample |
| `doit_01_03.sh` | Shell | Run `01_03` for one sample |
| `doit_01_04.sh` | Shell | Run `01_04` for one sample |
| `doit_01_05.sh` | Shell | Run `01_05` for one sample × one distance bin |
| `doit_01_06.sh` | Shell | Run `01_06` for one sample |

### Suggested execution order

```bash
# Step 1: Segment acini for each sample (parallelisable)
for sid in $(cat sample_list.txt); do bash doit_01_01.sh $sid & done

# Step 2: Merge per-sample results into AnnData
jupyter nbconvert --to notebook --execute 01_02_transfer_acini_to_anndata.ipynb

# Step 3: Estimate luminal mask area (parallelisable)
for sid in $(cat sample_list.txt); do bash doit_01_03.sh $sid & done

# Step 4: Pre-assign spatial bins (parallelisable)
for sid in $(cat sample_list.txt); do bash doit_01_04.sh $sid & done

# Step 5: Resolve overlapping pixels (parallelisable per sample × per bin)
for sid in $(cat sample_list.txt); do
  for colnm in acini_dist4 acini_dist8 acini_dist12 acini_dist16 acini_dist20; do
    bash doit_01_05.sh $sid $colnm &
  done
done

# Step 6: Map masks to cells (parallelisable)
for sid in $(cat sample_list.txt); do bash doit_01_06.sh $sid & done

# Step 7: Consolidate all results
jupyter nbconvert --to notebook --execute 01_07_consolidate_masks.ipynb
```

### Key parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pix_size` | 0.2125 µm | Pixel size at Xenium level-0 resolution |
| `d_cell` | 4.372 µm | Mean epithelial cell diameter (estimated from data) |
| `n_cell` | 6 | DBSCAN epsilon in cell-diameter units |
| `n_cell_dil` | 1 | Post-clustering gap-fill dilation radius |
| Spatial bins | 4–20 cells | Distance rings in 4-cell increments from acinus edge |

---

## 02 — H&E Slide Annotation and Alignment to Spatial Metabolomics

| File | Type | Description |
|------|------|-------------|
| `02_spmetabo_he_alignment.ipynb` | Notebook | Align H&E whole-slide images to MALDI-MSI pixel grids using SimpleITK rigid registration; project QuPath annotation masks (glands, ECM, adipose, blood vessel) onto metabolomics coordinates |

**Key dependencies:** `pyvips`, `SimpleITK`, `opencv-python`, `scikit-image`

---

## 03 — MALDI-MSI Spatial Metabolomics

Analysis of spatial metabolomics data (MALDI-MSI, positive and negative ion panels)
comparing Day 1 vs Day 60 post-treatment.

| File | Type | Description |
|------|------|-------------|
| `03_01_spatial_metabolomics_qc_de.ipynb` | Notebook | Full pipeline: METASPACE annotation (FDR < 0.05), normalization, ComBat batch correction, dimensionality reduction, H&E cell type annotation transfer, Wilcoxon DE per cell type, pseudobulk DESeq2 (Day60 vs Day1) |
| `03_02_pathway_enrichment.ipynb` | Notebook | WikiPathways enrichment (Fisher's exact test + BH-FDR) on Day60 vs Day1 DE metabolites; pathway redundancy analysis; LFC heatmaps; integration with bulk LC-MS lipidomics |

**Key dependencies:** `scanpy`, `anndata`, `pydeseq2`, `decoupler`, `metaspace-converter`

---

## 04 — BODIPY / Perilipin Adipocyte Staining

Lipid droplet quantification from a dedicated fluorescence staining experiment
(separate from the Lunaphore multiplex panel) using BODIPY (lipid content) and
Perilipin/PLIN1 (adipocyte membrane marker) on tissue sections imaged at 20×
(qPTIFF format).

| File | Type | Description |
|------|------|-------------|
| `04_lipid_droplet_quantification.ipynb` | Notebook | Segment lipid droplets via Otsu threshold → Perilipin-guided watershed; classify adipocytes by size filter + PLIN1 co-localization (≥10 px overlap); compute droplet count density and lipid area fraction per section; Day1 vs Day60 comparison |

**Key dependencies:** `scikit-image`, `tifffile`, `statannotations`

---

## 05 — Lunaphore Multiplex Immunofluorescence

Pixel-level analysis of multiplex immunofluorescence data (Lunaphore platform,
10 µm resolution) quantifying ECM remodeling and epithelial marker expression
across H&E-annotated tissue regions.

| File | Type | Description |
|------|------|-------------|
| `05_pixel_binarization_ecm_analysis.ipynb` | Notebook | Pixel-level marker binarization using GMM-based adaptive thresholding (`final_smart_threshold_ver3`: BIC model selection, per-region thresholding); periductal ECM area fraction analysis; ECM island collagen clustering |

**Key dependencies:** `scikit-image`, `tifffile`, `scikit-learn`, `statannotations`, `kmodes`

---

## Data Directories (expected layout)

```
project/
├── data/
│   ├── adata.h5ad                        # Xenium cell-level AnnData (input)
│   ├── adata_acini.h5ad                  # + acini assignments (01_02 output)
│   ├── adata_final.h5ad                  # + spatial bin assignments (01_07 output)
│   ├── adata_spmetabo_pos.h5ad           # MALDI-MSI positive panel
│   ├── adata_spmetabo_neg.h5ad           # MALDI-MSI negative panel
│   ├── imgs_crop_level0/                 # Cropped DAPI tiff images + index CSVs
│   ├── baysor_xen/{sid}/segmentation.csv # Baysor transcript-level output
│   ├── heslides/*.svs                    # H&E whole-slide images
│   ├── masks/*.png                       # QuPath annotation mask PNGs
│   └── bodipy_qtiffs/                    # Lunaphore qPTIFF images
├── results/
│   ├── dbscan_epi_dapi_cell/             # 01_01 outputs
│   ├── mask_coords_area/                 # 01_03, 01_04 outputs
│   ├── dfs_newlyassigned_rescaled2x/     # 01_05, 01_06 outputs
│   └── area_all.csv                      # 01_07 consolidated area table
├── logs/                                 # Per-sample log files from doit_*.sh
└── scripts/                              # This repository
```

---

## Dependencies

```
Python ≥ 3.10
anndata        scanpy         pydeseq2       decoupler
scikit-learn   scikit-image   scipy          numpy  pandas
matplotlib     seaborn        tifffile       pyvips
SimpleITK      opencv-python  statannotations kmodes
metaspace-converter            statsmodels
```
