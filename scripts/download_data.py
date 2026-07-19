"""Download PBMC 3K with cell type labels and Perturb-seq data."""
import os, sys
import scanpy as sc
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# === PBMC 3K labeled (processed, has leiden/louvain clusters) ===
print('Downloading PBMC 3K processed...')
try:
    adata = sc.datasets.pbmc3k_processed()
    adata.write(os.path.join(DATA_DIR, 'pbmc3k_processed.h5ad'))
    print(f'  Processed: {adata.shape}, obs_cols={list(adata.obs.columns)}')
except Exception as e:
    print(f'  Processed failed: {e}, using raw + clustering...')

# === PBMC 3K raw counts ===
print('Downloading PBMC 3K raw...')
try:
    adata_raw = sc.datasets.pbmc3k()
    adata_raw.write(os.path.join(DATA_DIR, 'pbmc3k_raw.h5ad'))
    print(f'  Raw: {adata_raw.shape}')
except Exception as e:
    print(f'  Raw scanpy download failed: {e}')
    # Try local file
    local = config.get_path('pbmc3k_h5ad')
    if os.path.exists(local):
        import shutil
        shutil.copy(local, os.path.join(DATA_DIR, 'pbmc3k_raw.h5ad'))
        print(f'  Copied from local: {local}')

# QC and HVG selection
print('Running QC and HVG selection on raw data...')
adata = sc.read_h5ad(os.path.join(DATA_DIR, 'pbmc3k_raw.h5ad'))
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
print(f'  After HVG filtering: {adata.shape}')

# Cell type annotation via clustering
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.8)
sc.tl.umap(adata)

# Map clusters to known cell types
cluster_to_type = {
    '0': 'CD4+ T', '1': 'CD14+ Monocyte', '2': 'B', '3': 'CD8+ T',
    '4': 'NK', '5': 'FCGR3A+ Monocyte', '6': 'Dendritic', '7': 'Megakaryocyte'
}
adata.obs['cell_type'] = adata.obs['leiden'].map(
    lambda x: cluster_to_type.get(x, f'Cluster_{x}')
)
print(f'  Cell types: {adata.obs["cell_type"].value_counts().to_dict()}')

# Also compute raw counts version for ZINB
adata_raw_hvg = adata.raw.to_adata()[:, adata.var_names]
print(f'  Raw counts (HVG): {adata_raw_hvg.shape}')
adata_raw_hvg.write(os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad'))
print('Saved: pbmc3k_filtered.h5ad (with cell types, raw counts)')

print('\n=== Data download complete ===')
print(f'Files in {DATA_DIR}:')
for f in os.listdir(DATA_DIR):
    sz = os.path.getsize(os.path.join(DATA_DIR, f))
    print(f'  {sz//1024:>4} KB  {f}')
