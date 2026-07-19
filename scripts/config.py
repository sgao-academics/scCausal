"""
scCausal data path configuration.
All scripts import from this module to locate external datasets.
Reviewers: place datasets in a single directory and set SC_CAUSAL_DATA
environment variable, or use the default ~/scCausal_data/.
"""
import os

_DEFAULT = os.path.join(os.path.expanduser('~'), 'scCausal_data')

def get_data_dir():
    """Return the scCausal external data directory.
    Priority: SC_CAUSAL_DATA env var > local external_data/ > ~/scCausal_data/
    """
    if 'SC_CAUSAL_DATA' in os.environ:
        return os.environ['SC_CAUSAL_DATA']
    local = os.path.join(os.path.dirname(__file__), '..', 'external_data')
    if os.path.isdir(local):
        return os.path.abspath(local)
    return _DEFAULT

def ensure_data_dir():
    """Create the data directory if it does not exist."""
    d = get_data_dir()
    os.makedirs(d, exist_ok=True)
    return d

# Dataset file paths (relative to data dir)
DATASETS = {
    'paul15': 'paul15.h5ad',
    'pbmc3k_h5ad': 'pbmc3k.h5ad',
    'string_aliases': 'string_aliases.txt.gz',
    'string_ppi': 'string_ppi_full.txt.gz',
    'trrust': 'trrust_human.tsv',
    'depmap_crispr': 'CRISPR_gene_effect.csv',
}

def get_path(dataset_key):
    """Return absolute path to a dataset file.
    Valid keys: paul15, string_aliases, string_ppi, trrust, depmap_crispr, pbmc3k_h5ad
    """
    data_dir = get_data_dir()
    fname = DATASETS.get(dataset_key, dataset_key)
    return os.path.join(data_dir, fname)

def print_download_instructions():
    """Print instructions for downloading external datasets."""
    d = get_data_dir()
    print(f"""
========================================
  EXTERNAL DATASET DOWNLOAD INSTRUCTIONS
  Data directory: {d}
========================================

All external datasets should be placed in the above directory.
Option 1: Set environment variable SC_CAUSAL_DATA to your data directory.
Option 2: Create a folder 'external_data' next to the scripts/ folder.

Required files:

  1. Paul15 hematopoietic differentiation:
     Download: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72857
     Save as: {os.path.join(d, 'paul15.h5ad')}

  2. STRING PPI v11 (human):
     Download: https://string-db.org/cgi/download
     Files needed:
       - 9606.protein.aliases.v11.0.txt.gz
         Save as: {os.path.join(d, 'string_aliases.txt.gz')}
       - 9606.protein.links.v11.0.txt.gz  
         Save as: {os.path.join(d, 'string_ppi_full.txt.gz')}

  3. TRRUST v2 (human):
     Download: https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv
     Save as: {os.path.join(d, 'trrust_human.tsv')}

  4. DepMap 23Q4 CRISPR:
     Download: https://depmap.org/portal/download/all/
     File: CRISPR_gene_effect.csv
     Save as: {os.path.join(d, 'CRISPR_gene_effect.csv')}

PBMC 3K is auto-downloaded by the pipeline and does not need manual setup.
========================================
""")
