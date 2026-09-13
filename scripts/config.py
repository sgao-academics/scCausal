"""
scCausal data path configuration.
All scripts import from this module to locate external datasets.
Reviewers: place datasets in a single directory and set SC_CAUSAL_DATA
environment variable, or use the default ~/scCausal_data/.
"""
import os
import json

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
    # human STRING v11 (taxid 9606) -- used for the PBMC network
    'string_aliases': 'string_aliases.txt.gz',
    'string_ppi': 'string_ppi_full.txt.gz',
    # mouse STRING v11 (taxid 10090) -- required for Paul15, which is mouse
    # data and returns zero hits when scored against human STRING
    'string_aliases_mouse': '10090.string_aliases.txt.gz',
    'string_ppi_mouse': '10090.string_ppi_full.txt.gz',
    'trrust': 'trrust_human.tsv',
    'depmap_crispr': 'CRISPR_gene_effect.csv',
    # Perturb-seq screens used for the interventional validation (Table 8).
    # Both are distributed by the scverse example-data archive.
    'norman_h5ad': 'norman_2019_raw.h5ad',
    'replogle_h5ad': 'replogle_2022_k562_essential.h5ad',
}

def get_path(dataset_key):
    """Return absolute path to a dataset file.
    Valid keys: paul15, pbmc3k_h5ad, string_aliases, string_ppi,
    string_aliases_mouse, string_ppi_mouse, trrust, depmap_crispr
    """
    data_dir = get_data_dir()
    fname = DATASETS.get(dataset_key, dataset_key)
    return os.path.join(data_dir, fname)

def resolve(dataset_key, env_var=None):
    """Resolve an external dataset path.

    Priority:
      1. the named environment variable, if set;
      2. an entry in scripts/local_paths.json, a private convenience file that
         is deliberately absent from the public package and is only used when
         the file it points at actually exists;
      3. the configured data directory (see get_path).

    Step 2 exists so that a local checkout can keep its datasets wherever they
    already live without any path appearing in the published source.
    """
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'local_paths.json')
    if os.path.exists(local):
        try:
            with open(local, encoding='utf-8') as fh:
                mapping = json.load(fh)
            p = mapping.get(dataset_key)
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return get_path(dataset_key)


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

  3. STRING PPI v11 (mouse) -- required for the Paul15 cross-tissue validation:
     Download: https://string-db.org/cgi/download
     Files needed:
       - 10090.protein.aliases.v11.0.txt.gz
         Save as: {os.path.join(d, '10090.string_aliases.txt.gz')}
       - 10090.protein.links.v11.0.txt.gz
         Save as: {os.path.join(d, '10090.string_ppi_full.txt.gz')}

     Note: the links files have 16 columns and the combined confidence score is
     the LAST column; an earlier filter keyed on the neighborhood channel
     (column 3) silently returns zero validated edges.

  4. TRRUST v2 (human):
     Download: https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv
     Save as: {os.path.join(d, 'trrust_human.tsv')}

  5. DepMap 23Q4 CRISPR:
     Download: https://depmap.org/portal/download/all/
     File: CRISPR_gene_effect.csv
     Save as: {os.path.join(d, 'CRISPR_gene_effect.csv')}

  6. Perturb-seq screens (only needed for the interventional validation of
     Table 8; scripts/norman_validation.py and scripts/perturbseq_validation.py
     read the screen with --h5 and default to this directory):
     Download: https://exampledata.scverse.org/pertpy/norman_2019_raw.h5ad
       Save as: {os.path.join(d, 'norman_2019_raw.h5ad')}
     Download: https://exampledata.scverse.org/pertpy/replogle_2022_k562_essential.h5ad
       Save as: {os.path.join(d, 'replogle_2022_k562_essential.h5ad')}

     Note: the observation arm and the gold standard are both derived from the
     screen itself -- non-targeting control cells give the skeleton, the
     single-gene perturbations give the intervention -- so no label file is
     needed and the guide columns already present in the h5ad are sufficient.

PBMC 3K ships with this package (data/pbmc3k_filtered.h5ad); it does not need
manual setup. The Reactome GMT (data/ReactomePathways.gmt) is also included.
========================================
""")
