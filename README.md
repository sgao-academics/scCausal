# scCausal — Replication Package

> **Gao, S. (2026).** scCausal: Distribution-Matched Conditional Independence Testing for Single-Cell Causal Discovery. Submitted to *BMC Genomics*.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. PBMC-only reproduction (no external data needed)
python run_all.py --pbmc-only

# 3. For full reproduction: set data directory (any of these methods)
#    Option A: export SC_CAUSAL_DATA=/path/to/your/data
#    Option B: create external_data/ folder next to scripts/
#    Option C: defaults to ~/scCausal_data/

# 4. Download external datasets to your data directory
python run_all.py --download   # Shows download instructions with correct paths

# 5. Run everything
python run_all.py              # Full pipeline

# 6. View results
ls results/          # JSON checkpoints
ls figures/          # Fig1-4 PDFs
```

## Validation Dimensions

scCausal is validated across five independent dimensions:

| Dimension | Script | Validation Target |
|:--|:--|:--|
| 1. STRING PPI | `run_pbmc.py`, `run_paul.py` | Biological plausibility (>=700 combined score) |
| 2. Reactome Pathway | `_reactome_validate.py` | Literature-curated mechanistic evidence (OR=4.01, p=0.028) |
| 3. CRISPR Co-Essentiality | `_depmap_coessential.py` | Perturbation-based causal evidence (Fisher p<0.0001) |
| 4. Synthetic Benchmarks | `run_synthetic.py` | Known ground truth (100 seeds, d=30/50/100) |
| 5. Permuted Controls | `run_pbmc.py` | Zero false positive verification |

## Package Contents

```
scCausal_Replication/
├── run_all.py                       # One-command reproduction runner
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/
│   ├── pbmc3k_filtered.h5ad         # PBMC 3K counts (5.1 MB)
│   └── pbmc3k_preprocessed.npz      # Preprocessed matrix
├── figures/                         # Pre-generated figures (4 PDFs + 4 PNGs)
│   ├── fig1_pipeline.pdf/png        #   Pipeline overview
│   ├── fig2_benchmark.pdf/png       #   Main benchmark (3x3 panel)
│   ├── fig3_network.pdf/png         #   Gene regulatory network
│   └── fig4_validation.pdf/png      #   Validation & robustness (2x2)
├── results/                         # Experiment checkpoints
│   ├── _sweep_checkpoint.json       #   Primary sweep (d=30/50/100/200)
│   ├── _everything_ckpt.json        #   58 experiments, all configs
│   ├── _best_paper_ckpt.json        #   Paper-quality benchmark results
│   ├── _final_push_ckpt.json        #   Cell-type Fisher z results
│   ├── _remaining_ckpt.json         #   Remaining cell-type results
│   ├── _fix30_ckpt.json             #   Fixed d=30 variant
│   ├── _paul_ckpt.json              #   Paul15 experiments
│   ├── simulation_results.json      #   Synthetic benchmark (100 seeds)
│   ├── depmap_coessentiality.json   #   CRISPR co-essentiality data
│   └── baseline_results.json        #   Baseline comparison (NOTEARS/GENIE3)
└── scripts/
    ├── config.py                     #   Data path configuration (SC_CAUSAL_DATA)
    ├── sccausal_ci.py               #   NB-LR conditional independence test
    ├── sccausal_engine.py            #   PC algorithm with NB-LR
    ├── sccausal_ci_fast.py           #   Fast CI variant
    ├── sccausal_twostage.py          #   Two-stage variant
    ├── sccausal_v2.py                #   Experimental V2
    ├── sccausal_v3.py                #   Experimental V3
    ├── run_pbmc.py                   #   PBMC 3K main experiments
    ├── run_paul.py                   #   Paul15 cross-tissue validation
    ├── run_synthetic.py              #   Synthetic benchmark generator
    ├── run_baselines.py              #   NOTEARS/GENIE3 baselines
    ├── run_everything.py             #   Full configuration sweep
    ├── run_fix30.py                  #   Fixed d=30 + cell types
    ├── sweep_all.py                  #   Exhaustive parameter sweep
    ├── sweep_ci.py                   #   CI test comparison sweep
    ├── sweep_full.py                 #   Full sweep (all dimensions)
    ├── download_data.py              #   Dataset download utility
    ├── _depmap_coessential.py        #   DepMap CRISPR validation
    ├── _reactome_validate.py         #   Reactome pathway validation
    ├── _download_reactome.py         #   Reactome data download
    └── gen_figures/
        ├── gen_fig1.py               #   Pipeline overview figure
        ├── gen_fig2.py               #   Benchmark composite (3x3)
        ├── gen_fig3.py               #   Network visualization
        └── gen_fig4.py               #   Validation & robustness
```

## Requirements

- Python >= 3.9
- Dependencies: numpy, scipy, pandas, matplotlib, networkx, scanpy, statsmodels, scikit-learn, torch, tqdm, scienceplots
- External datasets (for full reproduction): Paul15, STRING v11, DepMap 23Q4, Reactome

## Citation

```bibtex
@article{gao2026sccausal,
  title={scCausal: Distribution-Matched Conditional Independence Testing for Single-Cell Causal Discovery},
  author={Gao, Shuaidong},
  year={2026},
  note={Submitted to BMC Genomics}
}
```

## License

MIT License. See manuscript Declarations for details.
