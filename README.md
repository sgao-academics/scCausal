# scCausal — Replication Package

Reference implementation and reproduction pipeline for

> Gao, S. *scCausal: distribution-matched conditional independence testing for
> single-cell causal discovery.* Manuscript under review.

scCausal keeps the standard PC skeleton search but replaces Fisher's *z*
conditional-independence test with a negative binomial likelihood-ratio test
whose dispersion is estimated from the data by the method of moments. The point
is to match the test to the count distribution, rather than transforming the
data to fit the test.

All scripts are pure ASCII, use relative paths, and read external datasets
through `scripts/config.py` — see [External datasets](#external-datasets).

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Point the pipeline at your external datasets (any one of these)
#    export SC_CAUSAL_DATA=/path/to/your/data      (Linux/macOS)
#    set    SC_CAUSAL_DATA=D:\path\to\your\data    (Windows)
#    ...or create an  external_data/  folder next to scripts/
python run_all.py --download          # prints the exact expected file names

# 3. PBMC-only reproduction — no external datasets beyond PBMC 3K,
#    which ships inside this package
python run_all.py --pbmc-only

# 4. Full reproduction (also needs Paul15, human + mouse STRING, DepMap)
python run_all.py

# 5. Regenerate the figures from the shipped result files, no recomputation
python run_all.py --figs-only

# 6. Check that every result file the figures depend on is present and complete
python run_all.py --verify

# 7. A quick end-to-end smoke test (d=30 only, a few minutes)
python run_all.py --quick
```

## Headline results

Skeletons are scored by STRING v11 precision (combined score >= 700), under a
strictly fair protocol: identical pre-filter threshold, gene set and
preprocessing for every method.

PBMC 3K (n = 2,700 cells, top-*d* variable genes):

| *d* | NB-LR, method of moments | Fisher's *z* | Edges (NB / Fz) |
|----:|-------------------------:|-------------:|----------------:|
| 30  | **14.11 %**              | 13.82 %      | 241 / 246 |
| 50  | **10.07 %**              | 9.40 %       | 596 / 670 |
| 100 | **7.47 %**               | 5.95 %       | 1,379 / 2,084 |
| 200 | **6.50 %**               | 4.43 %       | 2,293 / 5,306 |

Paul15 (mouse haematopoiesis, validated against mouse STRING), *d* = 30:
**10.38 %** vs 9.64 %.

Cell-type-resolved PBMC skeletons at *d* = 30: B cells **29.89 %** vs 24.59 %,
CD4+ T cells **27.43 %** vs 24.41 %, CD8+ T cells **21.92 %** vs 20.49 %,
CD14+ monocytes **17.52 %** vs 16.67 %. NB-LR exceeds Fisher's *z* in all
15 PBMC cell-type x dimension settings, and in 13 of 14 Paul15
cluster x dimension settings.

Independent validation:

- **Reactome** pathway co-membership — curated from experimental literature and
  therefore independent of STRING's co-expression channel — is enriched over a
  matched background at odds ratio 4.82 (*p* = 9.0e-4, *d* = 30) and 2.42
  (*p* = 1.6e-3, *d* = 50).
- **DepMap** CRISPR co-essentiality supports individual predictions: 8 of the 49
  testable edges survive a Bonferroni correction, headed by RPL13–RPL8
  (*r* = 0.183, *p* = 1.6e-10). Co-essentiality is pervasive genome-wide
  (27.1 % of random gene pairs reach *p* < 0.05), so the edge set as a whole is
  **not** enriched (OR 1.19, *p* = 0.34). This negative is reported explicitly
  and is not relied upon as evidence.
- A **permuted negative control** returns zero edges.

## External datasets

`python run_all.py --download` prints the exact file names and URLs. Summary:

| Key | File | Needed for |
|:--|:--|:--|
| `paul15` | `paul15.h5ad` | cross-tissue validation (GEO GSE72857 / scanpy) |
| `string_aliases` | `string_aliases.txt.gz` | human STRING v11 aliases (9606) |
| `string_ppi` | `string_ppi_full.txt.gz` | human STRING v11 links (9606) |
| `string_aliases_mouse` | `10090.string_aliases.txt.gz` | mouse STRING v11 aliases (10090) |
| `string_ppi_mouse` | `10090.string_ppi_full.txt.gz` | mouse STRING v11 links (10090) |
| `trrust` | `trrust_human.tsv` | transcription-factor validation |
| `depmap_crispr` | `CRISPR_gene_effect.csv` | CRISPR co-essentiality (DepMap 23Q4) |

PBMC 3K (`data/pbmc3k_filtered.h5ad`) and the Reactome GMT
(`data/ReactomePathways.gmt`) are included in this package.

STRING links files have **16 columns**; the combined confidence score is the
**last** column. The mouse files are required because Paul15 is a mouse dataset
and scoring it against human STRING yields zero hits.

## Package contents

```
.
├── run_all.py                        # one-command reproduction runner
├── requirements.txt
├── README.md
├── data/
│   ├── pbmc3k_filtered.h5ad          # PBMC 3K counts (5.1 MB)
│   ├── pbmc3k_preprocessed.npz       # cached top-500 gene matrix
│   └── ReactomePathways.gmt          # Reactome pathways (1.0 MB)
├── figures/                          # pre-generated, vector PDF + PNG preview
│   ├── fig1_pipeline.pdf/png         #   pipeline overview (TikZ)
│   ├── fig2_benchmark.pdf/png        #   benchmark, 9 panels
│   ├── fig3_network.pdf/png          #   causal skeleton network (TikZ)
│   ├── fig4_validation.pdf/png       #   validation and robustness, 4 panels
│   ├── fig5_type1_zinb.pdf/png       #   type-I error under zero-inflation
│   └── graphical_abstract.png
├── results/                          # every number in the manuscript
│   ├── fair_pbmc.json                #   PBMC benchmark, d = 30/50/100/200
│   ├── fair_paul15.json              #   Paul15 cross-tissue
│   ├── fair_celltype.json            #   PBMC cell-type skeletons
│   ├── fair_celltype_paul15.json     #   Paul15 cluster skeletons
│   ├── fair_benchmark.json           #   combined benchmark record
│   ├── fair_supplementary.json       #   edge tables, alpha_sig / tau sweeps, offset
│   ├── fair_downstream.json          #   STRING thresholds, Reactome, DepMap
│   ├── table1_fair_all.json          #   Table 1 assembly
│   ├── zinb_type1_v2_results.json    #   type-I error grid (Fig 5a)
│   ├── zinb_calib_v2_results.json    #   Monte-Carlo calibration (Fig 5b)
│   └── ...                           #   supporting checkpoints
└── scripts/
    ├── config.py                     # dataset path resolution
    ├── sccausal_ci.py                # NB-LR conditional independence test
    ├── sccausal_engine.py            # differentiable skeleton engine
    ├── sccausal_ci_fast.py           # fast CI variant
    ├── sccausal_twostage.py          # two-stage variant
    ├── sccausal_v2.py, sccausal_v3.py
    ├── reproduce_benchmark.py        # fair PBMC + Paul15 benchmark  -> fair_*.json
    ├── reproduce_table1.py           # Table 1 assembly               -> table1_fair_all.json
    ├── fair_supplementary.py         # alpha_sig / tau / offset sweeps
    ├── fair_downstream.py            # STRING thresholds, Reactome, DepMap
    ├── run_pbmc.py, run_paul.py, run_synthetic.py, run_baselines.py
    ├── run_everything.py, run_fix30.py
    ├── sweep_all.py, sweep_ci.py, sweep_full.py
    ├── download_data.py
    ├── _depmap_coessential.py        # CRISPR co-essentiality validation
    ├── _reactome_validate.py         # Reactome pathway validation
    ├── _download_reactome.py
    └── gen_figures/
        ├── fig1_pipeline.tex         # hand-written TikZ source, Figure 1
        ├── gen_fig1.py               # compiles it into figures/
        ├── gen_fig2.py               # 9-panel benchmark composite
        ├── gen_fig3.py               # TikZ circular network layout
        ├── gen_fig4.py               # 4-panel validation composite
        ├── gen_fig5.py               # type-I error under zero-inflation
        └── gen_graphical_abstract.py
```

One figure, one script: `scripts/gen_figures/fig1_pipeline.tex` is
hand-written TikZ and sits next to the script that compiles it, while
`gen_fig3.py` builds its TikZ source programmatically at run time
(`fig3_network.tex` is therefore generated, not tracked).  Both write the
rendered PDF/PNG into `figures/`.

Every figure script is deterministic and reads its numbers from `results/`;
`gen_fig5.py` regenerates the published Figure 5 byte-for-byte.

## Requirements

- Python >= 3.9
- numpy, scipy, pandas, matplotlib, networkx, scanpy, statsmodels,
  scikit-learn, torch, tqdm, scienceplots
- LaTeX (with TikZ) only for Figures 1 and 3, which are TikZ sources

## Citation

```bibtex
@article{gao2026sccausal,
  title   = {scCausal: distribution-matched conditional independence testing
             for single-cell causal discovery},
  author  = {Gao, Shuaidong},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

MIT.
