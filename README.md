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

The full run also reproduces the matched-edge-budget comparison (Table 2 of
the manuscript) and the 100-seed sample-size sweep (Table 3), about five
minutes in total at `--jobs 4`. Add `--skip-sim` to omit the sweep, and
`--pbmc-only` to stay within the datasets that ship in this package.

## Headline results

Skeletons are scored by STRING v11 precision (combined score >= 700), under a
strictly fair protocol: identical pre-filter threshold, gene set and
preprocessing for every method.

PBMC 3K (n = 2,700 cells, top-*d* variable genes):

| *d* | NB-LR, method of moments | Fisher's *z* | Edges (NB / Fz) |
|----:|-------------------------:|-------------:|----------------:|
| 30  | 14.11 %                  | 13.82 %      | 241 / 246 |
| 50  | 10.07 %                  | 9.40 %       | 596 / 670 |
| 100 | 7.47 %                   | 5.95 %       | 1,379 / 2,084 |
| 200 | 6.50 %                   | 4.43 %       | 2,293 / 5,306 |

Precision is a ratio, and on these data the two tests do **not** run at the
same operating point, so the table above must not be read as an edge-quality
ranking. Truncating Fisher's *z* to NB-LR's edge count and comparing the two
edge sets with a Fisher exact test removes the difference: *p* = 1.000, 1.000,
0.886 and 1.000 at *d* = 30/50/100/200, and at *d* = 30 the two sets of
STRING-validated edges are identical (34 edges each, so the whole 0.29 pp gap
is a denominator effect, 241 vs 246 edges). Pooled over all 23 comparisons the
raw +1.92 pp difference becomes **-0.67 pp** at a matched budget, with 6 of 23
positive. Reproduce with `results/equal_count.json` and
`results/equal_count_celltype.json`.

Paul15 (mouse haematopoiesis, validated against mouse STRING), *d* = 30:
10.38 % vs 9.64 % raw; +0.18 pp across four clusters at a matched budget.

Cell-type-resolved PBMC skeletons at *d* = 30: B cells 29.89 % vs 24.59 %,
CD4+ T cells 27.43 % vs 24.41 %, CD8+ T cells 21.92 % vs 20.49 %, CD14+
monocytes 17.52 % vs 16.67 %. Cell-type resolution roughly doubles absolute
validation precision relative to the pooled network, but here too the raw gap
to Fisher's *z* does not survive a matched edge budget (-1.07 pp over 15
settings, 3 of 15 positive).

A 100-seed simulation benchmark with known ground truth locates where the two
tests actually differ. It sweeps the sample size at fixed dimension
(`results/sim_multiseed_*.json`):

| *d* | *n* | ΔF₁ (NB-LR − Fz) | Cohen's *d* | NB wins | *p* |
|----:|----:|-----------------:|------------:|--------:|----:|
| 30  | 300 | -0.030 | -0.42 | 26/100 | 5.6e-7 |
| 50  | 300 | -0.019 | -0.38 | 28/100 | 2.4e-8 |
| 30  | 500 | +0.012 | +0.19 | 61/100 | 1.7e-4 |
| 50  | 500 | +0.025 | +0.57 | 80/100 | 3.7e-13 |
| 100 | 700 | +0.006 | +0.14 | 67/100 | 2.7e-10 |
| 50  | 800 | +0.004 | +0.07 | 30/100 | 7.0e-4 |
| 50  | 1,500 | +0.003 | +0.07 | 9/50 | 7.4e-3 |

The sign follows *n*, not *d*: Fisher's *z* is significantly better below
*n* = 300, and NB-LR from *n* = 500 upward with an advantage that contracts as
the sample grows. At the PBMC sample size (*n* = 2,700) the two tests are
indistinguishable at a matched edge budget, which is the outcome the sweep
predicts.

What the method does deliver: *p*-values interpretable under the generative
model that produced the data, a data-driven dispersion that dominates the fixed
default (14.11 % vs 13.33 % at *d* = 30; 6.50 % vs 4.79 % at *d* = 200), and
Type-I error control characterised and repaired under heavy zero-inflation.

Independent validation:

- **Reactome** pathway co-membership — curated from experimental literature and
  therefore independent of STRING's co-expression channel — is enriched over a
  matched background at odds ratio 4.82 (*p* = 9.0e-4, *d* = 30) and 2.42
  (*p* = 1.6e-3, *d* = 50).
- **DepMap** CRISPR co-essentiality supports individual predictions: 9 of the 49
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
│   ├── fig1_pipeline.pdf/png         #   Figure 1: pipeline overview (TikZ)
│   ├── fig2_benchmark.pdf/png        #   Figure 2: benchmark, 9 panels
│   ├── fig3_type1_zinb.pdf/png       #   Figure 3: type-I error under zero-inflation
│   ├── fig4_network.pdf/png          #   Figure 4: causal skeleton network (TikZ)
│   ├── fig5_validation.pdf/png       #   Figure 5: validation and robustness, 4 panels
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
│   ├── zinb_type1_v2_results.json    #   type-I error grid (Figure 3a)
│   ├── zinb_calib_v2_results.json    #   Monte-Carlo calibration (Figure 3b)
│   ├── equal_count.json              #   matched-edge-budget comparison, pooled
│   ├── equal_count_celltype.json     #   matched-edge-budget comparison, cell type
│   ├── sim_multiseed_d30_n300.json   #   simulation sample-size sweep:
│   ├── sim_multiseed_d30_n500.json   #   one file per (d, n) configuration,
│   ├── sim_multiseed_d50_n300.json   #   100 paired seeds each (50 at
│   ├── sim_multiseed_d50_n500.json   #   n = 1,500)
│   ├── sim_multiseed_d50_n800.json
│   ├── sim_multiseed_d50_n1500.json
│   ├── sim_multiseed_d100_n700.json
│   └── ...                           #   supporting checkpoints
└── scripts/
    ├── config.py                     # dataset path resolution
    ├── sccausal_ci.py                # NB-LR conditional independence test (reference)
    ├── sccausal_ci_fast.py           # fast CI variant (statsmodels IRLS)
    ├── reproduce_benchmark.py        # fair PBMC + Paul15 benchmark  -> fair_*.json
    ├── reproduce_table1.py           # Table 1 assembly               -> table1_fair_all.json
    ├── fair_supplementary.py         # alpha_sig / tau / offset sweeps
    ├── fair_downstream.py            # STRING thresholds, Reactome, DepMap
    ├── run_paul.py, run_baselines.py
    ├── run_everything.py, run_fix30.py
    ├── sweep_ci.py, sweep_full.py
    ├── sim_multiseed.py              # simulation sweep, paired over seeds
    ├── equal_count_comparison.py     # matched-edge-budget, pooled PBMC
    ├── equal_count_celltype.py       # matched-edge-budget, cell-type networks
    ├── zinb_type1_v2.py              # type-I error grid (Fig. 3a)     -> results/zinb_type1_v2_results.json
    ├── zinb_calib_v2.py              # permutation-null calibration      -> results/zinb_calib_v2_results.json
    ├── download_data.py
    ├── _reactome_validate.py         # Reactome pathway validation
    ├── _download_reactome.py
    └── gen_figures/
        ├── fig1_pipeline.tex         # hand-written TikZ source, Figure 1
        ├── gen_fig1.py               # Figure 1: compiles it into figures/
        ├── gen_fig2.py               # Figure 2: 9-panel benchmark composite
        ├── gen_fig3.py               # Figure 3: type-I error under zero-inflation
        ├── gen_fig4.py               # Figure 4: TikZ circular network layout
        ├── gen_fig5.py               # Figure 5: 4-panel validation composite
        └── gen_graphical_abstract.py
```

`run_all.py` drives the reproduction pipeline: `reproduce_benchmark.py`,
`reproduce_table1.py`, `fair_supplementary.py`, `fair_downstream.py`,
`equal_count_comparison.py`, `equal_count_celltype.py`, `sim_multiseed.py`
and the `gen_figures/` scripts. The remaining scripts are exploratory runs
kept for provenance and are not required to reproduce any number reported in
the manuscript.

One figure, one script, and `gen_figN.py` always produces `figN_*`: the script
index is the figure number in the manuscript, so `gen_fig3.py` writes
`figures/fig3_type1_zinb.pdf/png`, `gen_fig4.py` writes
`figures/fig4_network.pdf/png`, and so on.

`scripts/gen_figures/fig1_pipeline.tex` is hand-written TikZ and sits next to
the script that compiles it, while `gen_fig4.py` builds its TikZ source
programmatically at run time (`fig4_network.tex` is therefore generated, not
tracked).  Both write the rendered PDF/PNG into `figures/`.

Every figure script is deterministic and reads its numbers from `results/`;
`gen_fig3.py` regenerates the published Figure 3 byte-for-byte.

## Requirements

- Python >= 3.9
- numpy, scipy, pandas, matplotlib, networkx, scanpy, statsmodels,
  scikit-learn, torch, tqdm, scienceplots
- LaTeX (with TikZ) only for Figures 1 and 4, which are TikZ sources

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
