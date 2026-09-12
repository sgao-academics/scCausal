# Result files

Every number in the manuscript is traceable to a file below and to the script
that produced it. Files are grouped by whether the manuscript cites them.

## 1. Files the manuscript cites

| File | Manuscript artefact | Produced by |
|:--|:--|:--|
| `fair_pbmc.json`, `fair_paul15.json` | Table 1 | `scripts/reproduce_table1.py` |
| `checkpoints/fair_pbmc_d{30,50,100,200}.json` | Table 1, Fig. 2(a-c,h) | `scripts/fair_supplementary.py` |
| `equal_count.json` | Table 2 (pooled rows) | `scripts/equal_count_comparison.py` |
| `equal_count_celltype.json` | Table 2 (cell-type rows) | `scripts/equal_count_celltype.py` |
| `sim_multiseed_d{30,50,100}_n{300,500,700,800,1500}.json` | Table 3 | `scripts/sim_multiseed.py` |
| `sim_disp_sweep.json` | Table 4 | `scripts/sim_disp_sweep.py` |
| `zinb_type1_v2_results.json` | Table 5(a) | `scripts/zinb_type1_v2.py` |
| `zinb_calib_v2_results.json` | Table 5(b) | `scripts/zinb_calib_v2.py` |
| `zinb_type1_moment.json` | Table 5(c) | `scripts/zinb_type1_moment.py` |
| `fair_celltype.json`, `fair_celltype_paul15.json` | Table 6, Table 7 | `scripts/fair_supplementary.py` |
| `checkpoints/ct*_d*.json` | Table 6, Table 7 | `scripts/fair_supplementary.py` |
| `fair_downstream.json` | Fig. 5(a), Reactome and DepMap blocks | `scripts/fair_downstream.py` |
| `checkpoints/ds_depmap.json` | CRISPR co-essentiality section (49 testable edges) | `scripts/fair_downstream.py` |
| `checkpoints/fair_paul15_d{50,100,200}.json` | Table 1 (Paul15 rows at `d=50/100/200`) | `scripts/fair_supplementary.py` |
| `string_background.json` | STRING random-pair background | `scripts/string_background_rate.py` |
| `checkpoints/supp_alpha_*.json`, `supp_tau_*.json`, `supp_offset_*.json` | parameter-sensitivity section | `scripts/fair_supplementary.py` |
| `checkpoints/supp_edges_d{30,50}.json` | edge lists for the network figure and the STRING background | `scripts/fair_supplementary.py` |

## 2. Exploratory runs, superseded

These files are kept for provenance only. They were produced by earlier
versions of the pipeline that used a different gene set, preprocessing or
dispersion estimator; **their numbers do not correspond to any table in the
manuscript and they are not inputs to any reported result.** The fair-protocol
files in section 1 supersede them.

| File | Why it differs |
|:--|:--|
| `realprec.json` | pre-fair-protocol run at `d=30` (215 edges, 14.88%); Table 1 reports 241 edges, 14.11% |
| `baseline_results.json` | early baseline run (`fisherz` 315 edges at `d=30`); Table 1 reports 246 |
| `fix_dropout.json` | dropout sweep used as a diagnostic while building the fair protocol |
| `fix_sim_mle.json`, `fix_sim_nsweep.json` | MLE-dispersion variants explored before settling on the moment estimator |
| `mech_disp_sweep.json`, `mech_theta_sweep.json` | 30-seed and 5-seed pilots; the reported sweep is the 100-seed `sim_disp_sweep.json` |
| `depmap_coessentiality.json`, `_depmap_log.txt` | early DepMap run over a 14-edge subset (11 testable; odds ratio undefined). The manuscript reports the 49-edge run in `checkpoints/ds_depmap.json` (OR 1.19, p 0.34) |
| `ci_full_sweep.json` | early dimension sweep of the pre-fair pipeline (`d=30`: 216 edges, 14.81%); Table 1 reports 241 edges, 14.11% |
| `multi_d_sweep.json` | pre-fair score-based sweep; its "scCausal" column is an earlier low-rank model, not the NB-LR PC test |
| `_libsize_main_ckpt.json` | intermediate offset checkpoint (11.90/8.80/6.40/5.90%); the manuscript values come from `fair_supplementary.json` `offset_d*` (12.25/10.08/7.81/7.48%) |
| `synthetic_benchmark.json`, `validation_results.json` | pre-fair-protocol runs kept for provenance |

## 3. Run checkpoints

`_*_ckpt.json` and `_*_checkpoint.json` are resumable-write checkpoints used to
make the long runs restartable. They are bookkeeping, not results.

## Reproducing

```bash
python scripts/reproduce_table1.py      # Table 1
python scripts/equal_count_comparison.py # Table 2 (pooled)
python scripts/sim_multiseed.py --d 30 --n 300 --ne 20 --seeds 100   # Table 3 (one row)
python scripts/sim_disp_sweep.py        # Table 4      (~3 min)
python scripts/zinb_type1_moment.py     # Table 5(c)   (~20 s)
python scripts/string_background_rate.py # STRING background
```

Dataset paths are resolved by `scripts/config.py` through the environment
variables documented in the top-level `README.md`; no absolute path appears in
the source.
