#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real Perturb-seq interventional validation on Norman 2019 (K562 CRISPRa).

Norman et al. (Science 2019) over-expressed transcription factors and TF pairs chosen
because their regulatory relationships were already known, so the screen carries far
stronger and more interpretable interventional signal than a genome-wide essential-gene
knockdown screen.

Observation arm
    The PC skeleton is learned from non-targeting cells only, i.e. cells whose two
    guides are both negative controls. These cells carry no CRISPRa perturbation and
    are therefore purely observational.

Intervention arm
    Single-gene perturbations (GENE + NegCtrl) provide the experimental do(A)
    operator: for every panel gene A, the expression of every other panel gene B is
    compared between A-overexpressing cells and non-targeting cells.

Panel
    The d panel genes are the perturbation targets themselves that are present in the
    count matrix, ranked by variance across non-targeting cells. Every panel gene
    therefore carries a measured interventional effect.

Repeats
    The NTC subsample is drawn under --n-seeds independent seeds; the skeleton is
    relearned each time and the per-seed odds ratios are summarised with a paired
    Wilcoxon signed-rank test of NB-LR against Fisher's z.

Usage
    python norman_validation.py --d 30 50 --n-seeds 10
    The screen is located through scripts/config.py (SC_CAUSAL_DATA or ~/scCausal_data/).
"""
import sys, os, json, time, argparse
import numpy as np
import h5py
from scipy.stats import mannwhitneyu, fisher_exact, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from reproduce_benchmark import estimate_alpha_moment, pc_nb, pc_fz      # noqa: E402
from perturbseq_validation import edge_table, build_gold, score           # noqa: E402

try:
    import config as _cfg
    DEFAULT_H5 = _cfg.get_path('norman_h5ad')
except Exception:
    DEFAULT_H5 = os.path.join(ROOT, 'data', 'norman_2019_raw.h5ad')
DEFAULT_OUT = os.path.join(ROOT, 'results', 'checkpoints', 'norman_validation.json')
NEG = 'NegCtrl'
GOLD_TAGS = [(0.05, 0.25, 'fdr05_lfc025'), (0.05, 0.10, 'fdr05_lfc010'),
             (0.05, 0.50, 'fdr05_lfc050')]


def _dec(x):
    return x.decode('utf-8', 'replace') if isinstance(x, (bytes, np.bytes_)) else str(x)


def read_uns_cat(f, name):
    o = f['uns'][name]
    if isinstance(o, h5py.Group):
        return [_dec(c) for c in np.asarray(o['categories'])]
    return [_dec(c) for c in np.asarray(o[:])]


def parse_labels(gi_cat):
    """'A_B__A_B' -> (A, B); aligned with category index."""
    out = []
    for c in gi_cat:
        try:
            a, b = c.split('__')[0].split('_', 1)
        except ValueError:
            a, b = c, c
        out.append((a, b))
    return out


def extract_csr(f, indptr, rows, cols_sorted):
    """X[rows, cols_sorted] for a CSR-based h5ad, as a dense float32 array."""
    data_ds, ind_ds = f['X/data'], f['X/indices']
    d = len(cols_sorted)
    out = np.zeros((len(rows), d), dtype=np.float32)
    for k, r in enumerate(rows):
        a, b = indptr[r], indptr[r + 1]
        if b <= a:
            continue
        cols = np.asarray(ind_ds[a:b], dtype=np.int64)
        vals = np.asarray(data_ds[a:b], dtype=np.float32)
        pos = np.clip(np.searchsorted(cols_sorted, cols), 0, d - 1)
        hit = cols_sorted[pos] == cols
        out[k, pos[hit]] = vals[hit]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default=DEFAULT_H5)
    ap.add_argument('--d', type=int, nargs='+', default=[30, 50])
    ap.add_argument('--n', type=int, nargs='+', default=[500, 1000, 3000, 0],
                    help='NTC subsample sizes; 0 = all')
    ap.add_argument('--n-seeds', type=int, default=10)
    ap.add_argument('--min-cells', type=int, default=50)
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--n-null', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    f = h5py.File(args.h5, 'r')
    gi_cat = read_uns_cat(f, 'guide_identity_categories')
    gnames = read_uns_cat(f, 'gene_symbols_categories')
    cat_parse = parse_labels(gi_cat)
    gid = np.asarray(f['obs']['guide_identity'], dtype=np.int64)
    vs = np.asarray(f['var']['gene_symbols'], dtype=np.int64)
    name2col = {}
    for c, n in enumerate(vs):
        name2col.setdefault(gnames[n], c)
    indptr = np.asarray(f['X/indptr'], dtype=np.int64)

    cell_a = cell_b = None
    ca = [cat_parse[i][0] for i in gid]
    cb = [cat_parse[i][1] for i in gid]
    cell_a, cell_b = np.array(ca), np.array(cb)
    neg_a = np.array([s.startswith(NEG) for s in ca])
    neg_b = np.array([s.startswith(NEG) for s in cb])
    ntc_rows = np.sort(np.where(neg_a & neg_b)[0])

    single = {}
    for i, (a, b) in enumerate(cat_parse):
        na, nb = a.startswith(NEG), b.startswith(NEG)
        if na and nb:
            continue
        if na or nb:
            single.setdefault(b if na else a, []).append(i)
    single_cells = {}
    for g, cat_ids in single.items():
        rows = np.where(np.isin(gid, cat_ids))[0]
        if len(rows) >= args.min_cells:
            single_cells[g] = np.sort(rows)

    print('[data] cells=%d genes=%d | NTC=%d | single-gene perturbations=%d'
          % (len(gid), len(gnames), len(ntc_rows), len(single_cells)))
    pool = [(g, r) for g, r in single_cells.items() if g in name2col]
    print('[pool] perturbation targets present in the matrix: %d' % len(pool))

    pool_genes = [g for g, _ in pool]
    pool_cols = np.array(sorted(name2col[g] for g in pool_genes), dtype=np.int64)
    t1 = time.time()
    Xntc = extract_csr(f, indptr, ntc_rows, pool_cols)
    pert_mats = {g: extract_csr(f, indptr, rows, pool_cols) for g, rows in pool}
    print('[extract] NTC %s + %d perturbation blocks in %.0fs'
          % (Xntc.shape, len(pert_mats), time.time() - t1))
    f.close()

    D_MAX = max(args.d)
    panel_idx = list(np.argsort(-np.log1p(Xntc).var(axis=0))[:D_MAX])
    Xp = Xntc[:, panel_idx]
    Xn_log = np.log1p(Xp)
    lfc = np.full((D_MAX, D_MAX), np.nan)
    padj = np.full((D_MAX, D_MAX), np.nan)
    for i, gi in enumerate(panel_idx):
        A = np.log1p(pert_mats[pool_genes[gi]][:, panel_idx])
        pv = np.ones(D_MAX)
        for j in range(D_MAX):
            if i == j:
                continue
            xa, xn = A[:, j], Xn_log[:, j]
            lfc[i, j] = float(xa.mean() - xn.mean())
            if xa.std() == 0 and xn.std() == 0:
                continue
            try:
                _, pv[j] = mannwhitneyu(xa, xn, alternative='two-sided')
            except ValueError:
                pv[j] = 1.0
        keep = [j for j in range(D_MAX) if j != i]
        padj[i, keep] = multipletests(pv[keep], method='fdr_bh')[1]
    ncell_panel = [len(single_cells[pool_genes[i]]) for i in panel_idx]
    print('[gold] median |lfc|=%.3f | cells/perturbation min=%d median=%d max=%d'
          % (np.nanmedian(np.abs(lfc[~np.eye(D_MAX, dtype=bool)])),
             min(ncell_panel), int(np.median(ncell_panel)), max(ncell_panel)))

    result = dict(meta=dict(
        source=os.path.basename(args.h5), cells=int(len(gid)), genes=int(len(gnames)),
        ntc_cells=int(len(ntc_rows)), single_gene_perturbations=len(single_cells),
        pool_in_matrix=len(pool), min_cells=args.min_cells, tau=args.tau,
        alpha=args.alpha, base_seed=args.seed, n_seeds=args.n_seeds, fair=True))
    results = {}

    for d in args.d:
        print('\n=== d=%d ===' % d)
        Xd = Xp[:, :d]
        all_edges = edge_table(d)
        gold_tags = {tag: build_gold(d, lfc, padj, fdr, thr) for fdr, thr, tag in GOLD_TAGS}
        rec = dict(panel_genes=[pool_genes[i] for i in panel_idx[:d]],
                   cells_per_perturbation_median=int(np.median(ncell_panel[:d])),
                   gold={tag: len(g) for tag, g in gold_tags.items()},
                   sweep={}, null={})
        for n in args.n:
            n_use = len(ntc_rows) if n == 0 else min(n, len(ntc_rows))
            reps = 1 if n_use >= len(ntc_rows) else args.n_seeds
            runs = []
            for si in range(reps):
                rng = np.random.default_rng(args.seed + 1000 * si + n_use)
                idx = np.sort(rng.choice(len(ntc_rows), n_use, replace=False))
                Xn = Xd[idx]
                e_nb = pc_nb(Xn, d, alpha=args.alpha, tau=args.tau, use_moment=True)
                e_fz = pc_fz(Xn, d, alpha=args.alpha, tau=args.tau)
                run = dict(seed=int(args.seed + 1000 * si), edges_nb=len(e_nb),
                           edges_fz=len(e_fz))
                for tag, gset in gold_tags.items():
                    run[tag] = dict(nb=score(e_nb, gset, all_edges),
                                    fisher_z=score(e_fz, gset, all_edges))
                runs.append(run)
                if reps == 1:
                    for tag, gset in gold_tags.items():
                        rec['scoring_%s' % tag] = run[tag]
                        for name, e in (('nb', e_nb), ('fisher_z', e_fz)):
                            k = len(e)
                            if k == 0:
                                continue
                            rngn = np.random.default_rng(args.seed)
                            draws = rngn.integers(0, len(all_edges), size=(args.n_null, k))
                            hits = np.array([len({all_edges[t] for t in rw} & gset)
                                             for rw in draws])
                            rec['null']['%s_%s' % (tag, name)] = dict(
                                mean=round(float(hits.mean()), 2),
                                sd=round(float(hits.std()), 2),
                                p_ge=float((hits >= run[tag][name]['hits']).mean()),
                                n_null=int(args.n_null))
            agg = dict(n=int(n_use), reps=reps, runs=runs)
            for tag in gold_tags:
                for name in ('nb', 'fisher_z'):
                    ors = [r[tag][name]['odds_ratio'] for r in runs if r[tag][name]['odds_ratio']]
                    hits = [r[tag][name]['hits'] for r in runs]
                    if ors:
                        agg['%s_%s' % (tag, name)] = dict(
                            or_mean=round(float(np.mean(ors)), 4),
                            or_sd=round(float(np.std(ors, ddof=1)), 4) if len(ors) > 1 else 0.0,
                            hit_mean=round(float(np.mean(hits)), 2),
                            n_or=len(ors))
            if reps > 1:
                d_ = [r['fdr05_lfc025']['nb']['odds_ratio'] - r['fdr05_lfc025']['fisher_z']['odds_ratio']
                      for r in runs
                      if r['fdr05_lfc025']['nb']['odds_ratio'] and r['fdr05_lfc025']['fisher_z']['odds_ratio']]
                if len(d_) >= 6:
                    agg['paired_wilcoxon_lfc025'] = dict(
                        median_diff=round(float(np.median(d_)), 4),
                        win_rate=round(float(np.mean([x > 0 for x in d_])), 3),
                        p=float(wilcoxon(d_, alternative='two-sided')[1]))
            rec['sweep']['%d' % n_use] = agg
            g = agg.get('fdr05_lfc025_nb', {})
            gz = agg.get('fdr05_lfc025_fisher_z', {})
            print('  n=%5d reps=%2d | NB OR=%.2f+-%.2f hit=%.1f | Fz OR=%.2f+-%.2f hit=%.1f%s'
                  % (n_use, reps, g.get('or_mean', 0), g.get('or_sd', 0), g.get('hit_mean', 0),
                     gz.get('or_mean', 0), gz.get('or_sd', 0), gz.get('hit_mean', 0),
                     ('  | paired p=%.3g win=%.2f' % (
                         agg['paired_wilcoxon_lfc025']['p'],
                         agg['paired_wilcoxon_lfc025']['win_rate']))
                     if 'paired_wilcoxon_lfc025' in agg else ''))
        results[str(d)] = rec
        result['results'] = results
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh_out:
            json.dump(result, fh_out, indent=2)
        print('[save] %s' % args.out)

    print('\n[done] %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
