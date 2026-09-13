#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real Perturb-seq interventional validation for scCausal (K562 essential, Replogle 2022).

Observation arm
    A PC skeleton is learned from non-targeting control (NTC) cells ONLY, using the
    same NB-LR and Fisher-z implementations as the manuscript (imported from
    reproduce_benchmark.py, never reimplemented). NTC cells carry no CRISPRi guide,
    so the input to the skeleton is purely observational.

Intervention arm
    CRISPRi knockdown effects are read off the same experiment: for every panel gene A,
    the expression of every other panel gene B is compared between A-knockdown cells and
    NTC cells. That contrast is the experimental do(A) operator, i.e. an interventional
    ground truth rather than a database co-occurrence.

Panel design
    The d panel genes are drawn from genes that are BOTH (i) actually perturbed in the
    screen with at least --min-cells cells and (ii) present in the count matrix, ranked
    by variance across NTC cells. Every panel gene therefore carries a measured
    interventional effect, so the gold standard spans the whole panel.

Repeats
    The NTC subsample is drawn under --n-seeds independent seeds; the skeleton is
    relearned each time and per-seed odds ratios are summarised with a paired Wilcoxon
    signed-rank test of NB-LR against Fisher's z.

Usage
    python perturbseq_validation.py --h5 data/replogle_2022_k562_essential.h5ad --d 50 --n-seeds 10
"""
import sys, os, json, time, argparse
import numpy as np
import h5py
from scipy.stats import mannwhitneyu, fisher_exact, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from reproduce_benchmark import estimate_alpha_moment, pc_nb, pc_fz  # noqa: E402

try:
    import config as _cfg
    DEFAULT_H5 = _cfg.get_path('replogle_h5ad')
except Exception:
    DEFAULT_H5 = os.path.join(ROOT, 'data', 'replogle_2022_k562_essential.h5ad')
DEFAULT_OUT = os.path.join(ROOT, 'results', 'checkpoints', 'perturbseq_validation.json')
NTC_LABELS = ('control', 'non-targeting', 'ntc', 'nontargeting', 'ctrl')
GOLD_TAGS = [(0.05, 0.25, 'fdr05_lfc025'), (0.05, 0.50, 'fdr05_lfc050'),
             (0.01, 0.50, 'fdr01_lfc050')]


def _dec(x):
    return x.decode('utf-8', 'replace') if isinstance(x, (bytes, np.bytes_)) else str(x)


def read_categorical(f, key):
    g = f[key]
    cats = [_dec(c) for c in np.asarray(g['categories'])]
    return cats, np.asarray(g['codes'], dtype=np.int64)


def load_perturbation(f):
    cats, codes = read_categorical(f, 'obs/perturbation')
    ntc = next((i for i, c in enumerate(cats) if c.lower() in NTC_LABELS), None)
    if ntc is None:
        raise RuntimeError('no non-targeting control label in obs/perturbation')
    return cats, codes, ntc


def usable_perturbed_genes(cats, codes, genes, ntc_idx, min_cells):
    col_of = {g: i for i, g in enumerate(genes)}
    counts = np.bincount(codes, minlength=len(cats))
    return [(g, int(counts[i]), col_of[g]) for i, g in enumerate(cats)
            if i != ntc_idx and g in col_of and counts[i] >= min_cells]


def read_submatrix(X, rows, cols):
    """X[rows, cols] with row fancy indexing and contiguous column runs."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    order = np.argsort(cols, kind='stable')
    cs = cols[order]
    out = np.empty((len(rows), len(cs)), dtype=np.float32)
    start = 0
    for k in range(1, len(cs) + 1):
        if k == len(cs) or cs[k] != cs[k - 1] + 1:
            out[:, start:k] = X[rows, int(cs[start]):int(cs[k - 1]) + 1]
            start = k
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    return out[:, inv]


def intervention_effects(X, panel, ntc_rows):
    """do(A) effects: knockdown of A vs NTC on every panel gene B."""
    d = len(panel)
    cols = np.array([p['col'] for p in panel], dtype=np.int64)
    lfc = np.full((d, d), np.nan)
    padj = np.full((d, d), np.nan)
    Xn = np.log1p(read_submatrix(X, ntc_rows, cols))
    for i, p in enumerate(panel):
        Xa = np.log1p(read_submatrix(X, p['rows'], cols))
        pv = np.ones(d)
        for j in range(d):
            if i == j:
                continue
            xa, xn = Xa[:, j], Xn[:, j]
            lfc[i, j] = float(xa.mean() - xn.mean())
            if xa.std() == 0 and xn.std() == 0:
                pv[j] = 1.0
                continue
            try:
                _, pv[j] = mannwhitneyu(xa, xn, alternative='two-sided')
            except ValueError:
                pv[j] = 1.0
        keep = [j for j in range(d) if j != i]
        if keep:
            padj[i, keep] = multipletests(pv[keep], method='fdr_bh')[1]
    return lfc, padj


def edge_table(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def build_gold(d, lfc, padj, fdr, lfc_thr):
    """Symmetric gold: {A,B} is gold if A's knockdown moves B or B's moves A."""
    gold = set()
    for (i, j) in edge_table(d):
        fwd = (not np.isnan(padj[i, j])) and padj[i, j] < fdr and abs(lfc[i, j]) > lfc_thr
        rev = (not np.isnan(padj[j, i])) and padj[j, i] < fdr and abs(lfc[j, i]) > lfc_thr
        if fwd or rev:
            gold.add((i, j))
    return gold


def score(sk_edges, gold, all_edges):
    sk = {tuple(sorted(e)) for e in sk_edges}
    a = len(sk & gold)
    b = len(sk - gold)
    c = len(gold - sk)
    dd = len(all_edges) - a - b - c
    if b > 0 and c > 0:
        orv = round(float(a * dd / (b * c)), 4)
        pv = float(fisher_exact([[a, b], [c, dd]], alternative='greater')[1])
    else:
        orv, pv = None, None
    return dict(edges=len(sk), hits=a, gold=len(gold), pairs=len(all_edges),
                odds_ratio=orv, fisher_p=pv,
                precision=round(100.0 * a / len(sk), 2) if sk else None,
                recall=round(100.0 * a / len(gold), 2) if gold else None)


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

    if not os.path.exists(args.h5):
        raise SystemExit('missing h5: %s' % args.h5)
    t_start = time.time()
    fh = h5py.File(args.h5, 'r')
    X = fh['X']
    genes = [_dec(g) for g in np.asarray(fh['var']['gene_name'][:])]
    cats, codes, ntc_idx = load_perturbation(fh)
    ntc_rows = np.sort(np.where(codes == ntc_idx)[0])
    n_cells, n_genes = X.shape
    print('[data] cells=%d genes=%d perturbations=%d NTC(%s)=%d'
          % (n_cells, n_genes, len(cats) - 1, cats[ntc_idx], len(ntc_rows)))

    usable = usable_perturbed_genes(cats, codes, genes, ntc_idx, args.min_cells)
    print('[pool] %d perturbed genes present with >=%d cells' % (len(usable), args.min_cells))

    t0 = time.time()
    Xntc_all = np.asarray(X[ntc_rows, :], dtype=np.float32)
    print('[read] NTC block %s in %.0fs' % (Xntc_all.shape, time.time() - t0))

    result = dict(meta=dict(
        source=os.path.basename(args.h5), cells=int(n_cells), genes=int(n_genes),
        perturbations=len(cats) - 1, ntc_label=cats[ntc_idx], ntc_cells=int(len(ntc_rows)),
        perturbation_pool=len(usable), min_cells=args.min_cells, tau=args.tau,
        alpha=args.alpha, base_seed=args.seed, n_seeds=args.n_seeds, fair=True))
    results = {}

    for d in args.d:
        print('\n=== d=%d ===' % d)
        ucols = np.array([u[2] for u in usable], dtype=np.int64)
        var = np.log1p(Xntc_all[:, ucols]).var(axis=0)
        rank = np.argsort(-var)[:d]
        panel = [dict(gene=usable[k][0], n_cells=usable[k][1], col=int(usable[k][2]))
                 for k in rank]
        cols = np.array([p['col'] for p in panel], dtype=np.int64)
        Xd = np.asarray(Xntc_all[:, cols], dtype=np.float32)
        alpha_hat = estimate_alpha_moment(Xd)
        print('[panel] d=%d cells/pert min=%d median=%d alpha_hat=%.4f zeros=%.1f%%'
              % (d, min(p['n_cells'] for p in panel),
                 int(np.median([p['n_cells'] for p in panel])), alpha_hat,
                 100 * (Xd == 0).mean()))

        for p in panel:
            p['rows'] = np.sort(np.where(codes == cats.index(p['gene']))[0])
        t0 = time.time()
        lfc, padj = intervention_effects(X, panel, ntc_rows)
        print('[do(A)] %d x %d contrasts (%.0fs)' % (d, d, time.time() - t0))

        all_edges = edge_table(d)
        gold_tags = {tag: build_gold(d, lfc, padj, fdr, thr) for fdr, thr, tag in GOLD_TAGS}
        rec = dict(alpha_hat=round(float(alpha_hat), 4),
                   zero_pct=round(float(100 * (Xd == 0).mean()), 2),
                   n_ntc=int(len(ntc_rows)),
                   cells_per_perturbation_median=int(np.median([p['n_cells'] for p in panel])),
                   panel_genes=[p['gene'] for p in panel],
                   gold={tag: len(g) for tag, g in gold_tags.items()},
                   sweep={}, null={})
        maxeff = np.array([np.nanmax([abs(lfc[i, j]), abs(lfc[j, i])]) for (i, j) in all_edges])

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
                if reps == 1:
                    rec['scoring_%s' % tag] = run[tag]
                    for name, e in (('nb', e_nb), ('fisher_z', e_fz)):
                        k = len(e)
                        if k == 0:
                            continue
                        rng0 = np.random.default_rng(args.seed)
                        draws = rng0.integers(0, len(all_edges), size=(args.n_null, k))
                        hits = np.array([len({all_edges[t] for t in rw} & gset)
                                         for rw in draws])
                        rec['null']['%s_%s' % (tag, name)] = dict(
                            mean=round(float(hits.mean()), 2),
                            sd=round(float(hits.std()), 2),
                            p_ge=float((hits >= run[tag][name]['hits']).mean()),
                            n_null=int(args.n_null))
                    skkey = {tuple(sorted(t)) for t in e}
                    skm = np.array([tuple(sorted(t)) in skkey for t in all_edges])
                    good = ~np.isnan(maxeff)
                    if 0 < skm.sum() < len(skm):
                        u, pv = mannwhitneyu(maxeff[skm & good], maxeff[(~skm) & good],
                                             alternative='greater')
                        rec['effect_%s' % name] = dict(
                            auc=round(float(u / ((skm & good).sum() * ((~skm) & good).sum())), 4),
                            mwu_p=float(pv))
                runs.append(run)
            agg = dict(n=int(n_use), reps=reps, runs=runs)
            for tag in gold_tags:
                for name in ('nb', 'fisher_z'):
                    ors = [r[tag][name]['odds_ratio'] for r in runs
                           if r[tag][name]['odds_ratio']]
                    hits = [r[tag][name]['hits'] for r in runs]
                    if ors:
                        agg['%s_%s' % (tag, name)] = dict(
                            or_mean=round(float(np.mean(ors)), 4),
                            or_sd=round(float(np.std(ors, ddof=1)), 4) if len(ors) > 1 else 0.0,
                            hit_mean=round(float(np.mean(hits)), 2), n_or=len(ors))
            if reps > 1:
                dif = [r['fdr05_lfc025']['nb']['odds_ratio'] -
                       r['fdr05_lfc025']['fisher_z']['odds_ratio'] for r in runs
                       if r['fdr05_lfc025']['nb']['odds_ratio'] and
                       r['fdr05_lfc025']['fisher_z']['odds_ratio']]
                if len(dif) >= 6:
                    agg['paired_wilcoxon_lfc025'] = dict(
                        median_diff=round(float(np.median(dif)), 4),
                        win_rate=round(float(np.mean([x > 0 for x in dif])), 3),
                        p=float(wilcoxon(dif, alternative='two-sided')[1]))
            rec['sweep']['%d' % n_use] = agg
            g, gz = agg.get('fdr05_lfc025_nb', {}), agg.get('fdr05_lfc025_fisher_z', {})
            print('  n=%5d reps=%2d | NB OR=%.2f+-%.2f hit=%.1f | Fz OR=%.2f+-%.2f hit=%.1f%s'
                  % (n_use, reps, g.get('or_mean', 0), g.get('or_sd', 0), g.get('hit_mean', 0),
                     gz.get('or_mean', 0), gz.get('or_sd', 0), gz.get('hit_mean', 0),
                     ('  | paired p=%.3g win=%.2f' % (agg['paired_wilcoxon_lfc025']['p'],
                                                      agg['paired_wilcoxon_lfc025']['win_rate']))
                     if 'paired_wilcoxon_lfc025' in agg else ''))
        results[str(d)] = rec
        result['results'] = results
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f_out:
            json.dump(result, f_out, indent=2)
        print('[save] %s' % args.out)

    fh.close()
    print('\n[done] %.0fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
