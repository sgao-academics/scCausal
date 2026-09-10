"""equal_count_celltype.py

Equal-count comparison of scCausal NB-LR (moment) against Fisher's z on the
per-cell-type subsets, where the raw precision gap in fair_celltype.json is
largest (e.g. +5.30pp for B cells at d=30).

The pooled PBMC analysis (equal_count_comparison.py) shows the gap there is a
denominator effect.  The cell-type subsets report a much larger delta, but the
edge counts still differ (87 vs 122 at B d=30; 892 vs 1484 at B d=200), so the
same question has to be asked again: is the gap real, or is it again a
consequence of scCausal emitting fewer edges?

Fisher's z is again handed its best case:
  (A) rank truncation  -- all candidate pairs with |marginal rho| > tau sorted by
      |rho|, top-k kept, k = scCausal's edge count;
  (B) skeleton truncation -- full Fisher's z skeleton reduced to its k strongest
      edges by |rho|.

Reports per (cell type, d): both precisions, the delta, and a two-sided Fisher
exact test on (hits, miss) x (scCausal, Fisher).

Writes results/equal_count_celltype.json.  ASCII output only.
"""
import sys, os, json, time
import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reproduce_benchmark as rb  # noqa: E402

ROOT = rb.ROOT
TAU = 0.10
MIN_CELLS = 200


def analyse(Xt, Gd, gene2s, ppi, d, k, sc_hits, tau=TAU):
    """Equal-budget Fisher's z at edge budget k; returns a result dict."""
    logX = np.log1p(Xt)
    corr = np.corrcoef(logX.T)
    corr = np.nan_to_num(corr, nan=0.0)

    cand = [(abs(corr[i, j]), i, j)
            for i in range(d) for j in range(i + 1, d)
            if abs(corr[i, j]) > tau]
    cand.sort(key=lambda t: -t[0])

    setA = [(i, j) for _, i, j in cand[:k]]
    nA, hA, pA = rb.score(setA, Gd, gene2s, ppi)

    fz_full = rb.pc_fz(Xt, d, alpha=0.05, tau=tau)
    nF, hF, pF = rb.score(fz_full, Gd, gene2s, ppi)
    rank = {(i, j): abs(corr[i, j]) for (i, j) in fz_full}
    setB = sorted(fz_full, key=lambda e: -rank[e])[:k]
    nB, hB, pB = rb.score(setB, Gd, gene2s, ppi)

    tab = [[sc_hits, k - sc_hits], [hB, k - hB]]
    try:
        odds, pv = fisher_exact(tab)
    except Exception:
        odds, pv = float('nan'), float('nan')

    return {
        'edges_scCausal': k,
        'hits_scCausal': sc_hits,
        'fisher_marginal_topk': {'edges': nA, 'hits': hA, 'precision': round(pA, 2)},
        'fisher_skeleton': {'edges': nF, 'hits': hF, 'precision': round(pF, 2)},
        'fisher_skeleton_topk': {'edges': nB, 'hits': hB, 'precision': round(pB, 2)},
        'fisher_exact_skeleton_topk': {'odds_ratio': round(float(odds), 4),
                                       'p_value': float(pv)},
    }


def run_block(label, X, genes, celltype, source, alias, ppi_path):
    gene2s, ppi = rb.load_string(alias, ppi_path, genes)
    order = np.argsort(X.var(axis=0))[::-1]
    types = [t for t in np.unique(celltype) if (celltype == t).sum() >= MIN_CELLS]
    print('\n' + '#' * 74)
    print('# %s : %d cell types with >= %d cells' % (label, len(types), MIN_CELLS))
    out = {}
    for t in types:
        mask = celltype == t
        n_cells = int(mask.sum())
        row = {}
        blk = source.get(str(t)) or source.get(t)
        if not blk:
            continue
        for d in sorted(int(k[1:]) for k in blk if k.startswith('d') and k[1:].isdigit()):
            if n_cells < 50 or n_cells < 2 * d:
                continue
            dr = blk['d%d' % d]
            if 'NB_moment' not in dr:
                continue
            idx = order[:d]
            Gd = [genes[i] for i in idx]
            Xt = X[mask][:, idx]
            k = dr['NB_moment']['edges']
            sc_hits = dr['NB_moment']['hits']
            sc_prec = dr['NB_moment']['precision']
            t0 = time.time()
            r = analyse(Xt, Gd, gene2s, ppi, d, k, sc_hits)
            r['n_cells'] = n_cells
            r['alpha_hat'] = dr.get('alpha_hat')
            r['scCausal_precision'] = sc_prec
            r['fisher_z_full_precision'] = dr['Fisher_z']['precision']
            r['delta_raw'] = dr['delta_moment_fisher']
            r['delta_equal_count'] = round(sc_prec - r['fisher_skeleton_topk']['precision'], 2)
            r['delta_equal_count_marginal'] = round(sc_prec - r['fisher_marginal_topk']['precision'], 2)
            row['d%d' % d] = r
            print('  %-20s d=%-4d n=%-5d k=%-5d | scCausal %5.2f%% (%d) | '
                  'Fz@k %5.2f%% (%d) | raw delta %+5.2f -> eq-count %+5.2f (p=%.3f) [%.0fs]'
                  % (str(t)[:20], d, n_cells, k, sc_prec, sc_hits,
                     r['fisher_skeleton_topk']['precision'], r['fisher_skeleton_topk']['hits'],
                     r['delta_raw'], r['delta_equal_count'],
                     r['fisher_exact_skeleton_topk']['p_value'], time.time() - t0))
        if row:
            out[str(t)] = row
    return out


def main():
    res = {}
    want = sys.argv[1:] or ['pbmc', 'paul']

    if 'pbmc' in want:
        X, genes, ct, _ = rb.load_pbmc()
        with open(os.path.join(ROOT, 'results', 'fair_celltype.json'), encoding='utf-8') as fh:
            src = json.load(fh)['types']
        res['PBMC_celltype'] = run_block('PBMC by cell type', X, genes, ct, src,
                                        rb.ALIAS, rb.PPI)

    if 'paul' in want:
        # The configured Paul15 path may not exist on this machine; fall back to
        # the package-relative copy without touching config.py (which ships in
        # the package). Set SC_CAUSAL_DATA or drop the file in external_data/ to
        # point at another location.
        cur = getattr(rb, 'PAUL_H5AD', None)
        if cur and not os.path.exists(cur):
            for cand in (os.environ.get('SC_CAUSAL_PAUL15', ''),
                         os.path.join(ROOT, 'data', 'paul15.h5ad'),
                         os.path.join(ROOT, 'external_data', 'paul15.h5ad')):
                if cand and os.path.exists(cand):
                    rb.PAUL_H5AD = cand
                    print('[data] Paul15 path overridden -> %s' % cand)
                    break
        X, genes, ct, _ = rb.load_paul()
        with open(os.path.join(ROOT, 'results', 'fair_celltype_paul15.json'), encoding='utf-8') as fh:
            src = json.load(fh)['types']
        res['Paul15_celltype'] = run_block('Paul15 (mouse) by cell type', X, genes, ct, src,
                                           rb.M_ALIAS, rb.M_PPI)

    dst = os.path.join(ROOT, 'results', 'equal_count_celltype.json')
    with open(dst, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print('\n-> %s' % dst)

    # summary
    for blk, rows in res.items():
        deltas = [r['delta_equal_count'] for t in rows for r in rows[t].values()]
        sig = [r for t in rows for r in rows[t].values()
               if r['fisher_exact_skeleton_topk']['p_value'] < 0.05]
        raw = [r['delta_raw'] for t in rows for r in rows[t].values()]
        if deltas:
            print('\n%s: %d configs | raw delta mean %+.2f | equal-count delta mean %+.2f '
                  '| positive %d/%d | p<0.05 %d'
                  % (blk, len(deltas), float(np.mean(raw)), float(np.mean(deltas)),
                     sum(1 for x in deltas if x > 0), len(deltas), len(sig)))


if __name__ == '__main__':
    main()
