"""equal_count_comparison.py

Equal-count (iso-budget) comparison of the scCausal skeleton against Fisher's z
on PBMC 3K under the fair protocol (shared tau = 0.10, identical top-d gene set,
identical log1p pre-filter).

Motivation
----------
scCausal emits fewer skeleton edges than Fisher's z at every dimension
(241 vs 246 at d=30, but 1379 vs 2084 at d=100 and 2293 vs 5306 at d=200).  A
reviewer is entitled to ask whether the reported STRING precision advantage is
an artefact of that smaller operating point -- a pure denominator effect --
rather than evidence that the recovered edges are of higher quality.

Both neutralisations below are applied to Fisher's z only, i.e. the baseline is
handed its *best* case:

  (A) marginal-rank truncation
      Every candidate pair with |marginal correlation| > tau -- the statistic
      Fisher's z thresholds in its first stage -- is ranked by |rho|, and the
      top k are kept, where k is the number of edges scCausal emits at that
      dimension.  This ignores the second-stage conditioning, so it can only
      flatter the baseline.

  (B) within-skeleton truncation
      The full Fisher's z skeleton is first computed, then reduced to its k
      strongest edges by |rho|.  This respects conditioning and is the more
      conservative of the two.

Reported for each dimension: edge count, STRING hits, precision, the overlap of
the validated-edge sets, and a two-sided Fisher exact test on the 2x2 table of
(hits, miss) x (scCausal, Fisher).

Writes results/equal_count.json.  ASCII output only.
"""
import sys, os, json, time
import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reproduce_benchmark as rb  # noqa: E402

ROOT = rb.ROOT
TAU = 0.10
# Dimensions are taken from argv so a single dimension can be smoke-tested
# before the full run:  python equal_count_comparison.py 30
DS = [int(a) for a in sys.argv[1:] if not a.startswith('-')] or [30, 50, 100, 200]


def scCausal_record(d):
    """Edge count and STRING hits of scCausal NB-LR (moment) at dimension d.

    Two result files carry the same numbers under different key names:
    table1_fair_all.json uses scCausal_NBLR_moment/string_hits, while
    fair_pbmc.json uses NB_moment/hits.  Both are read so either alone suffices.
    """
    for fname in ('table1_fair_all.json', 'fair_pbmc.json'):
        path = os.path.join(ROOT, 'results', fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as fh:
            blob = json.load(fh)
        if str(d) not in blob:
            continue
        rec = blob[str(d)]
        for key in ('scCausal_NBLR_moment', 'NB_moment'):
            if key in rec:
                r = rec[key]
                return {'edges': r['edges'],
                        'string_hits': r.get('string_hits', r.get('hits')),
                        'precision': r['precision']}
    raise KeyError('no scCausal record for d=%d' % d)


def validated_set(edges, genes, gene2s, ppi):
    """The subset of edges whose gene pair is a STRING high-confidence PPI."""
    keep = set()
    for (i, j) in edges:
        sid = gene2s.get(genes[i].upper())
        tid = gene2s.get(genes[j].upper())
        if sid and tid and ((sid, tid) in ppi):
            keep.add((genes[i].upper(), genes[j].upper()))
    return keep


def main():
    X, genes, ct, n_cells = rb.load_pbmc()
    gene2s, ppi = rb.load_string(rb.ALIAS, rb.PPI, genes)
    var = X.var(axis=0)
    order = np.argsort(var)[::-1]
    print('[data] PBMC %d cells x %d genes, STRING HC edges %d'
          % (X.shape[0], X.shape[1], len(ppi)))

    out = {'tau': TAU, 'note': 'baseline given best case; truncation applied to Fisher only'}
    for d in DS:
        idx = order[:d]
        Xd = X[:, idx]
        Gd = [genes[i] for i in idx]

        sc = scCausal_record(d)
        k = sc['edges']
        sc_hits = sc['string_hits']
        sc_prec = sc['precision']
        sc_valid = None
        print('\n=== d=%d ===' % d)
        print('  scCausal  k=%d  hits=%d  precision=%.2f%%' % (k, sc_hits, sc_prec))

        logX = np.log1p(Xd)
        corr = np.corrcoef(logX.T)
        corr = np.nan_to_num(corr, nan=0.0)
        cand = [(abs(corr[i, j]), i, j)
                for i in range(d) for j in range(i + 1, d)
                if abs(corr[i, j]) > TAU]
        cand.sort(key=lambda t: -t[0])
        print('  Fisher candidates above tau: %d' % len(cand))

        # ---- (A) marginal-rank truncation -------------------------------------
        setA = [(i, j) for _, i, j in cand[:k]]
        nA, hA, pA = rb.score(setA, Gd, gene2s, ppi)
        validA = validated_set(setA, Gd, gene2s, ppi)
        print('  (A) |rho| top-k : edges=%d hits=%d precision=%.2f%%' % (nA, hA, pA))

        # ---- (B) within-skeleton truncation -----------------------------------
        t0 = time.time()
        fz_full = rb.pc_fz(Xd, d, alpha=0.05, tau=TAU)
        nF, hF, pF = rb.score(fz_full, Gd, gene2s, ppi)
        rank = {(i, j): abs(corr[i, j]) for (i, j) in fz_full}
        fz_ranked = sorted(fz_full, key=lambda e: -rank[e])
        setB = fz_ranked[:k]
        nB, hB, pB = rb.score(setB, Gd, gene2s, ppi)
        validB = validated_set(setB, Gd, gene2s, ppi)
        print('  (B) full skeleton %d edges (hits=%d, %.2f%%) -> top-%d : hits=%d precision=%.2f%%  (%.0fs)'
              % (nF, hF, pF, k, hB, pB, time.time() - t0))

        # ---- scCausal validated set, for overlap ------------------------------
        supp = os.path.join(ROOT, 'results', 'fair_supplementary.json')
        if os.path.exists(supp) and d in (30, 50):
            with open(supp, encoding='utf-8') as fh:
                blob = json.load(fh)
            rec = blob.get('edges_d%d' % d)
            if rec:
                sc_valid = set(tuple(x) for x in rec['nb_validated'])

        row = {
            'd': d, 'n': int(Xd.shape[0]),
            'zero_pct': round(float(100 * (Xd == 0).mean()), 1),
            'scCausal': {'edges': k, 'hits': sc_hits, 'precision': sc_prec},
            'fisher_marginal_topk': {'edges': nA, 'hits': hA, 'precision': round(pA, 2)},
            'fisher_skeleton': {'edges': nF, 'hits': hF, 'precision': round(pF, 2)},
            'fisher_skeleton_topk': {'edges': nB, 'hits': hB, 'precision': round(pB, 2)},
        }
        # two-sided Fisher exact test on (hits, misses) x (scCausal, Fisher)
        for tag, hits in (('marginal', hA), ('skeleton_topk', hB)):
            tab = [[sc_hits, k - sc_hits], [hits, k - hits]]
            try:
                odds, pv = fisher_exact(tab)
            except Exception:
                odds, pv = float('nan'), float('nan')
            row['fisher_exact_%s' % tag] = {'odds_ratio': round(float(odds), 4),
                                            'p_value': float(pv)}
            print('  Fisher exact scCausal vs %-14s: OR=%.3f  p=%.4f' % (tag, odds, pv))

        if sc_valid is not None:
            row['validated_overlap_marginal'] = {
                'scCausal': len(sc_valid), 'fisher_topk': len(validA),
                'shared': len(sc_valid & validA),
                'scCausal_only': len(sc_valid - validA),
                'fisher_only': len(validA - sc_valid)}
            print('  validated overlap (A): shared=%d  scCausal-only=%d  fisher-only=%d'
                  % (len(sc_valid & validA), len(sc_valid - validA), len(validA - sc_valid)))

        out['d%d' % d] = row

    dst = os.path.join(ROOT, 'results', 'equal_count.json')
    with open(dst, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('\n-> %s' % dst)

    print('\n=== SUMMARY (equal edge budget k = scCausal edge count) ===')
    print('  %4s | %16s | %14s | %14s | %14s' %
          ('d', 'scCausal (k)', '(A) |rho| top-k', '(B) skel top-k', 'full skeleton'))
    for d in DS:
        r = out['d%d' % d]
        print('  %4d | %5d %4d %5.2f%% | %5d %4d %5.2f%% | %5d %4d %5.2f%% | %5d %4d %5.2f%%'
              % (d,
                 r['scCausal']['edges'], r['scCausal']['hits'], r['scCausal']['precision'],
                 r['fisher_marginal_topk']['edges'], r['fisher_marginal_topk']['hits'],
                 r['fisher_marginal_topk']['precision'],
                 r['fisher_skeleton_topk']['edges'], r['fisher_skeleton_topk']['hits'],
                 r['fisher_skeleton_topk']['precision'],
                 r['fisher_skeleton']['edges'], r['fisher_skeleton']['hits'],
                 r['fisher_skeleton']['precision']))


if __name__ == '__main__':
    main()
