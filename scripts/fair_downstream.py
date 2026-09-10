"""fair_downstream.py
Recompute the edge-set-dependent downstream analyses on the FAIR scCausal edge
sets, so that every number in the manuscript traces to the same protocol as
Table 1.

Produces results/fair_downstream.json with:

  edges_d100    : fair d=100 NB-moment and Fisher edge lists (needed by the
                  STRING multi-threshold panel; d=30/50 come from
                  fair_supplementary.json).
  threshold     : STRING precision vs combined-score threshold (400-900) for
                  NB-moment and Fisher at d = 30, 50, 100.
  go_d30        : functional composition of the STRING-validated d=30 gene set
                  over five curated categories, counted inside the top-30
                  variance-selected gene universe.
  reactome      : pathway co-membership enrichment of the STRING-validated
                  NB-moment edge set at d = 30 and d = 50 (Reactome GMT).
  depmap        : CRISPR co-essentiality (DepMap 23Q4) of the STRING-validated
                  NB-moment edges at d = 50, with an empirically measured
                  background rate.

All steps are checkpointed under results/checkpoints/ds_*.json.
"""
import sys, os, gzip, json, time, random
import numpy as np
from scipy.stats import pearsonr, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config                       # noqa: E402
import reproduce_benchmark as rb    # noqa: E402

ROOT = rb.ROOT
RESULT_DIR = rb.RESULT_DIR
DATA_DIR = rb.DATA_DIR
CKPT_DIR = rb.CKPT_DIR

REACTOME_GMT = os.path.join(DATA_DIR, 'ReactomePathways.gmt')
DEPMAP_CSV = config.resolve('depmap_crispr', 'SC_CAUSAL_DEPMAP')
STRING_PPI = rb.PPI

CATEGORIES = [
    ('HLA / MHC-II', lambda g: g.startswith('HLA-') or g in ('CD74', 'B2M')),
    ('Ribosomal',    lambda g: g.startswith(('RPL', 'RPS'))),
    ('S100 / Calcium', lambda g: g.startswith('S100') or g in ('LGALS1',)),
    ('Mitochondrial', lambda g: g.startswith('MT-')),
    ('Immune effectors', lambda g: g in ('GNLY', 'NKG7', 'GZMB', 'CCL5', 'SRGN',
                                         'IL32', 'PPBP', 'CTSS', 'LST1', 'TYROBP',
                                         'AIF1', 'FCER1G', 'CST3')),
]


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, 'a', encoding='utf-8', errors='ignore', buffering=1)

    def write(self, d):
        self.stream.write(d)
        self.stream.flush()
        self.fh.write(d)
        self.fh.flush()

    def flush(self):
        self.stream.flush()


def stream_ppi_scores(candidate_ids, path=STRING_PPI):
    """Single pass over the human STRING links file.

    candidate_ids : set of unordered STRING protein-ID tuples.
    Returns {(a, b): best_combined_score} for every candidate pair found.
    """
    best = {}
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as fh:
        fh.readline()
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            key = (p[0], p[1]) if (p[0], p[1]) <= (p[1], p[0]) else (p[1], p[0])
            if key in candidate_ids:
                # NOTE: the STRING links file has 16 columns; the *last* one is
                # the combined score (columns 2..14 are the individual channels).
                try:
                    s = int(p[-1])
                except ValueError:
                    continue
                if s > best.get(key, 0):
                    best[key] = s
    return best


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    _log = os.path.join(RESULT_DIR, 'run_log_downstream.txt')
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    t0 = time.time()
    print('\n' + '=' * 70)
    print('fair_downstream.py -- start')
    print('=' * 70)

    X, genes, ct, _ = rb.load_pbmc()
    gene2s, ppi700 = rb.load_string(rb.ALIAS, rb.PPI, genes)
    order = np.argsort(X.var(axis=0))[::-1]
    supp = json.load(open(os.path.join(RESULT_DIR, 'fair_supplementary.json'),
                          encoding='utf-8'))
    out = {}

    # ---------------- 1. fair edge lists ------------------------------------
    # d = 30, 50 come from fair_supplementary.json; that file stores the full
    # NB-moment edge list but only the Fisher COUNT, so the Fisher list is
    # regenerated here (pc_fz is deterministic and takes ~1 s per dimension).
    edges_by_d = {}
    for d in [30, 50]:
        rec = supp['edges_d%d' % d]
        idx = order[:d]
        Gd = [genes[i] for i in idx]
        fz = rb.pc_fz(X[:, idx], d, alpha=0.05, tau=0.10)
        rec['fz_edges'] = [[Gd[i], Gd[j]] for (i, j) in fz]
        edges_by_d[d] = rec
    for d in [100]:
        tag = 'ds_edges_d%d' % d
        cached = rb._ckpt_load_simple(tag)
        if cached is None:
            idx = order[:d]
            Xd = X[:, idx]
            Gd = [genes[i] for i in idx]
            t = time.time()
            a_est = rb.estimate_alpha_moment(Xd)
            nb_e = rb.pc_nb(Xd, d, alpha=0.05, tau=0.10, use_moment=True)
            fz_e = rb.pc_fz(Xd, d, alpha=0.05, tau=0.10)

            def _val(pairs):
                v = []
                for g1, g2 in pairs:
                    s1, s2 = gene2s.get(g1.upper()), gene2s.get(g2.upper())
                    if s1 and s2 and ((s1, s2) in ppi700):
                        v.append([g1, g2])
                return v

            nb_pairs = [[Gd[i], Gd[j]] for (i, j) in nb_e]
            fz_pairs = [[Gd[i], Gd[j]] for (i, j) in fz_e]
            cached = {'d': d, 'alpha_hat': round(a_est, 4), 'genes': Gd,
                      'nb_edges': nb_pairs, 'nb_validated': _val(nb_pairs),
                      'fz_edges': fz_pairs, 'fz_validated': _val(fz_pairs),
                      'time_s': round(time.time() - t, 1)}
            rb._ckpt_save_simple(tag, cached)
        else:
            print('[resume] edges d=%d' % d)
        edges_by_d[d] = cached
        print('[edges] d=%d NB %d (%d validated)  Fz %d (%d validated)'
              % (d, len(cached['nb_edges']), len(cached['nb_validated']),
                 len(cached['fz_edges']), len(cached['fz_validated'])))

    # ---------------- 2. STRING multi-threshold ----------------
    tag = 'ds_threshold'
    cached = rb._ckpt_load_simple(tag)
    if cached is None:
        cand = set()
        for d, rec in edges_by_d.items():
            for pair in rec['nb_edges'] + rec['fz_edges']:
                s1, s2 = gene2s.get(pair[0].upper()), gene2s.get(pair[1].upper())
                if s1 and s2:
                    cand.add((s1, s2) if (s1, s2) <= (s2, s1) else (s2, s1))
        print('[threshold] candidate STRING pairs: %d' % len(cand))
        t = time.time()
        scores = stream_ppi_scores(cand, STRING_PPI)
        print('[threshold] scored in %.0fs (%d pairs found)' % (time.time() - t, len(scores)))
        thresholds = [400, 500, 600, 700, 800, 900]

        def _prec(pairs, thr):
            hit = 0
            for g1, g2 in pairs:
                s1, s2 = gene2s.get(g1.upper()), gene2s.get(g2.upper())
                if not (s1 and s2):
                    continue
                key = (s1, s2) if (s1, s2) <= (s2, s1) else (s2, s1)
                if scores.get(key, 0) >= thr:
                    hit += 1
            n = len(pairs)
            return hit, n, round(100.0 * hit / n, 2) if n else 0.0

        cached = {}
        for d, rec in sorted(edges_by_d.items()):
            cached[str(d)] = {}
            for name, pairs in [('NB_moment', rec['nb_edges']),
                                ('Fisher_z', rec.get('fz_edges', []))]:
                if not pairs:
                    continue
                cached[str(d)][name] = {
                    str(thr): dict(zip(['hits', 'edges', 'precision'], _prec(pairs, thr)))
                    for thr in thresholds}
                print('[threshold] d=%3d %-10s %s' % (
                    d, name, ' '.join('%d:%.1f%%' % (thr, cached[str(d)][name][str(thr)]['precision'])
                                      for thr in thresholds)))
        rb._ckpt_save_simple(tag, cached)
    else:
        print('[resume] threshold')
    out['threshold'] = cached

    # ---------------- 3. GO-style composition at d = 30 ----------------
    tag = 'ds_go_d30'
    cached = rb._ckpt_load_simple(tag)
    if cached is None:
        rec = edges_by_d[30]
        universe = list(rec['genes'])
        validated_genes = sorted({g for p in rec['nb_validated'] for g in p})
        cached = {'universe_size': len(universe),
                  'validated_genes': validated_genes, 'categories': {}}
        for name, pred in CATEGORIES:
            in_cat = [g for g in universe if pred(g)]
            found = [g for g in in_cat if g in set(validated_genes)]
            cached['categories'][name] = {
                'total_in_cat': len(in_cat), 'found': len(found),
                'found_genes': found,
                'pct': round(100.0 * len(found) / len(in_cat), 1) if in_cat else 0.0}
        rb._ckpt_save_simple(tag, cached)
    else:
        print('[resume] go_d30')
    out['go_d30'] = cached
    for k, v in cached['categories'].items():
        print('[go] %-18s %d/%d = %.1f%%' % (k, v['found'], v['total_in_cat'], v['pct']))

    # ---------------- 4. Reactome pathway co-membership ----------------
    tag = 'ds_reactome'
    cached = rb._ckpt_load_simple(tag)
    if cached is None:
        pathway_genes = {}
        with open(REACTOME_GMT, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 3:
                    gs = set(g for g in p[2:] if g)
                    if len(gs) >= 2:
                        pathway_genes[p[0]] = gs
        print('[reactome] %d pathways' % len(pathway_genes))
        co = set()
        for gs in pathway_genes.values():
            gl = sorted(gs)
            for i in range(len(gl)):
                for j in range(i + 1, len(gl)):
                    co.add((gl[i], gl[j]))
        print('[reactome] %d co-member pairs' % len(co))

        cached = {'n_pathways': len(pathway_genes), 'n_co_member_pairs': len(co),
                  'by_d': {}}
        for d in [30, 50]:
            rec = edges_by_d[d]
            pairs = [tuple(sorted(p)) for p in rec['nb_validated']]
            universe = sorted({g for p in pairs for g in p})
            bg_pairs = [(universe[i], universe[j])
                        for i in range(len(universe))
                        for j in range(i + 1, len(universe))]
            e_co = sum(1 for p in pairs if p in co)
            b_co = sum(1 for p in bg_pairs if p in co)
            tab = [[e_co, len(pairs) - e_co], [b_co, len(bg_pairs) - b_co]]
            orr, pv = fisher_exact(tab, alternative='greater')
            detail = []
            for p in pairs:
                if p in co:
                    pws = [n for n, gs in pathway_genes.items()
                           if p[0] in gs and p[1] in gs]
                    detail.append({'pair': list(p), 'pathways': pws[:3],
                                   'n_pathways': len(pws)})
            cached['by_d'][str(d)] = {
                'n_edges': len(pairs), 'n_genes': len(universe),
                'co_member_edges': e_co,
                'edge_pct': round(100.0 * e_co / max(len(pairs), 1), 1),
                'bg_pct': round(100.0 * b_co / max(len(bg_pairs), 1), 1),
                'odds_ratio': round(orr, 3), 'fisher_p': float(pv),
                'n_bg_pairs': len(bg_pairs),
                'examples': sorted(detail, key=lambda x: -x['n_pathways'])[:8]}
            print('[reactome] d=%d %d/%d (%.1f%%) vs bg %.1f%% OR=%.2f p=%.4g'
                  % (d, e_co, len(pairs), cached['by_d'][str(d)]['edge_pct'],
                     cached['by_d'][str(d)]['bg_pct'], orr, pv))
        rb._ckpt_save_simple(tag, cached)
    else:
        print('[resume] reactome')
    out['reactome'] = cached

    # ---------------- 5. DepMap CRISPR co-essentiality ----------------
    tag = 'ds_depmap'
    cached = rb._ckpt_load_simple(tag)
    if cached is None:
        import pandas as pd
        print('[depmap] loading %s ...' % DEPMAP_CSV)
        t = time.time()
        df = pd.read_csv(DEPMAP_CSV, index_col=0)
        df.columns = [c.split(' (')[0] for c in df.columns]
        print('[depmap] %d cell lines x %d genes in %.0fs'
              % (df.shape[0], df.shape[1], time.time() - t))
        rec = edges_by_d[50]
        pairs = [tuple(p) for p in rec['nb_validated']]
        have = [p for p in pairs if p[0] in df.columns and p[1] in df.columns]
        missing = sorted({g for p in pairs for g in p if g not in df.columns})
        print('[depmap] testable edges %d/%d ; missing genes: %s'
              % (len(have), len(pairs), missing))
        res = []
        for g1, g2 in have:
            r, p = pearsonr(df[g1].values, df[g2].values)
            if not (np.isfinite(r) and np.isfinite(p)):
                # a constant CRISPR profile across cell lines -> correlation undefined
                print('[depmap] dropped %s-%s: non-finite correlation' % (g1, g2))
                continue
            res.append({'pair': [g1, g2], 'r': round(float(r), 4),
                        'p': float(p), 'significant': bool(p < 0.05)})
        n_sig = sum(1 for x in res if x['significant'])

        rng = random.Random(0)
        cols = list(df.columns)
        bg = []
        for _ in range(10000):
            a, b = rng.sample(cols, 2)
            try:
                _, p = pearsonr(df[a].values, df[b].values)
                bg.append(bool(p < 0.05))
            except Exception:
                bg.append(False)
        bg_rate = float(np.mean(bg))
        tab = [[n_sig, len(res) - n_sig],
               [int(round(bg_rate * 10000)), 10000 - int(round(bg_rate * 10000))]]
        orr, pv = fisher_exact(tab, alternative='greater')
        cached = {'n_cell_lines': int(df.shape[0]), 'n_genes_depmap': int(df.shape[1]),
                  'edges_tested': len(res), 'n_significant': n_sig,
                  'sig_pct': round(100.0 * n_sig / max(len(res), 1), 1),
                  'background_rate_pct': round(100.0 * bg_rate, 2),
                  'fisher_OR': float(orr) if np.isfinite(orr) else None,
                  'fisher_p': float(pv), 'missing_genes': missing,
                  'all_edges': sorted(res, key=lambda x: x['p'])}
        rb._ckpt_save_simple(tag, cached)
    else:
        print('[resume] depmap')
    out['depmap'] = cached
    print('[depmap] %d/%d significant (%.1f%%) vs background %.2f%%'
          % (cached['n_significant'], cached['edges_tested'],
             cached['sig_pct'], cached['background_rate_pct']))

    with open(os.path.join(RESULT_DIR, 'fair_downstream.json'), 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print('\n-> results/fair_downstream.json')
    print('TOTAL %.1f min' % ((time.time() - t0) / 60.0))
    print('DONE')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
