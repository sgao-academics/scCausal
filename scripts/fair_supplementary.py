"""fair_supplementary.py
Supplementary computations for the FAIR scCausal protocol (PBMC 3K).

Everything here reuses the authoritative functions of reproduce_benchmark.py so
that the fair protocol (identical tau, identical top-d gene set, identical
log1p pre-filter for every method) is preserved exactly.

Parts
-----
A. Edge-list export  : d = 30, 50  -> skeleton edges + STRING-validated subset.
   Needed because the downstream analyses (network figure, Reactome
   co-membership, GO enrichment) previously ran on pre-fair edge sets.
B. Alpha sweep       : significance level in {0.01,0.03,0.05,0.07,0.10} at
   d = 30, NB-LR (moment) vs Fisher's z, identical tau = 0.10.
C. Tau sweep         : pre-filter tau in {0.05,0.10,0.15,0.20} at d = 30,
   both methods always evaluated at the SAME tau.
D. Library-size      : NB-LR (moment) with a log(total-UMI) GLM offset versus
   the no-offset baseline already stored in fair_pbmc.json.

Every item is checkpointed under results/checkpoints/supp_*.json and the run
resumes from the last completed item.
"""
import sys, os, json, time
import numpy as np
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reproduce_benchmark as rb   # noqa: E402

ROOT = rb.ROOT
RESULT_DIR = rb.RESULT_DIR
CKPT_DIR = rb.CKPT_DIR


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, 'a', encoding='utf-8', errors='ignore', buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass


# ---------------------------------------------------------------
# D. NB-LR with a GLM offset (library-size ablation)
# ---------------------------------------------------------------
def nb_lr_offset(X, i, j, cond, alpha, off):
    y = X[:, j]
    if len(cond) > 0:
        Xn = sm.add_constant(X[:, list(cond)])
    else:
        Xn = np.ones((len(y), 1))
    try:
        fam = NegativeBinomial(alpha=alpha)
        m0 = sm.GLM(y, Xn, family=fam, offset=off).fit(maxiter=50, disp=0)
        if m0.llf is None:
            return 1.0
        Xa = np.column_stack([Xn, X[:, i]])
        m1 = sm.GLM(y, Xa, family=fam, offset=off).fit(maxiter=50, disp=0)
        if m1.llf is None:
            return 1.0
        return 1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1)
    except Exception:
        return 1.0


def pc_nb_offset(X, d, off, alpha_sig=0.05, tau=0.10, max_k=1):
    """pc_nb() with a log-library-size GLM offset (otherwise identical)."""
    alpha_est = rb.estimate_alpha_moment(X[:, :d])
    corr = np.corrcoef(np.log1p(X[:, :d]).T)
    edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(corr[i, j]) > tau}
    edges = {(i, j) for (i, j) in edges
             if nb_lr_offset(X[:, :d], i, j, [], alpha_est, off) < alpha_sig}
    if max_k >= 1:
        removed = set()
        for (i, j) in list(edges):
            nb_i, nb_j = set(), set()
            for (a, b) in edges:
                if a == i and b != j:
                    nb_i.add(b)
                elif b == i and a != j:
                    nb_i.add(a)
                if a == j and b != i:
                    nb_j.add(b)
                elif b == j and a != i:
                    nb_j.add(a)
            for k in list(nb_i & nb_j)[:5]:
                if nb_lr_offset(X[:, :d], i, j, [k], alpha_est, off) >= alpha_sig:
                    removed.add((i, j))
                    break
        edges -= removed
    return alpha_est, sorted(edges)


# ---------------------------------------------------------------
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, os.path.join(RESULT_DIR, 'run_log_supp.txt'))
    t_start = time.time()
    print('\n' + '=' * 70)
    print('fair_supplementary.py -- start')
    print('=' * 70)

    X, genes, ct, n_cells = rb.load_pbmc()
    gene2s, ppi = rb.load_string(rb.ALIAS, rb.PPI, genes)
    var = X.var(axis=0)
    order = np.argsort(var)[::-1]
    libsize_full = X.sum(axis=1).astype(np.float64)
    log_lib = np.log(np.maximum(libsize_full, 1.0))
    print('[data] PBMC %s cells x %s genes, STRING symbols %d'
          % (X.shape[0], X.shape[1], sum(1 for g in genes if gene2s.get(g.upper()))))

    out = {}

    # ---------------- A. edge lists ----------------
    for d in [30, 50]:
        tag = 'supp_edges_d%d' % d
        cached = rb._ckpt_load_simple(tag)
        if cached is not None:
            out['edges_d%d' % d] = cached
            print('[resume] edges d=%d' % d)
            continue
        idx = order[:d]
        Xd = X[:, idx]
        Gd = [genes[i] for i in idx]
        t0 = time.time()
        a_est = rb.estimate_alpha_moment(Xd)
        nb_edges = rb.pc_nb(Xd, d, alpha=0.05, tau=0.10, use_moment=True)
        fz_edges = rb.pc_fz(Xd, d, alpha=0.05, tau=0.10)
        nb_pairs = [[Gd[i], Gd[j]] for (i, j) in nb_edges]
        fz_pairs = [[Gd[i], Gd[j]] for (i, j) in fz_edges]

        def _validated(pairs):
            v = []
            for g1, g2 in pairs:
                s1 = gene2s.get(g1.upper())
                s2 = gene2s.get(g2.upper())
                if s1 and s2 and ((s1, s2) in ppi):
                    v.append([g1, g2])
            return v

        rec = {'d': d, 'n': int(Xd.shape[0]), 'alpha_hat': round(a_est, 4),
               'tau': 0.10, 'genes': Gd,
               'nb_edges': nb_pairs, 'nb_validated': _validated(nb_pairs),
               'fz_edges_count': len(fz_pairs), 'fz_validated': _validated(fz_pairs),
               'time_s': round(time.time() - t0, 1)}
        rb._ckpt_save_simple(tag, rec)
        out['edges_d%d' % d] = rec
        print('[edges] d=%d NB %d edges (%d validated)  Fz %d edges (%d validated)  %.0fs'
              % (d, len(nb_pairs), len(rec['nb_validated']), len(fz_pairs),
                 len(rec['fz_validated']), rec['time_s']))

    # ---------------- B. alpha sweep ----------------
    d = 30
    idx = order[:d]
    Xd = X[:, idx]
    Gd = [genes[i] for i in idx]
    for a in [0.01, 0.03, 0.05, 0.07, 0.10]:
        tag = 'supp_alpha_%s_d%d' % (str(a).replace('.', 'p'), d)
        cached = rb._ckpt_load_simple(tag)
        if cached is not None:
            out['alpha_%s' % a] = cached
            print('[resume] alpha=%s' % a)
            continue
        t0 = time.time()
        nb_e = rb.pc_nb(Xd, d, alpha=a, tau=0.10, use_moment=True)
        ne, nh, npc = rb.score(nb_e, Gd, gene2s, ppi)
        fz_e = rb.pc_fz(Xd, d, alpha=a, tau=0.10)
        fe, fh, fpc = rb.score(fz_e, Gd, gene2s, ppi)
        rec = {'alpha_sig': a, 'tau': 0.10, 'd': d,
               'nb_edges': ne, 'nb_hits': nh, 'nb_precision': round(npc, 2),
               'fz_edges': fe, 'fz_hits': fh, 'fz_precision': round(fpc, 2),
               'time_s': round(time.time() - t0, 1)}
        rb._ckpt_save_simple(tag, rec)
        out['alpha_%s' % a] = rec
        print('[alpha] %.2f  NB %5d/%4d = %5.2f%%   Fz %5d/%4d = %5.2f%%  (%.0fs)'
              % (a, ne, nh, npc, fe, fh, fpc, rec['time_s']))

    # ---------------- C. tau sweep ----------------
    for t in [0.05, 0.10, 0.15, 0.20]:
        tag = 'supp_tau_%s_d%d' % (str(t).replace('.', 'p'), d)
        cached = rb._ckpt_load_simple(tag)
        if cached is not None:
            out['tau_%s' % t] = cached
            print('[resume] tau=%s' % t)
            continue
        t0 = time.time()
        nb_e = rb.pc_nb(Xd, d, alpha=0.05, tau=t, use_moment=True)
        ne, nh, npc = rb.score(nb_e, Gd, gene2s, ppi)
        fz_e = rb.pc_fz(Xd, d, alpha=0.05, tau=t)
        fe, fh, fpc = rb.score(fz_e, Gd, gene2s, ppi)
        rec = {'tau': t, 'alpha_sig': 0.05, 'd': d,
               'nb_edges': ne, 'nb_hits': nh, 'nb_precision': round(npc, 2),
               'fz_edges': fe, 'fz_hits': fh, 'fz_precision': round(fpc, 2),
               'time_s': round(time.time() - t0, 1)}
        rb._ckpt_save_simple(tag, rec)
        out['tau_%s' % t] = rec
        print('[tau] %.2f  NB %5d/%4d = %5.2f%%   Fz %5d/%4d = %5.2f%%  (%.0fs)'
              % (t, ne, nh, npc, fe, fh, fpc, rec['time_s']))

    # ---------------- D. library-size offset ablation ----------------
    for d in [30, 50, 100, 200]:
        tag = 'supp_offset_d%d' % d
        cached = rb._ckpt_load_simple(tag)
        if cached is not None:
            out['offset_d%d' % d] = cached
            print('[resume] offset d=%d' % d)
            continue
        idx = order[:d]
        Xd = X[:, idx]
        Gd = [genes[i] for i in idx]
        t0 = time.time()
        a_est, edges = pc_nb_offset(Xd, d, log_lib)
        ne, nh, npc = rb.score(edges, Gd, gene2s, ppi)
        rec = {'d': d, 'alpha_hat': round(a_est, 4), 'with_offset': True,
               'edges': ne, 'hits': nh, 'precision': round(npc, 2),
               'time_s': round(time.time() - t0, 1)}
        rb._ckpt_save_simple(tag, rec)
        out['offset_d%d' % d] = rec
        print('[offset] d=%3d  %5d edges  %4d hits  %5.2f%%  (%.0fs)'
              % (d, ne, nh, npc, rec['time_s']))

    with open(os.path.join(RESULT_DIR, 'fair_supplementary.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\n-> results/fair_supplementary.json')
    print('TOTAL %.1f min' % ((time.time() - t_start) / 60.0))
    print('DONE')


if __name__ == '__main__':
    main()
