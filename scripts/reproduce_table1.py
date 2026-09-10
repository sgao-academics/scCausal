"""reproduce_table1.py
Authoritative single-script reproduction of Table 1 (PBMC 3K, raw counts,
NB-LR with method-of-moments dispersion vs Fisher's z).

What it does
------------
1. Loads PBMC 3K (pbmc3k_filtered.h5ad) directly with h5py + scipy.sparse
   (no scanpy dependency; works in a minimal statsmodels env).
2. Selects the top-d highly variable genes by variance.
3. Runs two PC conditional-independence skeletons on identical data:
     - scCausal  : PC + NB-LR with a dispersion estimated from the data by
                   the method of moments (alpha_hat = median[(s^2 - m)/m^2]),
                   in closed form and O(d).
     - Fisher's z: the standard Fisher-z PC algorithm on raw counts.
4. Scores each recovered skeleton by STRING v11 precision (combined score
   >= 700, high confidence), exactly as in the paper.

The output JSON is under results/table1_<d>.json and is the direct evidence
behind Table 1 in the manuscript.

Usage
-----
    python reproduce_table1.py 30
    python reproduce_table1.py 30 50 100 200
"""
import sys, os, gzip, json, time, numpy as np
from scipy.stats import chi2, norm, pearsonr
from statsmodels.genmod.families import NegativeBinomial
import statsmodels.api as sm
import h5py
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402  (scCausal data path configuration)

ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data', 'pbmc3k_filtered.h5ad')
# STRING v11 resources; see config.py for how to point at your local copies.
ALIAS = config.resolve('string_aliases', 'SC_CAUSAL_STRING_ALIASES')
PPI = config.resolve('string_ppi', 'SC_CAUSAL_STRING_PPI')


def load_string(alias_path, ppi_path, gene_names):
    """Build STRING symbol->ENSP map and high-confidence (>=700) PPI edge set."""
    symbol2string = {}
    with gzip.open(alias_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.strip().split('\t')
            if len(p) >= 2 and not p[1].isdigit():
                # Map gene symbol (col 1) -> STRING ENSP (col 0)
                symbol2string[p[1].upper()] = p[0]
    gene2string = {g: symbol2string.get(g.upper(), '') for g in gene_names}
    ppi_set = set()
    with gzip.open(ppi_path, 'rt', encoding='utf-8', errors='ignore') as f:
        f.readline()  # header
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                try:
                    if int(p[-1]) >= 700:
                        ppi_set.add((p[0], p[1]))
                except ValueError:
                    continue
    return gene2string, ppi_set


def estimate_alpha_moment(X):
    """NB2 dispersion by the method of moments: Var = mu + alpha * mu^2.

    alpha_hat(g) = (s^2_g - xbar_g) / xbar_g^2  for each expressed gene g,
    and the global dispersion is the median over expressed genes. Closed-form,
    O(d), no iterative optimization. On PBMC 3K top-d genes this yields
    alpha_hat ~= 2.4, well above the statsmodels default alpha = 1.0.
    """
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    valid = mu > 0
    alphas = (var[valid] - mu[valid]) / np.maximum(mu[valid] ** 2, 1e-8)
    alphas = np.clip(alphas, 1e-4, None)
    return float(np.median(alphas))


def nb_lr(X, i, j, cond, alpha=None):
    """NB-LR CI test on raw counts. alpha=None keeps statsmodels default."""
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    try:
        fam = NegativeBinomial(alpha=alpha) if alpha is not None else NegativeBinomial()
        m0 = sm.GLM(y, Xn, family=fam).fit(maxiter=50, disp=0)
        if m0.llf is None:
            return 1.0
        Xa = np.column_stack([Xn, X[:, i]]) if Xn.shape[1] > 0 else sm.add_constant(X[:, i].reshape(-1, 1))
        m1 = sm.GLM(y, Xa, family=fam).fit(maxiter=50, disp=0)
        if m1.llf is None:
            return 1.0
        return 1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1)
    except Exception:
        return 1.0


def pc_nb(X, d, alpha=0.05, tau=0.10, max_k=1, use_moment=True):
    """PC skeleton using NB-LR CI tests. use_moment=True estimates dispersion."""
    alpha_est = estimate_alpha_moment(X[:, :d]) if use_moment else None
    corr = np.corrcoef(np.log1p(X).T)
    edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(corr[i, j]) > tau}
    new_e = set()
    for (i, j) in edges:
        if nb_lr(X, i, j, [], alpha=alpha_est) < alpha:
            new_e.add((i, j))
    edges = new_e
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
                if nb_lr(X, i, j, [k], alpha=alpha_est) >= alpha:
                    removed.add((i, j))
                    break
        edges -= removed
    return sorted(edges)


def pc_fz(X, d, alpha=0.05, tau=0.10, max_k=1):
    """Standard Fisher-z PC skeleton on raw counts (log1p inside)."""
    n = X.shape[0]
    logX = np.log1p(X)
    corr = np.corrcoef(logX.T)
    edges = set()
    for i in range(d):
        for j in range(i + 1, d):
            if abs(corr[i, j]) > tau:
                z = 0.5 * np.log((1 + corr[i, j]) / max(1 - corr[i, j], 1e-10))
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 3)))) < alpha:
                    edges.add((i, j))
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
                A = np.column_stack([np.ones(n), logX[:, k]])
                res_i = logX[:, i] - A @ np.linalg.lstsq(A, logX[:, i], rcond=None)[0]
                res_j = logX[:, j] - A @ np.linalg.lstsq(A, logX[:, j], rcond=None)[0]
                r, _ = pearsonr(res_i, res_j)
                z = 0.5 * np.log((1 + r) / max(1 - r, 1e-10))
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 4)))) >= alpha:
                    removed.add((i, j))
                    break
        edges -= removed
    return sorted(edges)


def score(edges, genes, gene2s, ppi):
    hits = 0
    for (i, j) in edges:
        sid = gene2s.get(genes[i].upper())
        tid = gene2s.get(genes[j].upper())
        if sid and tid and ((sid, tid) in ppi):
            hits += 1
    prec = 100.0 * hits / len(edges) if edges else 0.0
    return len(edges), hits, prec


def _xname(d):
    return d

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ds = [int(x) for x in args] or [30, 50, 100, 200]
    # Optional --tau VAL to override the fair shared pre-filter.
    tau_override = None
    for a in sys.argv[1:]:
        if a.startswith('--tau='):
            tau_override = float(a.split('=', 1)[1])
    f = h5py.File(DATA, 'r')
    Xg = f['X']
    nrows = int(Xg['indptr'].shape[0]) - 1
    ncols = int(len(f['var']['index']))
    data = np.asarray(Xg['data'], dtype=np.float32)
    indices = np.asarray(Xg['indices'], dtype=np.int32)
    indptr = np.asarray(Xg['indptr'], dtype=np.int32)
    mat = sp.csr_matrix((data, indices, indptr), shape=(nrows, ncols))
    X = np.asarray(mat.toarray(), dtype=np.float32)
    raw_idx = np.asarray(f['var']['index'])
    genes = [x.decode('utf-8', errors='ignore') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw_idx]
    f.close()

    var = X.var(axis=0)
    order = np.argsort(var)[::-1]  # descending variance, top-d selected below

    all_results = {}
    for d in ds:
        idx = order[:d]
        Xd = X[:, idx]
        Gd = [genes[i] for i in idx]
        print(f'\n=== PBMC 3K top-{d}: {Xd.shape[0]} cells, '
              f'{100 * (Xd == 0).mean():.1f}% zeros ===')

        gene2s, ppi = load_string(ALIAS, PPI, Gd)
        mapped = sum(1 for g in Gd if gene2s.get(g.upper()))
        print(f'STRING: mapped {mapped}/{d} genes, {len(ppi)} HC edges')

        alpha_est = estimate_alpha_moment(Xd)
        print(f'moment alpha_hat = {alpha_est:.4f}')

        # FAIR shared pre-filter: NB and Fisher ALWAYS use the same tau.
        # This removes the confound in the old pipeline where NB got tau=0.20
        # at d=30 while Fisher got tau=0.10 (an unfair, favourable comparison).
        tau = tau_override if tau_override is not None else 0.10
        print(f'FAIR shared pre-filter tau = {tau} (identical for both methods)')

        res = {}
        # scCausal NB-LR with method-of-moments dispersion (the improved method).
        t0 = time.time()
        edges_nb_m = pc_nb(Xd, d, alpha=0.05, tau=tau, use_moment=True)
        el = time.time() - t0
        n_e, n_h, prec = score(edges_nb_m, Gd, gene2s, ppi)
        res['scCausal_NBLR_moment'] = {"edges": n_e, "string_hits": n_h,
                                       "precision": round(prec, 2), "time_s": round(el, 1)}
        print(f'  {"scCausal NB (moment)":<20} edges={n_e:5d}  STRING hits={n_h:4d}  precision={prec:5.2f}%  ({el:.0f}s)')

        # scCausal NB-LR with the statsmodels fixed alpha=1.0 (transparency
        # control: shows whether the adaptive dispersion actually helps).
        t0 = time.time()
        edges_nb_f = pc_nb(Xd, d, alpha=0.05, tau=tau, use_moment=False)
        el = time.time() - t0
        n_e, n_h, prec = score(edges_nb_f, Gd, gene2s, ppi)
        res['scCausal_NBLR_fixed'] = {"edges": n_e, "string_hits": n_h,
                                      "precision": round(prec, 2), "time_s": round(el, 1)}
        print(f'  {"scCausal NB (fixed a=1)":<20} edges={n_e:5d}  STRING hits={n_h:4d}  precision={prec:5.2f}%  ({el:.0f}s)')

        # Fisher's z standard PC (the baseline), same tau.
        t0 = time.time()
        edges_fz = pc_fz(Xd, d, alpha=0.05, tau=tau)
        el = time.time() - t0
        n_e, n_h, prec = score(edges_fz, Gd, gene2s, ppi)
        res['Fisher_z'] = {"edges": n_e, "string_hits": n_h,
                           "precision": round(prec, 2), "time_s": round(el, 1)}
        print(f'  {"Fisher-z (baseline)":<20} edges={n_e:5d}  STRING hits={n_h:4d}  precision={prec:5.2f}%  ({el:.0f}s)')

        res["meta"] = {"d": d, "n": Xd.shape[0], "zero_pct": round(float(100 * (Xd == 0).mean()), 1),
                       "alpha_hat": round(alpha_est, 4), "tau": tau, "fair": True}
        all_results[str(d)] = res
        out = os.path.join(ROOT, 'results', f'table1_{d}.json')
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=2)
        print(f'  -> {out}')

    # Combined, authoritative output JSON (the single source of truth for Table 1).
    combined = os.path.join(ROOT, 'results', 'table1_fair_all.json')
    with open(combined, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print(f'\n  FAIR TABLE (NB vs Fisher, identical tau) -> {combined}')
    print(f'  {"d":>4} | {"NB(moment) E":>11} | {"NB(m) P%":>8} | {"NB(fixed) P%":>10} | {"Fisher E":>9} | {"Fisher P%":>9} | {"Delta P%":>8}')
    print(f'  ' + '-' * 78)
    for d in ds:
        r = all_results.get(str(d), {})
        nb = r.get('scCausal_NBLR_moment', {})
        nb_f = r.get('scCausal_NBLR_fixed', {})
        fz = r.get('Fisher_z', {})
        delta = round(nb.get('precision', 0) - fz.get('precision', 0), 2)
        print(f'  {d:>4} | {nb.get("edges",0):>11} | {nb.get("precision",0):>8.2f} | '
              f'{nb_f.get("precision",0):>10.2f} | {fz.get("edges",0):>9} | '
              f'{fz.get("precision",0):>9.2f} | {delta:>+8.2f}')


if __name__ == "__main__":
    main()
