"""sim_multiseed.py

Multi-seed paired simulation benchmark for the scCausal paper.

Extends scripts/_fix_sim_mle.py -- which only ever ran d=30 -- to the other
dimensions quoted in the manuscript (d=50 and d=100), and adds the variant the
paper actually ships (method-of-moments dispersion, not just MLE).

Data generation is copied verbatim from _fix_sim_mle.py so the two are directly
comparable, and so d=30 can be used as a regression check: rerunning d=30 with
100 seeds must reproduce fix_sim_mle.json exactly (same seeds, same generator,
same CI tests).

  make_dag(d, ne)        linear DAG, weights U[0.2, 0.8], i<j
  gen_nb(adj, n, 1.5)    Poisson(Gamma(1/disp, mu/disp)) = true NB counts

Four CI tests are compared inside an identical PC skeleton search
(tau = 0.10, max_k = 1, alpha = 0.05 -- shared, so the comparison is fair):

  fz    Fisher's z on log(1+x)                          (baseline)
  mle   NB likelihood ratio, alpha by MLE               (slow, gold standard)
  mom   NB likelihood ratio, alpha by method of moments (what the paper ships)
  glm   NB likelihood ratio, alpha fixed at 1.0         (the statsmodels default)

Scored by F1 against the true DAG skeleton, paired across seeds.

Usage
-----
    python sim_multiseed.py --d 30 --n 300 --ne 20 --seeds 100
    python sim_multiseed.py --d 50 --n 500 --ne 30 --seeds 100 --jobs 8
"""
import sys, os, time, json, argparse, warnings
import numpy as np

# statsmodels emits a ValueWarning on every single GLM fit when alpha is left at
# its default (the alpha=1.0 control).  With O(1e5) fits per dimension that
# floods the console and hides the actual results.
warnings.filterwarnings('ignore')

from scipy.stats import chi2, norm, pearsonr, wilcoxon, ttest_rel
from statsmodels.genmod.families import NegativeBinomial as GNM_NB
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as NBCOUNT

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.abspath(os.path.join(HERE, '..', 'results'))
DISP = 1.5


# ---------------------------------------------------------------- data ----
def make_dag(d, ne):
    adj = np.zeros((d, d))
    e = 0
    while e < ne:
        i, j = np.random.randint(0, d, 2)
        if i < j and adj[i, j] == 0:
            adj[i, j] = np.random.uniform(0.2, 0.8)
            e += 1
    return adj


def gen_nb(adj, n, disp=DISP):
    d = adj.shape[0]
    X = np.zeros((n, d))
    for j in range(d):
        mu = np.ones(n) * 0.5
        for i in range(d):
            if adj[i, j] != 0:
                mu += adj[i, j] * X[:, i]
        mu = np.maximum(mu, 0.1)
        X[:, j] = np.random.poisson(np.random.gamma(1.0 / disp, mu / disp))
    return X, adj


# ------------------------------------------------------------------ CI ----
def estimate_alpha_moment(X):
    """NB2 dispersion by method of moments (same estimator as reproduce_table1)."""
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    ok = mu > 0
    a = (var[ok] - mu[ok]) / np.maximum(mu[ok] ** 2, 1e-8)
    return float(np.median(np.clip(a, 1e-4, None)))


def nb_lr_glm(X, i, j, cond, alpha_sig=0.05):
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    try:
        fam = GNM_NB()          # alpha fixed at the statsmodels default, 1.0
        m0 = sm.GLM(y, Xn, family=fam).fit(maxiter=50, disp=0)
        if m0.llf is None:
            return 1.0
        Xa = np.column_stack([Xn, X[:, i]])
        m1 = sm.GLM(y, Xa, family=fam).fit(maxiter=50, disp=0)
        if m1.llf is None:
            return 1.0
        return 1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1)
    except Exception:
        return 1.0


def nb_lr_moment(X, i, j, cond, alpha_hat, alpha_sig=0.05):
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    try:
        fam = GNM_NB(alpha=alpha_hat)
        m0 = sm.GLM(y, Xn, family=fam).fit(maxiter=50, disp=0)
        if m0.llf is None:
            return 1.0
        Xa = np.column_stack([Xn, X[:, i]])
        m1 = sm.GLM(y, Xa, family=fam).fit(maxiter=50, disp=0)
        if m1.llf is None:
            return 1.0
        return 1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1)
    except Exception:
        return 1.0


def nb_lr_mle(X, i, j, cond, alpha_sig=0.05):
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    if Xn.shape[1] == 1:
        Xa = sm.add_constant(X[:, i])
    else:
        Xa = np.column_stack([Xn, X[:, i]])
    try:
        m0 = NBCOUNT(y, Xn).fit(method='nm', maxiter=200, disp=False)
        if m0.llf is None or not np.isfinite(m0.llf):
            return 1.0
        m1 = NBCOUNT(y, Xa).fit(method='nm', maxiter=200, disp=False)
        if m1.llf is None or not np.isfinite(m1.llf):
            return 1.0
        return 1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1)
    except Exception:
        return 1.0


# --------------------------------------------------------------- search ----
def pc_for(X, d, ci_fn, alpha_sig=0.05, tau=0.10, max_k=1):
    corr = np.corrcoef(np.log1p(X).T)
    edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(corr[i, j]) > tau}
    edges = {(i, j) for (i, j) in edges if ci_fn(X, i, j, []) < alpha_sig}
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
                if ci_fn(X, i, j, [k]) >= alpha_sig:
                    removed.add((i, j))
                    break
        edges -= removed
    return sorted(edges)


def pc_fz(X, d, alpha_sig=0.05, tau=0.10, max_k=1):
    n = X.shape[0]
    logX = np.log1p(X)
    corr = np.corrcoef(logX.T)
    edges = set()
    for i in range(d):
        for j in range(i + 1, d):
            if abs(corr[i, j]) > tau:
                z = 0.5 * np.log((1 + corr[i, j]) / max(1 - corr[i, j], 1e-10))
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 3)))) < alpha_sig:
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
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 4)))) >= alpha_sig:
                    removed.add((i, j))
                    break
        edges -= removed
    return sorted(edges)


def scores(es, ts):
    """F1 plus its ingredients, so an F1 gap can be attributed to precision or
    recall rather than to the operating point alone."""
    es, ts = set(es), set(ts)
    tp = len(es & ts)
    fp = len(es - ts)
    fn = len(ts - es)
    prec = tp / len(es) if es else 0.0
    rec = tp / len(ts) if ts else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, prec, rec, tp, fp, fn


# ----------------------------------------------------------------- run ----
def run_seed(job):
    seed, d, n, ne = job
    np.random.seed(seed)
    adj = make_dag(d, ne)
    X, _ = gen_nb(adj, n)
    ts = {(i, j) for i in range(d) for j in range(d) if adj[i, j] != 0}
    out = {'seed': seed}
    a_hat = estimate_alpha_moment(X)
    out['alpha_hat'] = round(a_hat, 4)

    runs = [
        ('mle', lambda: pc_for(X, d, nb_lr_mle)),
        ('mom', lambda: pc_for(X, d, lambda A, i, j, c: nb_lr_moment(A, i, j, c, a_hat))),
        ('glm', lambda: pc_for(X, d, nb_lr_glm)),
        ('fz', lambda: pc_fz(X, d)),
    ]
    for key, fn in runs:
        t = time.time()
        e = fn()
        f1v, p, r, tp, fp, fnn = scores(e, ts)
        out[key] = round(f1v, 4)
        out['p_' + key] = round(p, 4)
        out['r_' + key] = round(r, 4)
        out['tp_' + key] = tp
        out['fp_' + key] = fp
        out['fn_' + key] = fnn
        out['n_' + key] = len(e)
        out['t_' + key] = round(time.time() - t, 3)

    out['true_edges'] = len(ts)
    return out


def paired(a, b, name, N, rng):
    dd = a - b
    try:
        _, pw = wilcoxon(a, b)
    except Exception:
        pw = float('nan')
    try:
        _, pt = ttest_rel(a, b)
    except Exception:
        pt = float('nan')
    bs = dd[rng.integers(0, N, (20000, N))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {
        'label': name,
        'mean_diff': round(float(dd.mean()), 4),
        'median_diff': round(float(np.median(dd)), 4),
        'sd_diff': round(float(dd.std(ddof=1)), 4),
        'wins': int((a > b).sum()),
        'n': int(N),
        'wilcoxon_p': float(pw),
        'paired_t_p': float(pt),
        'boot_ci95': [round(float(lo), 4), round(float(hi), 4)],
        'excludes_zero': bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=30)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--ne', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    jobs = [(s, args.d, args.n, args.ne) for s in seeds]
    print('d=%d n=%d ne=%d seeds=%d jobs=%d' % (args.d, args.n, args.ne, args.seeds, args.jobs))
    t0 = time.time()
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            res = pool.map(run_seed, jobs)
    else:
        res = []
        for j in jobs:
            res.append(run_seed(j))
            print('  seed=%d mle=%.4f mom=%.4f glm=%.4f fz=%.4f' %
                  (res[-1]['seed'], res[-1]['mle'], res[-1]['mom'],
                   res[-1]['glm'], res[-1]['fz']), flush=True)
    wall = time.time() - t0

    N = len(res)
    keys = ('mle', 'mom', 'glm', 'fz')
    arr = {k: np.array([r[k] for r in res]) for k in keys}
    parr = {k: np.array([r['p_' + k] for r in res]) for k in keys}
    rarr = {k: np.array([r['r_' + k] for r in res]) for k in keys}
    narr = {k: np.array([r['n_' + k] for r in res]) for k in keys}
    rng = np.random.default_rng(0)

    agg = {'d': args.d, 'n': args.n, 'ne': args.ne, 'seeds': N, 'wall_s': round(wall, 1)}
    for k in keys:
        agg['%s_f1' % k] = round(float(arr[k].mean()), 4)
        agg['%s_f1_std' % k] = round(float(arr[k].std()), 4)
        agg['%s_prec' % k] = round(float(parr[k].mean()), 4)
        agg['%s_rec' % k] = round(float(rarr[k].mean()), 4)
        agg['%s_edges' % k] = round(float(narr[k].mean()), 1)
    agg['paired'] = [paired(arr[k], arr['fz'], '%s vs fz' % k, N, rng)
                     for k in ('mle', 'mom', 'glm')]
    # An F1 gap is only meaningful if it is not just fewer edges: report the
    # precision and recall gaps separately.
    agg['paired_precision'] = [paired(parr[k], parr['fz'], '%s vs fz (P)' % k, N, rng)
                               for k in ('mle', 'mom', 'glm')]
    agg['paired_recall'] = [paired(rarr[k], rarr['fz'], '%s vs fz (R)' % k, N, rng)
                            for k in ('mle', 'mom', 'glm')]

    print('\n' + '=' * 78)
    print('d=%d  n=%d  ne=%d  seeds=%d  (%.0fs wall)' % (args.d, args.n, args.ne, N, wall))
    print('  %-13s %8s %8s %8s %8s %7s' % ('method', 'F1 mean', 'F1 sd', 'prec', 'recall', 'edges'))
    for k, lbl in [('fz', 'Fisher z'), ('mle', 'NB-MLE'), ('mom', 'NB-moment'), ('glm', 'NB-GLM a=1')]:
        print('  %-13s %8.4f %8.4f %8.4f %8.4f %7.1f'
              % (lbl, arr[k].mean(), arr[k].std(), parr[k].mean(), rarr[k].mean(), narr[k].mean()))
    print()
    for tag, plist in [('F1', agg['paired']), ('P ', agg['paired_precision']),
                       ('R ', agg['paired_recall'])]:
        for p in plist:
            print('  [%s] %-18s diff %+.4f  wins %d/%d  Wilcoxon p=%.4g  CI95 [%+.4f, %+.4f] %s'
                  % (tag, p['label'], p['mean_diff'], p['wins'], p['n'], p['wilcoxon_p'],
                     p['boot_ci95'][0], p['boot_ci95'][1],
                     'EXCLUDES 0' if p['excludes_zero'] else 'includes 0'))
        print()

    os.makedirs(RESULT_DIR, exist_ok=True)
    dst = os.path.join(RESULT_DIR, 'sim_multiseed_d%d_n%d.json' % (args.d, args.n))
    with open(dst, 'w') as fh:
        json.dump({'agg': agg, 'per_seed': res}, fh, indent=2)
    print('\n-> %s' % dst)


if __name__ == '__main__':
    main()
