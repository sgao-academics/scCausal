"""
Dispersion x zero-inflation sweep for the NB-LR vs Fisher's z comparison.

Motivation
----------
The simulation benchmark in the manuscript (Table 2) is generated at a single
NB2 dispersion, alpha = 1.5, whereas the estimated dispersion on the real data
spans roughly 0.63 (Paul15) to 7.78 (PBMC at d=200).  A reviewer is entitled to
ask whether the headline finding -- "Fisher's z is significantly better at
n <= 300 and NB-LR from n >= 500 upward" -- survives at the dispersion levels
that the real data actually exhibit, and whether it survives when the generator
is zero-inflated (which is what scRNA-seq counts are).

This script answers both questions directly, using the SAME estimator the
manuscript ships (method-of-moments dispersion) and the SAME PC search
(k_max = 1, shared pre-filter tau = 0.10), so the numbers are directly
comparable to Table 2.

Grid
----
  (A) pure NB      : alpha in {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0}
  (B) zero-inflated: pi0 in {0, 0.3, 0.5, 0.7} at alpha = 1.5
  each at n in {300, 500}, d = 30, ne = 20, 100 seeds.

Writes results/sim_disp_sweep.json.

Usage
-----
    python scripts/sim_disp_sweep.py --seeds 100
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import chi2, norm, pearsonr, wilcoxon
from statsmodels.genmod.families import NegativeBinomial as GNM_NB
import statsmodels.api as sm

# statsmodels emits a PerfectSeparationWarning on every degenerate GLM fit and
# numpy a divide warning when a gene is all-zero (high-dispersion cells); with
# O(1e5) fits these flood the console and hide the results table.
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))

D = 30
NE = 20
TAU = 0.10
ALPHA = 0.05
MAX_K = 1


# --------------------------------------------------------------- generator --
def make_dag(d, ne, rng):
    adj = np.zeros((d, d))
    e = 0
    while e < ne:
        i, j = rng.integers(0, d, 2)
        if i < j and adj[i, j] == 0:
            adj[i, j] = rng.uniform(0.2, 0.8)
            e += 1
    return adj


def gen_counts(adj, n, disp, pi0, rng):
    """NB2 counts (variance = mu + mu^2/disp) with optional zero-inflation pi0."""
    d = adj.shape[0]
    X = np.zeros((n, d))
    for j in range(d):
        mu = np.ones(n) * 0.5
        for i in range(d):
            if adj[i, j] != 0:
                mu += adj[i, j] * X[:, i]
        mu = np.maximum(mu, 0.1)
        # Gamma-Poisson mixture == NB2 with dispersion `disp`
        lam = rng.gamma(1.0 / disp, mu / disp)
        X[:, j] = rng.poisson(lam)
    if pi0 > 0:
        mask = rng.random((n, d)) < pi0
        X[mask] = 0.0
    return X


# --------------------------------------------------------------------- CI ---
def estimate_alpha_moment(X):
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    ok = mu > 0
    a = (var[ok] - mu[ok]) / np.maximum(mu[ok] ** 2, 1e-8)
    return float(np.median(np.clip(a, 1e-4, None)))


def nb_lr(X, i, j, cond, alpha_hat, alpha_sig=ALPHA):
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
        return float(1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1))
    except Exception:
        return 1.0


def pc_nb(X, d, alpha_hat, alpha_sig=ALPHA, tau=TAU, max_k=MAX_K):
    corr = np.corrcoef(np.log1p(X).T)
    edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(corr[i, j]) > tau}
    edges = {(i, j) for (i, j) in edges if nb_lr(X, i, j, [], alpha_hat, alpha_sig) < alpha_sig}
    if max_k >= 1:
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
                if nb_lr(X, i, j, [k], alpha_hat, alpha_sig) >= alpha_sig:
                    edges.discard((i, j))
                    break
    return edges


def pc_fz(X, d, alpha_sig=ALPHA, tau=TAU, max_k=MAX_K):
    n = X.shape[0]
    logX = np.log1p(X)
    corr = np.corrcoef(logX.T)
    edges = {tuple(sorted((i, j))) for i in range(d) for j in range(i + 1, d)
             if abs(corr[i, j]) > tau}
    edges = {(i, j) for (i, j) in edges
             if 2 * (1 - norm.cdf(abs(0.5 * np.log((1 + corr[i, j]) / max(1 - corr[i, j], 1e-10)))
                                  * np.sqrt(n - 3))) < alpha_sig}
    if max_k >= 1:
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
                ri = logX[:, i] - A @ np.linalg.lstsq(A, logX[:, i], rcond=None)[0]
                rj = logX[:, j] - A @ np.linalg.lstsq(A, logX[:, j], rcond=None)[0]
                r, _ = pearsonr(ri, rj)
                z = 0.5 * np.log((1 + r) / max(1 - r, 1e-10))
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 4)))) >= alpha_sig:
                    edges.discard((i, j))
                    break
    return edges


def f1(edges, true_edges):
    if not edges:
        return 0.0, 0.0, 0.0
    tp = len(edges & true_edges)
    p = tp / len(edges)
    r = tp / len(true_edges) if true_edges else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) > 0 else 0.0)


# ------------------------------------------------------------------- sweep --
def run_cell(n, disp, pi0, seeds, tag):
    fnb, ffz = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        adj = make_dag(D, NE, rng)
        true_edges = {(i, j) for i in range(D) for j in range(D) if adj[i, j] != 0}
        X = gen_counts(adj, n, disp, pi0, rng)
        a_hat = estimate_alpha_moment(X)
        _, _, f_nb = f1(pc_nb(X, D, a_hat), true_edges)
        _, _, f_fz = f1(pc_fz(X, D), true_edges)
        fnb.append(f_nb)
        ffz.append(f_fz)
    fnb, ffz = np.array(fnb), np.array(ffz)
    diff = fnb - ffz
    try:
        pv = float(wilcoxon(fnb, ffz).pvalue)
    except Exception:
        pv = float("nan")
    sd = np.sqrt((fnb.var(ddof=1) + ffz.var(ddof=1)) / 2)
    rec = {
        "tag": tag, "n": n, "disp": disp, "pi0": pi0, "seeds": seeds,
        "nb_f1": round(float(fnb.mean()), 4), "nb_sd": round(float(fnb.std(ddof=1)), 4),
        "fz_f1": round(float(ffz.mean()), 4), "fz_sd": round(float(ffz.std(ddof=1)), 4),
        "delta_f1": round(float(diff.mean()), 4),
        "cohen_d": round(float(diff.mean() / sd), 3) if sd > 0 else None,
        "nb_wins": int((diff > 0).sum()), "p_wilcoxon": pv,
    }
    print("  %-16s n=%-4d disp=%-4.1f pi0=%-4.1f  F1: NB=%.4f FZ=%.4f  dF1=%+.4f  wins=%d/%d  p=%.2g"
          % (tag, n, disp, pi0, rec["nb_f1"], rec["fz_f1"], rec["delta_f1"],
             rec["nb_wins"], seeds, pv), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = 10 if args.quick else args.seeds

    out = []
    t0 = time.time()
    print("=== (A) pure NB: dispersion sweep (d=30, ne=20, %d seeds) ===" % seeds)
    for n in (300, 500):
        for disp in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            out.append(run_cell(n, disp, 0.0, seeds, "nb"))
    print("=== (B) zero-inflated: pi0 sweep at alpha=1.5 ===")
    for n in (300, 500):
        for pi0 in (0.3, 0.5, 0.7):
            out.append(run_cell(n, 1.5, pi0, seeds, "zinb"))

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "sim_disp_sweep.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"d": D, "ne": NE, "tau": TAU, "alpha_sig": ALPHA,
                   "seeds": seeds, "results": out}, fh, indent=2)
    print("\nwall: %.0fs   saved: %s" % (time.time() - t0, path))

    print("\n=== SUMMARY (delta F1 = NB - Fisher) ===")
    print("  %-6s %-6s %-6s %9s %8s" % ("n", "disp", "pi0", "dF1", "wins"))
    for r in out:
        print("  %-6d %-6.1f %-6.1f %+9.4f %3d/%d" %
              (r["n"], r["disp"], r["pi0"], r["delta_f1"], r["nb_wins"], r["seeds"]))


if __name__ == "__main__":
    main()
