import config  # scCausal data path configuration
"""Paul15 hematopoietic differentiation data -- cross-tissue validation."""
import sys, os, json, time, gzip, numpy as np
import scanpy as sc
from scipy.stats import chi2
from statsmodels.genmod.families import NegativeBinomial
import statsmodels.api as sm

CKPT = os.path.join(os.path.dirname(__file__), '..', 'results', '_paul_ckpt.json')
ALIAS = config.resolve('string_aliases', 'SC_CAUSAL_STRING_ALIASES')
PPI = config.resolve('string_ppi', 'SC_CAUSAL_STRING_PPI')

def load_string():
    s2s = {}
    with gzip.open(ALIAS, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.strip().split('\t')
            if len(p) >= 2 and not p[1].isdigit():
                s2s[p[1].upper()] = p[0]
    ppi = set()
    with gzip.open(PPI, 'rt', encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.strip().split()
            try:
                if int(p[-1]) >= 700:
                    ppi.add((p[0], p[1]))
            except: pass
    return s2s, ppi

def validate(edges, genes, s2s, ppi):
    hits = 0
    for (i, j) in edges:
        g1 = genes[i].upper(); g2 = genes[j].upper()
        s1 = s2s.get(g1); s2 = s2s.get(g2)
        if s1 and s2 and ((s1, s2) in ppi or (s2, s1) in ppi):
            hits += 1
    return hits

def estimate_alpha_moment(X):
    """NB2 dispersion via the method of moments: Var = mu + alpha*mu^2."""
    mu = X.mean(axis=0); var = X.var(axis=0)
    valid = mu > 0
    alphas = (var[valid] - mu[valid]) / np.maximum(mu[valid] ** 2, 1e-8)
    alphas = np.clip(alphas, 1e-4, None)
    return float(np.median(alphas))

def nb_lr(X, i, j, cond, alpha=None):
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    try:
        fam = NegativeBinomial(alpha=alpha) if alpha is not None else NegativeBinomial()
        m0 = sm.GLM(y, Xn, family=fam).fit(maxiter=50, disp=0)
        ll0 = m0.llf if m0.llf is not None else -1e9
    except: return 1.0
    Xa = np.column_stack([Xn, X[:, i]]) if Xc is not None else np.column_stack([np.ones((len(y), 1)), X[:, i]])
    try:
        m1 = sm.GLM(y, Xa, family=fam).fit(maxiter=50, disp=0)
        ll1 = m1.llf if m1.llf is not None else -1e9
    except: return 1.0
    return 1 - chi2.cdf(max(2*(ll1-ll0), 0), 1)

def pc_skeleton(X, genes, d, alpha=0.05, tau=0.10, use_moment=True):
    alpha_est = estimate_alpha_moment(X[:, :d]) if use_moment else None
    corr = np.corrcoef(np.log1p(X).T)
    edges = {(i, j) for i in range(d) for j in range(i+1, d) if abs(corr[i, j]) > tau}
    # Marginal
    new_e = set()
    for (i, j) in edges:
        if nb_lr(X, i, j, [], alpha=alpha_est) < alpha: new_e.add((i, j))
    edges = new_e
    # 1st-order
    removed = set()
    for (i, j) in list(edges):
        nb_i = set(); nb_j = set()
        for (a, b) in edges:
            if a == i and b != j: nb_i.add(b)
            elif b == i and a != j: nb_i.add(a)
            if a == j and b != i: nb_j.add(b)
            elif b == j and a != i: nb_j.add(a)
        for k in list(nb_i & nb_j)[:5]:
            if nb_lr(X, i, j, [k], alpha=alpha_est) >= alpha:
                removed.add((i, j)); break
    return sorted(edges - removed)

print("Loading Paul15 (hematopoietic differentiation)...")
adata = sc.read_h5ad(config.resolve('paul15', 'SC_CAUSAL_PAUL15'))
X_all = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X, dtype=np.float32)
all_genes = list(adata.var_names.astype(str))
print(f"  {X_all.shape[0]} cells x {X_all.shape[1]} genes")

# Check for cell type annotations
if 'paul15_clusters' in adata.obs.columns:
    clusters = adata.obs['paul15_clusters']
    print(f"  Cell types: {sorted(clusters.unique())}")
else:
    clusters = None

variances = X_all.var(axis=0)
ranked = np.argsort(variances)[::-1]

s2s, ppi = load_string()
results = {}

for d in [30, 50]:
    idx = ranked[:d]
    X = X_all[:, idx]
    genes = [all_genes[i] for i in idx]
    
    t0 = time.time()
    edges = pc_skeleton(X, genes, d)
    elapsed = time.time() - t0
    sh = validate(edges, genes, s2s, ppi)
    pct = round(100.0*sh/len(edges), 1) if edges else 0.0
    
    # Also run Fisher's z for comparison
    from scipy.stats import norm as norm_dist
    corr = np.corrcoef(np.log1p(X).T)
    fz_edges = set()
    for i in range(d):
        for j in range(i+1, d):
            if abs(corr[i, j]) > 0.10:
                z = 0.5 * np.log((1+corr[i,j])/(1-corr[i,j]+1e-10))
                z_stat = z * np.sqrt(X.shape[0] - 3)
                pval = 2*(1-norm_dist.cdf(abs(z_stat)))
                if pval < 0.05:
                    fz_edges.add((i, j))
    fz_sh = validate(list(fz_edges), genes, s2s, ppi)
    fz_pct = round(100.0*fz_sh/len(fz_edges), 1) if fz_edges else 0.0
    
    print(f"  d={d}: NB-LR {len(edges)} edges @ {pct}%, Fisher {len(fz_edges)} edges @ {fz_pct}%, {elapsed:.0f}s")
    results[f"d{d}"] = {
        "nb_edges": len(edges), "nb_string_pct": pct, "nb_string_hits": sh,
        "fz_edges": len(fz_edges), "fz_string_pct": fz_pct, "fz_string_hits": fz_sh,
        "time_s": round(elapsed), "tissue": "bone_marrow_hematopoietic",
        "cells": X_all.shape[0], "genes_total": X_all.shape[1]
    }

with open(CKPT, 'w') as f:
    json.dump(results, f, indent=2)
print("\n=== PAUL15 RESULTS ===")
print(json.dumps(results, indent=2))
