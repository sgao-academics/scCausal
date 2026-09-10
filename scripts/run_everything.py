import config  # scCausal data path configuration
"""EVERYTHING: Full-scale sweep of all experiments at maximum limits.
Checkpoint-based resume -- safe to interrupt and restart.

Experiments:
A. PBMC 3K: d=[30,50,100,200,300] x (NB-LR + Fisher raw + Fisher log1p)
B. Paul15: d=[30,50,100,200] x (NB-LR + Fisher)
C. Cell-type PBMC: 5 types x d=[30,50] x (NB-LR + Fisher)
D. Cell-type Paul15: 11 types x d=[30,50] x (NB-LR + Fisher)
E. Simulation: d=[30,50,100] x 100 seeds x (NB-LR + Fisher)
F. GENIE3: PBMC d=30/50 + Paul15 d=30/50
G. STRING: all thresholds for all d values
"""
import sys, os, json, time, gzip, numpy as np
from scipy.stats import chi2, norm, pearsonr
from statsmodels.genmod.families import NegativeBinomial
import statsmodels.api as sm
import scanpy as sc

CKPT = os.path.join(os.path.dirname(__file__), '..', 'results', '_everything_ckpt.json')
ALIAS = config.resolve('string_aliases', 'SC_CAUSAL_STRING_ALIASES')
PPI = config.resolve('string_ppi', 'SC_CAUSAL_STRING_PPI')

ckpt = json.load(open(CKPT)) if os.path.exists(CKPT) else {}

def save():
    with open(CKPT, 'w') as f:
        json.dump(ckpt, f, indent=2, default=str)

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
                if int(p[-1]) >= 700: ppi.add((p[0], p[1]))
            except: pass
    return s2s, ppi

def validate(edges, genes, s2s, ppi):
    hits = 0
    for (i, j) in edges:
        g1 = genes[i].upper(); g2 = genes[j].upper()
        s1 = s2s.get(g1); s2 = s2s.get(g2)
        if s1 and s2 and ((s1, s2) in ppi or (s2, s1) in ppi): hits += 1
    return hits

def estimate_alpha_moment(X):
    """NB2 dispersion via the method of moments: Var = mu + alpha*mu^2.

    Closed-form and O(d): alpha_hat(g) = (s^2_g - xbar_g) / xbar_g^2 for each
    expressed gene, and the global dispersion is the median over genes. This
    adapts the NB-LR test to the observed overdispersion instead of fixing the
    statsmodels default alpha = 1.0.
    """
    mu = X.mean(axis=0); var = X.var(axis=0)
    valid = mu > 0
    alphas = (var[valid] - mu[valid]) / np.maximum(mu[valid] ** 2, 1e-8)
    alphas = np.clip(alphas, 1e-4, None)
    return float(np.median(alphas))

def nb_lr(X, i, j, cond, libsize=None, alpha=None):
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    off = libsize if libsize is not None else None
    try:
        fam = NegativeBinomial(alpha=alpha) if alpha is not None else NegativeBinomial()
        m0 = sm.GLM(y, Xn, family=fam, offset=off).fit(maxiter=50, disp=0)
        if m0.llf is None: return 1.0
        Xa = np.column_stack([Xn, X[:, i]]) if Xn.shape[1] > 0 else sm.add_constant(X[:, i].reshape(-1,1))
        m1 = sm.GLM(y, Xa, family=fam, offset=off).fit(maxiter=50, disp=0)
        if m1.llf is None: return 1.0
        return 1 - chi2.cdf(max(2*(m1.llf-m0.llf), 0), 1)
    except: return 1.0

def pc_nb(X, genes, d, alpha=0.05, tau=0.10, max_k=1, use_moment=True):
    if use_moment:
        alpha_est = estimate_alpha_moment(X[:, :d])
    else:
        alpha_est = None
    corr = np.corrcoef(np.log1p(X).T)
    edges = {(i, j) for i in range(d) for j in range(i+1, d) if abs(corr[i, j]) > tau}
    new_e = set()
    for (i, j) in edges:
        if nb_lr(X, i, j, [], alpha=alpha_est) < alpha: new_e.add((i, j))
    edges = new_e
    if max_k >= 1:
        removed = set()
        for (i, j) in list(edges):
            nb_i, nb_j = set(), set()
            for (a, b) in edges:
                if a == i and b != j: nb_i.add(b)
                elif b == i and a != j: nb_i.add(a)
                if a == j and b != i: nb_j.add(b)
                elif b == j and a != i: nb_j.add(a)
            for k in list(nb_i & nb_j)[:5]:
                if nb_lr(X, i, j, [k], alpha=alpha_est) >= alpha:
                    removed.add((i, j)); break
        edges -= removed
    return sorted(edges)

def pc_fz(X, d, alpha=0.05, tau=0.10, max_k=1):
    n = X.shape[0]
    logX = np.log1p(X)
    corr = np.corrcoef(logX.T)
    edges = set()
    for i in range(d):
        for j in range(i+1, d):
            if abs(corr[i, j]) > tau:
                z = 0.5 * np.log((1+corr[i,j])/max(1-corr[i,j], 1e-10))
                if 2*(1-norm.cdf(abs(z*np.sqrt(n-3)))) < alpha:
                    edges.add((i, j))
    if max_k >= 1:
        removed = set()
        for (i, j) in list(edges):
            nb_i, nb_j = set(), set()
            for (a, b) in edges:
                if a == i and b != j: nb_i.add(b)
                elif b == i and a != j: nb_i.add(a)
                if a == j and b != i: nb_j.add(b)
                elif b == j and a != i: nb_j.add(a)
            for k in list(nb_i & nb_j)[:5]:
                A = np.column_stack([np.ones(n), logX[:, k]])
                res_i = logX[:, i] - A @ np.linalg.lstsq(A, logX[:, i], rcond=None)[0]
                res_j = logX[:, j] - A @ np.linalg.lstsq(A, logX[:, j], rcond=None)[0]
                r, _ = pearsonr(res_i, res_j)
                z = 0.5 * np.log((1+r)/max(1-r, 1e-10))
                if 2*(1-norm.cdf(abs(z*np.sqrt(n-4)))) >= alpha:
                    removed.add((i, j)); break
        edges -= removed
    return sorted(edges)

def run_nb_fz(name, X, genes, d, s2s, ppi):
    t0 = time.time()
    nb_edges = pc_nb(X, genes, d)
    nb_t = time.time() - t0
    nb_sh = validate(nb_edges, genes, s2s, ppi)
    nb_pct = round(100.0*nb_sh/len(nb_edges), 1) if nb_edges else 0.0
    
    t0 = time.time()
    fz_edges = pc_fz(X, d)
    fz_t = time.time() - t0
    fz_sh = validate(fz_edges, genes, s2s, ppi)
    fz_pct = round(100.0*fz_sh/len(fz_edges), 1) if fz_edges else 0.0
    
    r = {"nb_edges": len(nb_edges), "nb_string_hits": nb_sh, "nb_string_pct": nb_pct, "nb_time_s": round(nb_t),
         "fz_edges": len(fz_edges), "fz_string_hits": fz_sh, "fz_string_pct": fz_pct, "fz_time_s": round(fz_t)}
    print(f"  {name}: NB {len(nb_edges)}e@{nb_pct}%, FZ {len(fz_edges)}e@{fz_pct}% [{nb_t:.0f}s]")
    return r

# ============ Load shared data ============
s2s, ppi = load_string()
print(f"STRING: {len(ppi)} HC pairs")

pb = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '..', 'data', 'pbmc3k_filtered.h5ad'))
X_pb = pb.X.toarray() if hasattr(pb.X, 'toarray') else np.array(pb.X, dtype=np.float32)
G_pb = list(pb.var_names.astype(str))
V_pb = X_pb.var(axis=0); R_pb = np.argsort(V_pb)[::-1]
print(f"PBMC: {X_pb.shape[0]}x{X_pb.shape[1]}")

pa = sc.read_h5ad(config.resolve('paul15', 'SC_CAUSAL_PAUL15'))
X_pa = pa.X.toarray() if hasattr(pa.X, 'toarray') else np.array(pa.X, dtype=np.float32)
G_pa = list(pa.var_names.astype(str))
V_pa = X_pa.var(axis=0); R_pa = np.argsort(V_pa)[::-1]
pa_ct = pa.obs['paul15_clusters'].values if 'paul15_clusters' in pa.obs.columns else None
pb_ct = pb.obs['cell_type'].values if 'cell_type' in pb.obs.columns else None
print(f"Paul15: {X_pa.shape[0]}x{X_pa.shape[1]}")

# ============================================================
# A. PBMC 3K full sweep
# ============================================================
print("\n" + "="*60)
print("A. PBMC 3K full sweep [30,50,100,200,300]")
for d in [30, 50, 100, 200, 300]:
    key = f"A_pbmc_d{d}"
    if key in ckpt: print(f"  d={d}: SKIP"); continue
    idx = R_pb[:d]; X = X_pb[:, idx]; G = [G_pb[i] for i in idx]
    ckpt[key] = run_nb_fz(f"PBMC_d{d}", X, G, d, s2s, ppi); save()

# ============================================================
# B. Paul15 full sweep
# ============================================================
print("\n" + "="*60)
print("B. Paul15 full sweep [30,50,100,200]")
for d in [30, 50, 100, 200]:
    key = f"B_paul_d{d}"
    if key in ckpt: print(f"  d={d}: SKIP"); continue
    idx = R_pa[:d]; X = X_pa[:, idx]; G = [G_pa[i] for i in idx]
    ckpt[key] = run_nb_fz(f"Paul15_d{d}", X, G, d, s2s, ppi); save()

# ============================================================
# C. Cell-type PBMC
# ============================================================
print("\n" + "="*60)
print("C. Cell-type PBMC [5 types x d=30,50]")
if pb_ct is not None:
    for ct in sorted(set(pb_ct)):
        mask = pb_ct == ct; n = int(mask.sum())
        if n < 100: continue
        for d in [30, 50]:
            key = f"C_pbmc_{ct}_d{d}"
            if key in ckpt: print(f"  {ct}_d{d}: SKIP"); continue
            Xc = X_pb[mask]; Vc = Xc.var(axis=0); Rc = np.argsort(Vc)[::-1][:d]
            Xd = Xc[:, Rc]; Gd = [G_pb[i] for i in Rc]
            ckpt[key] = run_nb_fz(f"PBMC_{ct}_d{d}", Xd, Gd, d, s2s, ppi); save()

# ============================================================
# D. Cell-type Paul15
# ============================================================
print("\n" + "="*60)
print("D. Cell-type Paul15 [11 types x d=30]")
if pa_ct is not None:
    for ct in sorted(set(pa_ct)):
        mask = pa_ct == ct; n = int(mask.sum())
        if n < 50: continue
        key = f"D_paul_{ct}_d30"
        if key in ckpt: print(f"  {ct}_d30: SKIP"); continue
        Xc = X_pa[mask]; Vc = Xc.var(axis=0); Rc = np.argsort(Vc)[::-1][:30]
        Xd = Xc[:, Rc]; Gd = [G_pa[i] for i in Rc]
        ckpt[key] = run_nb_fz(f"Paul15_{ct}_d30", Xd, Gd, 30, s2s, ppi); save()

# ============================================================
# E. Simulation 100 seeds
# ============================================================
print("\n" + "="*60)
print("E. Simulation 100 seeds [d=30,50,100]")
def make_dag(d, ne):
    adj = np.zeros((d, d)); e = 0
    while e < ne:
        i, j = np.random.randint(0, d, 2)
        if i < j and adj[i, j] == 0: adj[i, j] = np.random.uniform(0.2,0.8); e += 1
    return adj

def gen_nb(adj, n, disp=1.5):
    d = adj.shape[0]; X = np.zeros((n, d))
    for j in range(d):
        mu = np.ones(n)*0.5
        for i in range(d):
            if adj[i, j] != 0: mu += adj[i, j] * X[:, i]
        mu = np.maximum(mu, 0.1)
        X[:, j] = np.random.poisson(np.random.gamma(1.0/disp, mu/disp))
    return X, adj

for d, n_cells, true_e in [(30, 300, 20), (50, 500, 30), (100, 700, 40)]:
    key = f"E_sim_d{d}"
    if key in ckpt: print(f"  d={d}: SKIP"); continue
    print(f"  d={d} (100 seeds)...")
    f1_nb, f1_fz = [], []
    for seed in range(100):
        np.random.seed(seed)
        adj = make_dag(d, true_e)
        X, _ = gen_nb(adj, n_cells)
        ts = {(i, j) for i in range(d) for j in range(d) if adj[i, j] != 0}
        nb_e = set(pc_nb(X, [f"G{i}" for i in range(d)], d))
        fz_e = set(pc_fz(X, d))
        for name, es in [("nb", nb_e), ("fz", fz_e)]:
            tp = len(es & ts)
            prec = tp/len(es) if es else 0
            rec = tp/len(ts) if ts else 0
            f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
            if name == "nb": f1_nb.append(f1)
            else: f1_fz.append(f1)
    ckpt[key] = {"nb_f1": round(float(np.mean(f1_nb)),4), "nb_f1_std": round(float(np.std(f1_nb)),4),
                  "fz_f1": round(float(np.mean(f1_fz)),4), "fz_f1_std": round(float(np.std(f1_fz)),4),
                  "nb_wins": sum(1 for a,b in zip(f1_nb,f1_fz) if a>b), "seeds": 100}
    print(f"    NB {ckpt[key]['nb_f1']:.4f}+/-{ckpt[key]['nb_f1_std']:.4f}, FZ {ckpt[key]['fz_f1']:.4f}+/-{ckpt[key]['fz_f1_std']:.4f}, NB wins {ckpt[key]['nb_wins']}/100")
    save()

# ============================================================
# F. GENIE3 comparison
# ============================================================
print("\n" + "="*60)
print("F. GENIE3 on PBMC + Paul15 [d=30,50]")
try:
    from sklearn.ensemble import RandomForestRegressor
    def genie3(X, genes, d, n_trees=100):
        """Simple GENIE3: RandomForest importance for each gene."""
        adj = np.zeros((d, d))
        for j in range(d):
            y = X[:, j]
            X_rest = np.delete(X, j, axis=1)
            rf = RandomForestRegressor(n_estimators=n_trees, random_state=42, n_jobs=-1)
            rf.fit(X_rest, y)
            imps = rf.feature_importances_
            # Keep top edges
            order = np.argsort(imps)[::-1]
            for k in range(min(d//2, len(order))):
                reg = order[k]
                target_idx = reg if reg < j else reg + 1
                adj[reg, j] = imps[reg] if imps[reg] > np.percentile(imps, 75) else 0
        edges = [(i, j) for i in range(d) for j in range(d) if adj[i, j] > 0]
        return edges

    for name, X_all, genes_all, R, d_vals in [
        ("PBMC", X_pb, G_pb, R_pb, [30, 50]),
        ("Paul15", X_pa, G_pa, R_pa, [30, 50])
    ]:
        for d in d_vals:
            key = f"F_genie3_{name}_d{d}"
            if key in ckpt: print(f"  GENIE3_{name}_d{d}: SKIP"); continue
            idx = R[:d]; X = X_all[:, idx]; G = [genes_all[i] for i in idx]
            t0 = time.time()
            edges = genie3(X, G, d)
            t = time.time() - t0
            sh = validate(edges, G, s2s, ppi)
            pct = round(100.0*sh/len(edges), 1) if edges else 0.0
            print(f"  GENIE3_{name}_d{d}: {len(edges)} edges @ {pct}% [{t:.0f}s]")
            ckpt[key] = {"edges": len(edges), "string_hits": sh, "string_pct": pct, "time_s": round(t)}
            save()
except ImportError:
    print("  sklearn not available, skipping GENIE3")

# ============================================================
# G. STRING multi-threshold (all d for PBMC)
# ============================================================
print("\n" + "="*60)
print("G. STRING multi-threshold [PBMC d=30,50,100]")
for d in [30, 50, 100]:
    for thresh in [400, 500, 600, 700, 800, 900]:
        key = f"G_string_d{d}_t{thresh}"
        if key in ckpt: continue
        # Load threshold-specific PPI
        ppi_t = set()
        with gzip.open(PPI, 'rt', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                p = line.strip().split()
                try:
                    if int(p[-1]) >= thresh: ppi_t.add((p[0], p[1]))
                except: pass
        idx = R_pb[:d]; G = [G_pb[i] for i in idx]
        # Get NB edges for this d
        d_key = f"A_pbmc_d{d}"
        if d_key not in ckpt: continue
        nb_edges_raw = ckpt[d_key].get('nb_edges', 0)
        # Re-run quickly to get edge list
        X = X_pb[:, idx]
        edges = pc_nb(X, G, d)
        sh = validate(edges, G, s2s, ppi_t)
        pct = round(100.0*sh/len(edges), 1) if edges else 0.0
        ckpt[key] = {"hits": sh, "total": len(edges), "pct": pct}
        if thresh in [400, 700, 900]:
            print(f"  d={d} STRING>{thresh}: {sh}/{len(edges)} ({pct}%)")
    save()

print("\n=== ALL DONE ===")
done = len(ckpt)
total = 5 + 4 + 10 + 11 + 3 + 4 + 18  # approximate expected total: 55
print(f"Completed: {done}/{total}")
for k in sorted(ckpt.keys()):
    v = ckpt[k]
    if isinstance(v, dict) and 'nb_string_pct' in v:
        print(f"  {k}: NB {v['nb_string_pct']}% FZ {v['fz_string_pct']}%")
    elif isinstance(v, dict) and 'nb_f1' in v:
        print(f"  {k}: NB {v['nb_f1']:.4f} FZ {v['fz_f1']:.4f} wins {v.get('nb_wins','?')}/100")
