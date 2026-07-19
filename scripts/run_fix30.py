import config  # scCausal data path configuration
"""Fix two failed experiments:
1. 30-seed simulation with REAL NB-LR and Fisher's z tests
2. STRING multi-threshold validation
"""
import sys, os, json, time, gzip, numpy as np
from scipy.stats import chi2, norm, pearsonr
from statsmodels.genmod.families import NegativeBinomial
import statsmodels.api as sm

CKPT = os.path.join(os.path.dirname(__file__), '..', 'results', '_fix30_ckpt.json')
SEEDS = 30
results = {}

# ============================================================
# EXPERIMENT 1: 30-seed simulation with REAL tests
# ============================================================
print("=" * 60)
print("EXPERIMENT 1: 30-seed simulation (REAL NB-LR + Fisher's z)")
print("=" * 60)

def make_dag(d, n_edges):
    adj = np.zeros((d, d))
    edges = 0
    while edges < n_edges:
        i, j = np.random.randint(0, d, 2)
        if i < j and adj[i, j] == 0:
            adj[i, j] = np.random.uniform(0.2, 0.8)
            edges += 1
    return adj

def gen_nb_data(adj, n, dispersion=1.5):
    """Generate NB-distributed data from linear DAG."""
    d = adj.shape[0]
    X = np.zeros((n, d))
    order = list(range(d))
    np.random.shuffle(order)
    for j in order:
        mu = np.ones(n) * 0.5
        for i in range(d):
            if adj[i, j] != 0:
                mu += adj[i, j] * X[:, i]
        mu = np.maximum(mu, 0.1)
        r = 1.0 / max(dispersion, 0.01)
        X[:, j] = np.random.gamma(r, mu / r)
    X = np.random.poisson(X).astype(np.float64)
    return X, adj

def real_nb_lr_test(X, i, j):
    """Real NB LR test for marginal independence."""
    y = X[:, j]; x = X[:, i]
    Xn = np.ones((len(y), 1))
    Xa = np.column_stack([np.ones((len(y), 1)), x])
    try:
        m0 = sm.GLM(y, Xn, family=NegativeBinomial()).fit(maxiter=50, disp=0)
        m1 = sm.GLM(y, Xa, family=NegativeBinomial()).fit(maxiter=50, disp=0)
        stat = max(2 * (m1.llf - m0.llf), 0)
        return 1 - chi2.cdf(stat, 1)
    except:
        return 1.0

def real_fisher_z_test(X, i, j):
    """Real Fisher's z partial correlation test (marginal)."""
    r, _ = pearsonr(X[:, i], X[:, j])
    z = 0.5 * np.log((1 + r) / max(1 - r, 1e-10))
    z_stat = z * np.sqrt(X.shape[0] - 3)
    return 2 * (1 - norm.cdf(abs(z_stat)))

def run_simulation(d, n_cells, true_edges, alpha=0.05, tau=0.10):
    """Run true NB-LR vs Fisher's z on synthetic data across 30 seeds."""
    f1_nb = []; f1_fz = []; prec_nb = []; prec_fz = []
    rec_nb = []; rec_fz = []
    nb_wins = 0; fz_wins = 0
    
    for seed in range(SEEDS):
        np.random.seed(seed)
        adj = make_dag(d, true_edges)
        X, _ = gen_nb_data(adj, n_cells)
        
        true_set = {(i, j) for i in range(d) for j in range(d) if adj[i, j] != 0}
        
        # Correlation pre-filter
        logX = np.log1p(X)
        corr = np.corrcoef(logX.T)
        
        # NB-LR: only test pairs passing corr filter
        nb_edges = set()
        for i in range(d):
            for j in range(i+1, d):
                if abs(corr[i, j]) > tau:
                    p = real_nb_lr_test(X, i, j)
                    if p < alpha:
                        nb_edges.add((i, j))
        
        # Fisher's z: same corr filter
        fz_edges = set()
        for i in range(d):
            for j in range(i+1, d):
                if abs(corr[i, j]) > tau:
                    p = real_fisher_z_test(X, i, j)
                    if p < alpha:
                        fz_edges.add((i, j))
        
        # Compute metrics for both
        for name, e_set in [("NB", nb_edges), ("FZ", fz_edges)]:
            tp = len(e_set & true_set)
            prec = tp / len(e_set) if e_set else 0
            rec = tp / len(true_set) if true_set else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if name == "NB":
                f1_nb.append(f1); prec_nb.append(prec); rec_nb.append(rec)
            else:
                f1_fz.append(f1); prec_fz.append(prec); rec_fz.append(rec)
        
        if f1_nb[-1] > f1_fz[-1]: nb_wins += 1
        elif f1_fz[-1] > f1_nb[-1]: fz_wins += 1
    
    return {
        "nb_f1_mean": round(float(np.mean(f1_nb)), 4),
        "nb_f1_std": round(float(np.std(f1_nb)), 4),
        "nb_prec_mean": round(float(np.mean(prec_nb)), 4),
        "nb_rec_mean": round(float(np.mean(rec_nb)), 4),
        "fz_f1_mean": round(float(np.mean(f1_fz)), 4),
        "fz_f1_std": round(float(np.std(f1_fz)), 4),
        "fz_prec_mean": round(float(np.mean(prec_fz)), 4),
        "fz_rec_mean": round(float(np.mean(rec_fz)), 4),
        "nb_wins": nb_wins, "fz_wins": fz_wins, "ties": SEEDS - nb_wins - fz_wins,
        "seeds": SEEDS
    }

for d, n, e in [(30, 300, 20), (50, 500, 30)]:
    print(f"\nd={d}, n={n}, true_edges={e}")
    t0 = time.time()
    r = run_simulation(d, n, e)
    elapsed = time.time() - t0
    print(f"  NB-LR: F1={r['nb_f1_mean']:.4f}+/-{r['nb_f1_std']:.4f}, "
          f"P={r['nb_prec_mean']:.4f}, R={r['nb_rec_mean']:.4f}")
    print(f"  Fisher: F1={r['fz_f1_mean']:.4f}+/-{r['fz_f1_std']:.4f}, "
          f"P={r['fz_prec_mean']:.4f}, R={r['fz_rec_mean']:.4f}")
    print(f"  NB wins {r['nb_wins']}/{SEEDS}, Fisher wins {r['fz_wins']}/{SEEDS}, "
          f"ties {r['ties']}")
    print(f"  Time: {elapsed:.0f}s")
    results[f"sim30_d{d}"] = r

# ============================================================
# EXPERIMENT 2: STRING multi-threshold validation
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: STRING multi-threshold validation")
print("=" * 60)

ALIAS = r'config.get_path("string_aliases")'
PPI = r'config.get_path("string_ppi")'

# Load existing NB-LR edge results from a working checkpoint
# Use the _libsize_main_ckpt.json (it has d=30 results with gene list)
import scanpy as sc
adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '..', 'data', 'pbmc3k_filtered.h5ad'))
X_all = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X, dtype=np.float32)
all_genes = list(adata.var_names.astype(str))
variances = X_all.var(axis=0)
ranked = np.argsort(variances)[::-1]

# Re-run NB-LR at d=30 once to get edge list + gene names
d30_idx = ranked[:30]
X30 = X_all[:, d30_idx]
g30 = [all_genes[i] for i in d30_idx]

# Run scCausal at d=30 to get edge list
def real_nb_lr_edge(X, i, j):
    y = X[:, j]; x = X[:, i]
    Xn = np.ones((len(y), 1))
    Xa = np.column_stack([np.ones((len(y), 1)), x])
    try:
        m0 = sm.GLM(y, Xn, family=NegativeBinomial()).fit(maxiter=50, disp=0)
        m1 = sm.GLM(y, Xa, family=NegativeBinomial()).fit(maxiter=50, disp=0)
        stat = max(2 * (m1.llf - m0.llf), 0)
        return 1 - chi2.cdf(stat, 1)
    except:
        return 1.0

print("Computing NB-LR edges at d=30...")
t0 = time.time()
corr = np.corrcoef(np.log1p(X30).T)
nb_edges = []
for i in range(30):
    for j in range(i+1, 30):
        if abs(corr[i, j]) > 0.10:
            p = real_nb_lr_edge(X30, i, j)
            if p < 0.05:
                # 1st-order check
                independent = True
                for k in range(30):
                    if k != i and k != j and abs(corr[i, k]) > 0.10 and abs(corr[j, k]) > 0.10:
                        # Quick 1st-order CI test
                        y = X30[:, j]
                        Xn = np.column_stack([np.ones(len(y)), X30[:, k]])
                        Xa = np.column_stack([np.ones(len(y)), X30[:, k], X30[:, i]])
                        try:
                            m0 = sm.GLM(y, Xn, family=NegativeBinomial()).fit(maxiter=50, disp=0)
                            m1 = sm.GLM(y, Xa, family=NegativeBinomial()).fit(maxiter=50, disp=0)
                            stat = max(2 * (m1.llf - m0.llf), 0)
                            if 1 - chi2.cdf(stat, 1) >= 0.05:
                                independent = False
                                break
                        except:
                            pass
                if independent:
                    nb_edges.append((i, j))
print(f"  {len(nb_edges)} edges in {time.time()-t0:.0f}s")

# Validate against multiple STRING thresholds
for thresh in [400, 500, 600, 700, 800, 900]:
    # Load STRING at this threshold
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
                if int(p[-1]) >= thresh:
                    ppi.add((p[0], p[1]))
            except: pass
    
    hits = 0
    for (i, j) in nb_edges:
        g1 = g30[i].upper(); g2 = g30[j].upper()
        s1 = s2s.get(g1); s2 = s2s.get(g2)
        if s1 and s2 and ((s1, s2) in ppi or (s2, s1) in ppi):
            hits += 1
    pct = round(100.0 * hits / len(nb_edges), 1) if nb_edges else 0.0
    print(f"  STRING >= {thresh}: {hits}/{len(nb_edges)} ({pct}%)")
    results[f"string_thresh_{thresh}"] = {"hits": hits, "total": len(nb_edges), "pct": pct}

# Save
with open(CKPT, 'w') as f:
    json.dump(results, f, indent=2)
print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))
