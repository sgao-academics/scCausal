import config  # scCausal data path configuration
"""
CRITICAL baselines: Fisher's z PC + Permuted control + FDR correction.
Runs on exact same d=30/50/100/200 gene sets as NB-LR CI sweep.
Determines whether the paper has a valid claim.
"""
import os, sys, json, time, gzip
import numpy as np
import scanpy as sc
from scipy.stats import norm

DEVICE = 'cpu'  # Fisher's z is pure numpy/math, no GPU needed
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CKPT_PATH = os.path.join(RESULT_DIR, '_baseline_checkpoint.json')
D_VALUES = [30, 50, 100, 200]
ALPHA = 0.05
FISHER_TAU = {30: 0.10, 50: 0.12, 100: 0.15, 200: 0.20}  # correlation pre-filter

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_pbmc(d):
    """Load PBMC 3K and select top-d HVGs."""
    adata = sc.read_h5ad(os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad'))
    if hasattr(adata, 'raw') and adata.raw is not None:
        X = np.array(adata.raw.X.todense() if hasattr(adata.raw.X, 'todense')
                      else adata.raw.X, dtype=np.float32)
    else:
        X = np.array(adata.X.todense() if hasattr(adata.X, 'todense')
                      else adata.X, dtype=np.float32)
    genes = list(adata.var_names.astype(str))
    # Select top-d by variance
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[-d:]
    X = X[:, top_idx]
    genes = [genes[i] for i in top_idx]
    return X, genes

# ============================================================
# 2. FISHER'S Z PC (k_max=1)
# ============================================================
def partial_corr(X, i, j, S):
    """Compute partial correlation rho_{ij|S}."""
    n = X.shape[0]
    if len(S) == 0:
        # Marginal correlation
        xi = X[:, i] - X[:, i].mean()
        xj = X[:, j] - X[:, j].mean()
        return np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-10)
    # With conditioning set
    Z = np.column_stack([X[:, k] for k in S])
    Z = Z - Z.mean(axis=0)
    # Residualize
    try:
        beta_i = np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
        beta_j = np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    ri = X[:, i] - Z @ beta_i
    rj = X[:, j] - Z @ beta_j
    return np.dot(ri, rj) / (np.linalg.norm(ri) * np.linalg.norm(rj) + 1e-10)

def fisher_z_test(X, i, j, S):
    """Fisher's z-test for conditional independence."""
    n = X.shape[0]
    rho = partial_corr(X, i, j, S)
    rho = np.clip(rho, -0.999, 0.999)
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - len(S) - 3)
    p = 2 * (1 - norm.cdf(abs(z)))
    return p, rho

def fisher_z_pc(X, genes, d, tau):
    """PC algorithm with Fisher's z-test, k_max=1."""
    n, d_actual = X.shape
    # --- Marginal CI (k=0) ---
    print(f"  Marginal CI: testing {d_actual*(d_actual-1)//2} pairs...")
    adj = set()
    n_tested_marginal = 0
    for i in range(d_actual):
        for j in range(i+1, d_actual):
            # Pre-filter by Pearson correlation
            xi = X[:, i] - X[:, i].mean()
            xj = X[:, j] - X[:, j].mean()
            rho_raw = np.dot(xi, xj) / (np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-10)
            if abs(rho_raw) < tau:
                continue
            n_tested_marginal += 1
            p, _ = fisher_z_test(X, i, j, [])
            if p < ALPHA:
                adj.add((i, j))
    
    print(f"    After marginal CI: {len(adj)} edges (tested {n_tested_marginal})")
    
    # --- First-order CI (k=1) ---
    removed = 0
    n_tested_first = 0
    adj_list = list(adj)
    for (i, j) in adj_list:
        if (i, j) not in adj:
            continue
        # Get common neighbors
        neighbors_i = set()
        neighbors_j = set()
        for (a, b) in adj:
            if a == i: neighbors_i.add(b)
            elif b == i: neighbors_i.add(a)
            if a == j: neighbors_j.add(b)
            elif b == j: neighbors_j.add(a)
        common = (neighbors_i & neighbors_j) - {i, j}
        
        for k in list(common)[:5]:  # Limit conditioning sets
            n_tested_first += 1
            p, _ = fisher_z_test(X, i, j, [k])
            if p > ALPHA:  # Independent given k → remove edge
                adj.discard((i, j))
                removed += 1
                break
    
    print(f"    After 1st-order CI: {len(adj)} edges (tested {n_tested_first}, removed {removed})")
    
    # Convert to edge list
    edges = []
    for (i, j) in adj:
        edges.append({'source': i, 'target': j, 'source_name': genes[i], 'target_name': genes[j]})
    
    return edges, {'marginal_tests': n_tested_marginal, 'first_order_tests': n_tested_first}

# ============================================================
# 3. PERMUTED NEGATIVE CONTROL
# ============================================================
def permuted_control(X, genes, d):
    """Permute each gene's counts independently → should find 0 edges."""
    X_perm = X.copy()
    for col in range(X.shape[1]):
        np.random.shuffle(X_perm[:, col])
    
    # Run Fisher's z PC on permuted data
    tau = FISHER_TAU.get(d, 0.15)
    edges, stats = fisher_z_pc(X_perm, genes, d, tau)
    return len(edges), stats

# ============================================================
# 4. FDR CORRECTION (Benjamini-Hochberg)
# ============================================================
# FDR correction handled in separate script (run_fdr.py)

# ============================================================
# 5. STRING VALIDATION
# ============================================================
def load_string_mapping():
    """Load ENSP ID → Gene Symbol mapping + high-confidence PPI pairs."""
    alias_path = config.get_path('string_aliases')
    ppi_path = config.get_path('string_ppi')
    
    print(f"  Loading STRING aliases from {alias_path}...")
    gene_to_ensp = {}
    with gzip.open(alias_path, 'rt', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                ensp_id = p[0].strip()
                gene_name = p[1].strip().upper()
                if gene_name not in gene_to_ensp:
                    gene_to_ensp[gene_name] = ensp_id
    
    print(f"  Loading STRING PPI from {ppi_path}...")
    ppi = set()
    with gzip.open(ppi_path, 'rt', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                try:
                    score = int(p[-1])
                    if score >= 700:
                        ppi.add((p[0], p[1]))
                except (ValueError, IndexError):
                    continue
    
    print(f"  Loaded {len(gene_to_ensp)} gene→ENSP mappings, {len(ppi)} high-conf PPI pairs")
    return gene_to_ensp, ppi

def validate_edges(edges, gene_to_ensp, ppi):
    """Count STRING-validated edges. Returns (0, 0.0, []) if STRING unavailable."""
    if gene_to_ensp is None or ppi is None:
        return 0, 0.0, []
    n_valid = 0
    validated = []
    for e in edges:
        g1 = e['source_name'].upper()
        g2 = e['target_name'].upper()
        e1 = gene_to_ensp.get(g1, '')
        e2 = gene_to_ensp.get(g2, '')
        if e1 and e2 and ((e1, e2) in ppi):
            n_valid += 1
            validated.append((g1, g2))
    precision = n_valid / len(edges) * 100 if edges else 0.0
    return n_valid, precision, validated

# ============================================================
# 6. MAIN RUNNER WITH CHECKPOINT
# ============================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Load or initialize checkpoint
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: {len(ckpt)} tasks done")
    else:
        ckpt = {}
    
    # Load STRING once (graceful fallback if not available)
    print("=== Loading STRING database ===")
    try:
        g2e, ppi = load_string_mapping()
        string_available = True
    except (FileNotFoundError, OSError) as e:
        print(f"  STRING data not found ({e}). Validation skipped.")
        print(f"  To enable: download STRING v11 to your SC_CAUSAL_DATA directory.")
        print(f"  Run 'python run_all.py --download' for instructions.")
        g2e, ppi = None, None
        string_available = False
    
    results = {}
    t_start = time.time()
    
    for d in D_VALUES:
        print(f"\n{'='*60}")
        print(f"=== d={d} ===")
        print(f"{'='*60}")
        
        # Load data
        X, genes = load_pbmc(d)
        print(f"  Data: {X.shape}, sparsity={(X==0).sum()/X.size*100:.1f}%")
        
        result_d = {'d': d, 'n_genes': len(genes), 'sparsity': float((X==0).sum()/X.size*100)}
        
        # --- Fisher's z PC ---
        key_fz = f'd_{d}_fisherz'
        if key_fz in ckpt:
            print(f"  [SKIP] Fisher's z: already done")
            result_d['fisherz'] = ckpt[key_fz]
        else:
            print(f"  Running Fisher's z PC...")
            t0 = time.time()
            tau = FISHER_TAU.get(d, 0.15)
            fz_edges, fz_stats = fisher_z_pc(X, genes, d, tau)
            fz_time = time.time() - t0
            
            # Validate
            fz_valid, fz_prec, fz_list = validate_edges(fz_edges, g2e, ppi)
            fz_result = {
                'edges': len(fz_edges),
                'string_validated': fz_valid,
                'string_precision': fz_prec,
                'time_s': fz_time,
                'marginal_tests': fz_stats['marginal_tests'],
                'first_order_tests': fz_stats['first_order_tests'],
                'top_edges': fz_list[:15]
            }
            result_d['fisherz'] = fz_result
            ckpt[key_fz] = fz_result
            # Save checkpoint after each run
            with open(CKPT_PATH, 'w') as f:
                json.dump(ckpt, f, indent=2)
            print(f"    Fisher's z: {len(fz_edges)} edges, STRING={fz_prec:.1f}% ({fz_valid} hits), {fz_time:.0f}s")
        
        # --- Permuted control (only d=30) ---
        if d == 30:
            key_perm = 'permuted_d30'
            if key_perm in ckpt:
                print(f"  [SKIP] Permuted control: already done")
                result_d['permuted'] = ckpt[key_perm]
            else:
                print(f"  Running permuted negative control...")
                t0 = time.time()
                np.random.seed(42)
                n_perm_edges, perm_stats = permuted_control(X, genes, d)
                perm_time = time.time() - t0
                perm_result = {
                    'edges': n_perm_edges,
                    'time_s': perm_time,
                    'marginal_tests': perm_stats['marginal_tests'],
                    'first_order_tests': perm_stats['first_order_tests'],
                    'conclusion': 'PASS' if n_perm_edges == 0 else f'FAIL: {n_perm_edges} edges found on permuted data'
                }
                result_d['permuted'] = perm_result
                ckpt[key_perm] = perm_result
                with open(CKPT_PATH, 'w') as f:
                    json.dump(ckpt, f, indent=2)
                print(f"    Permuted: {n_perm_edges} edges → {perm_result['conclusion']}")
        
        results[f'd_{d}'] = result_d
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"=== FINAL RESULTS ===")
    print(f"{'='*80}")
    print(f"  {'d':>5} | {'NB-LR edges':>11} | {'NB-LR STR%':>9} | {'Fisher-z edges':>13} | {'Fisher-z STR%':>11} | {'Delta STR%':>10}")
    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*9}-+-{'-'*13}-+-{'-'*11}-+-{'-'*10}")
    
    # Load NB-LR results from sweep
    sweep_ckpt = os.path.join(RESULT_DIR, '_sweep_checkpoint.json')
    nb_data = {}
    if os.path.exists(sweep_ckpt):
        with open(sweep_ckpt) as f:
            nb_data = json.load(f)
    
    for d in D_VALUES:
        d_key = f'd_{d}'
        nb = nb_data.get(d_key, {})
        fz = results.get(d_key, {}).get('fisherz', {})
        
        nb_edges = nb.get('edges', '?')
        nb_prec = nb.get('string_precision', 0)
        fz_edges = fz.get('edges', '?')
        fz_prec = fz.get('string_precision', 0)
        delta = nb_prec - fz_prec
        
        winner = 'scCausal WINS!' if delta > 0 else ('Fisher WINS' if delta < 0 else 'TIE')
        print(f"  {d:>5} | {str(nb_edges):>11} | {nb_prec:>8.1f}% | {str(fz_edges):>13} | {fz_prec:>10.1f}% | {delta:>+9.1f}%  ({winner})")
    
    # Permuted result
    if 'd_30' in results:
        perm = results['d_30'].get('permuted', {})
        print(f"\n  Permuted control (d=30): {perm.get('edges', '?')} edges → {perm.get('conclusion', '?')}")
    
    # Save full results
    full_path = os.path.join(RESULT_DIR, 'baseline_results.json')
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.0f}s")
    print(f"  Results saved to: {full_path}")

if __name__ == '__main__':
    main()
