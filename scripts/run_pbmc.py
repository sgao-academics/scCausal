import config  # scCausal data path configuration
"""
Full PBMC experiment: scCausal vs baselines + ablation + STRING validation.
Independent of causalscale. GPU-accelerated. Run: python run_pbmc.py
"""
import sys, os, json, time, gzip, gc
import numpy as np
import torch
import scanpy as sc

sys.path.insert(0, os.path.dirname(__file__))
from sccausal_engine import (scCausalModel, train_sccausal, extract_edges,
                              dag_constraint, estimate_rank, ZINBLoss)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)
print(f'Device: {DEVICE}')

# ============================================================
# 1. LOAD & PREPROCESS
# ============================================================
print('\n=== STEP 1: Load PBMC 3K ===')
adata = sc.read_h5ad(os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad'))
print(f'Shape: {adata.shape}')
print(f'Cell types: {dict(adata.obs["cell_type"].value_counts())}')

# Raw count matrix: try .raw.X first, fall back to .X
if hasattr(adata, 'raw') and adata.raw is not None:
    X_raw = np.array(adata.raw.X.todense() if hasattr(adata.raw.X, 'todense')
                      else adata.raw.X, dtype=np.float32)
else:
    X_raw = np.array(adata.X.todense() if hasattr(adata.X, 'todense')
                      else adata.X, dtype=np.float32)
gene_names = list(adata.var_names.astype(str))

# Use top N highly variable genes for computational efficiency
# (d=500 is standard in causal discovery literature; NOTEARS original max d=100)
N_GENES = 500
if X_raw.shape[1] > N_GENES:
    variances = np.var(X_raw, axis=0)
    top_idx = np.argsort(variances)[-N_GENES:]
    X_raw = X_raw[:, top_idx]
    gene_names = [gene_names[i] for i in top_idx]
    print(f'Subsampled to {N_GENES} highly variable genes')
n, d = X_raw.shape
print(f'Raw counts: {n} cells x {d} genes, {100*(X_raw==0).sum()/X_raw.size:.1f}% zeros')
print(f'Count range: [{X_raw.min():.0f}, {X_raw.max():.0f}]')

X_torch = torch.tensor(X_raw, dtype=torch.float32)
r = estimate_rank(X_torch, 0.80)
print(f'Estimated rank: {r} (clamped to [16,128])')

np.savez_compressed(os.path.join(DATA_DIR, 'pbmc3k_preprocessed.npz'),
                    X=X_raw, gene_names=np.array(gene_names, dtype=str))

# ============================================================
# 2. RUN METHODS
# ============================================================
configs = [
    ('scCausal_ZINB_LR', scCausalModel(d, rank=r, dropout=True), True),
    ('Abl_Gaussian_LR', scCausalModel(d, rank=r, dropout=True), False),
    ('Abl_NB_LR', scCausalModel(d, rank=r, dropout=False), True),
    ('Abl_ZINB_Full', scCausalModel(d, rank=None, dropout=True), True),
    ('Baseline_NOTEARS', scCausalModel(d, rank=None, dropout=True), False),
]

all_results = {}
for name, model, use_zinb in configs:
    print(f'\n=== {name} ===')
    t0 = time.time()
    
    if not use_zinb:
        # Use Gaussian (MSE) loss instead of ZINB
        import torch.nn as nn
        class GaussianModel(nn.Module):
            def __init__(self, d, rank):
                super().__init__()
                self.d = d
                if rank and rank < d:
                    self.U = nn.Parameter(torch.randn(d, rank) * 0.1)
                    self.V = nn.Parameter(torch.randn(d, rank) * 0.1)
                    self.use_lr = True
                else:
                    self.W_flat = nn.Parameter(torch.randn(d*d) * 0.01)
                    self.use_lr = False
            def get_W(self):
                W = self.U @ self.V.T if self.use_lr else self.W_flat.view(self.d, self.d)
                return W - torch.diag(torch.diag(W))
        
        gauss_model = GaussianModel(d, r).to(DEVICE)
        X_cuda = X_torch.to(DEVICE)
        rho = 1.0; alpha = torch.tensor(0.0, device=DEVICE)
        h_prev = float('inf')
        
        for outer in range(200):
            opt = torch.optim.Adam(gauss_model.parameters(), lr=0.002)
            for inner in range(500):
                opt.zero_grad()
                W = gauss_model.get_W()
                recon = X_cuda - X_cuda @ W
                mse = (recon ** 2).mean()
                l1 = 0.001 * W.abs().sum()
                h_val = dag_constraint(W)
                loss = mse + l1 + alpha * h_val + 0.5 * rho * h_val ** 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gauss_model.parameters(), 10.0)
                opt.step()
            
            with torch.no_grad():
                h_new = dag_constraint(gauss_model.get_W()).item()
            alpha = alpha + rho * h_new
            if h_new > 0.25 * h_val.item():
                rho = min(rho * 10, 1e10)
            
            if outer % 30 == 0:
                print(f'  outer={outer}: h={h_new:.2e}, edges={(gauss_model.get_W().abs()>0.3).sum().item()}')
            
            if h_new < 1e-8: break
            h_prev = h_new
        
        W_final = gauss_model.get_W().detach()
        t_elapsed = time.time() - t0
    else:
        # Use ZINB loss
        W_final = train_sccausal(model, X_torch, n_outer=200, device=DEVICE, verbose=True)
        t_elapsed = time.time() - t0
    
    edges = extract_edges(W_final, threshold=0.3)
    print(f'  Result: {len(edges)} edges in {t_elapsed:.0f}s')
    
    all_results[name] = {
        'n_edges': len(edges),
        'time_s': t_elapsed,
        'W': W_final.cpu().numpy().tolist(),
        'edges': edges[:50]
    }
    
    torch.save({
        'W': W_final.cpu(), 'edges': edges, 'n_edges': len(edges), 'time': t_elapsed
    }, os.path.join(RESULTS_DIR, f'{name}.pt'))
    
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

# ============================================================
# 3. STRING + TRRUST VALIDATION
# ============================================================
print('\n=== STEP 3: External Validation ===')

def load_string(alias_path, ppi_path, gene_names):
    """Build STRING reference and validate."""
    symbol2string = {}
    with gzip.open(alias_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.strip().split('\t')
            if len(p) >= 2 and not p[1].isdigit():
                symbol2string[p[1].upper()] = p[0]
    
    gene2string = {g: symbol2string.get(g.upper(), '') for g in gene_names}
    mapped = sum(1 for v in gene2string.values() if v)
    print(f'  STRING mapping: {mapped}/{len(gene_names)} genes')
    
    ppi_set = set()
    with gzip.open(ppi_path, 'rt', encoding='utf-8', errors='ignore') as f:
        header = f.readline()  # Skip header
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                try:
                    combined_score = int(p[-1])
                    if combined_score >= 700:
                        ppi_set.add((p[0], p[1]))
                except ValueError:
                    continue
    
    print(f'  STRING HC edges: {len(ppi_set)}')
    return gene2string, ppi_set

string_alias = config.resolve('string_aliases', 'SC_CAUSAL_STRING_ALIASES')
string_ppi = config.resolve('string_ppi', 'SC_CAUSAL_STRING_PPI')

if os.path.exists(string_ppi):
    gene2s, ppi = load_string(string_alias, string_ppi, gene_names)
else:
    gene2s, ppi = {}, set()

# TRRUST
trrust_pairs = set()
trrust_path = config.resolve('trrust', 'SC_CAUSAL_TRRUST')
if os.path.exists(trrust_path):
    with open(trrust_path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                trrust_pairs.add((p[0].upper(), p[1].upper()))
    print(f'  TRRUST pairs: {len(trrust_pairs)}')

# Validate all methods
print(f'\n{"Method":<25} {"Edges":>6} {"STRING%":>10} {"TRRUST%":>10}')
print('-'*55)
for name in all_results:
    edges = all_results[name]['edges']
    if not edges: continue
    
    s_val = 0; t_val = 0
    for e in edges:
        sid = gene2s.get(gene_names[e['source']], '')
        tid = gene2s.get(gene_names[e['target']], '')
        # STRING: check full ENSP ID pair
        if sid and tid and ((sid, tid) in ppi):
            s_val += 1
        # TRRUST: gene symbol match
        if (gene_names[e['source']].upper(), gene_names[e['target']].upper()) in trrust_pairs:
            t_val += 1
    
    sp = s_val/len(edges)*100 if edges else 0
    tp = t_val/len(edges)*100 if edges else 0
    all_results[name]['string_precision'] = sp
    all_results[name]['string_validated'] = s_val
    all_results[name]['trrust_precision'] = tp
    all_results[name]['trrust_validated'] = t_val
    print(f'{name:<25} {len(edges):>6} {sp:>9.1f}% {tp:>9.1f}%')

# ============================================================
# 4. CELL-TYPE IMMUNE MARKER ANALYSIS
# ============================================================
print('\n=== STEP 4: Immune Marker Recovery ===')
immune_pairs = [
    ('CD3D','CD3E'), ('CD8A','CD8B'), ('IL7R','CD4'), ('MS4A1','CD19'),
    ('CD14','FCGR3A'), ('NKG7','GNLY'), ('CD4','LCK'), ('CD8A','GZMK'),
    ('CCR7','SELL'), ('CD79A','CD79B'), ('CST3','LYZ')
]
sc_edges = all_results['scCausal_ZINB_LR']['edges']
found = 0
for (s,t) in immune_pairs:
    for e in sc_edges:
        gs = gene_names[e['source']].upper()
        gt = gene_names[e['target']].upper()
        if (gs==s and gt==t) or (gs==t and gt==s):
            print(f'  [FOUND] {s}->{t}: weight={e["weight"]:.3f}')
            found += 1
            break
print(f'Immune markers recovered: {found}/{len(immune_pairs)}')

# ============================================================
# 5. SAVE
# ============================================================
all_results['meta'] = {
    'n_cells': n, 'n_genes': d, 'rank': r,
    'sparsity_pct': float(100*(X_raw==0).sum()/X_raw.size),
    'device': DEVICE, 'gene_names': gene_names[:50]
}
with open(os.path.join(RESULTS_DIR, 'pbmc_full_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'\n=== DONE: {RESULTS_DIR}/pbmc_full_results.json ===')
