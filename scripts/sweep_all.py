import config  # scCausal data path configuration
"""
Multi-d robustness sweep + Synthetic benchmark + Perturb-seq prep.
Runs: d=[100,200,300,500] x (scCausal, NOTEARS) + synthetic F1.
"""
import sys, os, json, time, gc, gzip
import numpy as np
import torch
import scanpy as sc

sys.path.insert(0, os.path.dirname(__file__))
from sccausal_engine import scCausalModel, train_sccausal, extract_edges, dag_constraint, estimate_rank

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load data once
# ============================================================
print('Loading PBMC...')
adata = sc.read_h5ad(os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad'))
if hasattr(adata, 'raw') and adata.raw is not None:
    X_raw = np.array(adata.raw.X.todense() if hasattr(adata.raw.X, 'todense')
                      else adata.raw.X, dtype=np.float32)
else:
    X_raw = np.array(adata.X.todense(), dtype=np.float32)
all_genes = list(adata.var_names.astype(str))

# Select by variance ranking
variances = np.var(X_raw, axis=0)
ranked_idx = np.argsort(variances)[::-1]  # descending

# ============================================================
# STRING/TRRUST loader (once)
# ============================================================
alias_path = r'config.get_path("string_aliases")'
ppi_path = r'config.get_path("string_ppi")'

symbol2string = {}
with gzip.open(alias_path, 'rt', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.strip().split('\t')
        if len(p) >= 2 and not p[1].isdigit():
            symbol2string[p[1].upper()] = p[0]

ppi_set = set()
with gzip.open(ppi_path, 'rt', encoding='utf-8', errors='ignore') as f:
    f.readline()  # header
    for i, line in enumerate(f):
        p = line.strip().split()
        try:
            if int(p[-1]) >= 700:
                ppi_set.add((p[0], p[1]))
        except: pass
        if i % 5000000 == 0 and i > 0:
            print(f'  STRING: {i//1000000}M lines, {len(ppi_set)//1000}K pairs')
print(f'STRING: {len(ppi_set)//1000}K HC pairs loaded')

trrust_pairs = set()
with open(r'config.get_path("trrust")') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            trrust_pairs.add((p[0].upper(), p[1].upper()))
print(f'TRRUST: {len(trrust_pairs)} pairs')

# ============================================================
# Gaussian solver (reusable)
# ============================================================
class GaussianModel(torch.nn.Module):
    def __init__(self, d, rank=None):
        super().__init__()
        self.d = d
        if rank and rank < d:
            self.U = torch.nn.Parameter(torch.randn(d, rank) * 0.1)
            self.V = torch.nn.Parameter(torch.randn(d, rank) * 0.1)
            self.use_lr = True
        else:
            self.W_flat = torch.nn.Parameter(torch.randn(d*d) * 0.01)
            self.use_lr = False
    def get_W(self):
        W = self.U @ self.V.T if self.use_lr else self.W_flat.view(self.d, self.d)
        return W - torch.diag(torch.diag(W))

def train_gaussian(model, X, device, n_outer=50):
    X_c = X.to(device)
    model = model.to(device)
    rho, alpha = 1.0, torch.tensor(0.0, device=device)
    for outer in range(n_outer):
        opt = torch.optim.Adam(model.parameters(), lr=0.002)
        for _ in range(300):
            opt.zero_grad()
            W = model.get_W()
            recon = X_c - X_c @ W
            mse = (recon**2).mean()
            h = dag_constraint(W)
            loss = mse + 0.001*W.abs().sum() + alpha*h + 0.5*rho*h**2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        with torch.no_grad():
            h_new = dag_constraint(model.get_W()).item()
        alpha = alpha + rho * h_new
        if h_new > 0.25 * h.item(): rho = min(rho*10, 1e10)
        if outer % 20 == 0: print(f'    G outer={outer}: h={h_new:.2e}')
        if h_new < 1e-8: break
    return model.get_W().detach()

# ============================================================
# SWEEP: d = 100, 200, 300, 500
# ============================================================
D_VALUES = [100, 200, 300, 500]
sweep_results = {}

for d in D_VALUES:
    print(f'\n{"="*60}')
    print(f'D={d}')
    print(f'{"="*60}')
    
    # Subset
    idx = ranked_idx[:d]
    X_d = X_raw[:, idx]
    genes_d = [all_genes[i] for i in idx]
    sparsity = (X_d == 0).sum() / X_d.size * 100
    print(f'  Shape: {X_d.shape}, sparsity: {sparsity:.1f}%')
    
    X_t = torch.tensor(X_d, dtype=torch.float32)
    r = estimate_rank(X_t, 0.80)
    rank = min(r, max(16, d//4))
    print(f'  Rank: {rank}')
    
    res = {'d': d, 'rank': rank, 'sparsity': sparsity, 'n_genes': len(genes_d)}
    
    # Gene->STRING mapping
    gene2s = {}
    for g in genes_d:
        gene2s[g] = symbol2string.get(g.upper(), '')
    mapped = sum(1 for v in gene2s.values() if v)
    print(f'  STRING mapped: {mapped}/{d}')
    
    # === scCausal ===
    t0 = time.time()
    model = scCausalModel(d, rank=rank, dropout=True)
    W_sc = train_sccausal(model, X_t, n_outer=80, n_inner=300, lambda1=0.0005,
                          device=DEVICE, verbose=True, warmup=30)
    t_sc = time.time() - t0
    edges_sc = extract_edges(W_sc, threshold=0.3)
    n_sc = len(edges_sc)
    
    # Validate
    s_sc = sum(1 for e in edges_sc if gene2s.get(genes_d[e['source']],'') and 
               gene2s.get(genes_d[e['target']],'') and 
               (gene2s[genes_d[e['source']]], gene2s[genes_d[e['target']]]) in ppi_set)
    t_sc_v = sum(1 for e in edges_sc if (genes_d[e['source']].upper(), genes_d[e['target']].upper()) in trrust_pairs)
    
    res['scCausal'] = {'edges': n_sc, 'time': t_sc, 'STRING': s_sc, 'STRING%': s_sc/max(n_sc,1)*100,
                        'TRRUST': t_sc_v, 'TRRUST%': t_sc_v/max(n_sc,1)*100}
    
    print(f'  scCausal: {n_sc} edges, STRING={s_sc}({s_sc/max(n_sc,1)*100:.1f}%), TRRUST={t_sc_v}, {t_sc:.0f}s')
    
    # === NOTEARS (Gaussian + full rank, same as ablation) ===
    t0 = time.time()
    gm = GaussianModel(d, rank=None)
    W_nt = train_gaussian(gm, X_t, DEVICE, n_outer=50)
    t_nt = time.time() - t0
    edges_nt = extract_edges(W_nt, threshold=0.3)
    n_nt = len(edges_nt)
    
    s_nt = sum(1 for e in edges_nt if gene2s.get(genes_d[e['source']],'') and 
               gene2s.get(genes_d[e['target']],'') and 
               (gene2s[genes_d[e['source']]], gene2s[genes_d[e['target']]]) in ppi_set)
    t_nt_v = sum(1 for e in edges_nt if (genes_d[e['source']].upper(), genes_d[e['target']].upper()) in trrust_pairs)
    
    res['NOTEARS'] = {'edges': n_nt, 'time': t_nt, 'STRING': s_nt, 'STRING%': s_nt/max(n_nt,1)*100,
                       'TRRUST': t_nt_v, 'TRRUST%': t_nt_v/max(n_nt,1)*100}
    
    print(f'  NOTEARS:  {n_nt} edges, STRING={s_nt}({s_nt/max(n_nt,1)*100:.1f}%), TRRUST={t_nt_v}, {t_nt:.0f}s')
    
    # Compute delta
    delta_sc = s_sc - s_nt
    res['delta_STRING'] = delta_sc
    print(f'  Delta: scCausal finds {delta_sc:+d} more STRING edges than NOTEARS')
    
    sweep_results[f'd_{d}'] = res
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

with open(os.path.join(RESULTS_DIR, 'multi_d_sweep.json'), 'w') as f:
    json.dump(sweep_results, f, indent=2)

# ============================================================
# SUMMARY TABLE
# ============================================================
print(f'\n{"="*70}')
print(f'MULTI-D SWEEP SUMMARY')
print(f'{"="*70}')
print(f'{"D":>4} {"Method":>12} {"Edges":>6} {"STRING":>8} {"TRRUST":>8} {"Time":>6}')
print(f'{"-"*46}')
for d in D_VALUES:
    r = sweep_results[f'd_{d}']
    sc = r['scCausal']
    nt = r['NOTEARS']
    print(f'{d:>4} {"scCausal":>12} {sc["edges"]:>6} {sc["STRING%"]:>7.1f}% {sc["TRRUST%"]:>7.1f}% {sc["time"]:>5.0f}s')
    print(f'{"":>4} {"NOTEARS":>12} {nt["edges"]:>6} {nt["STRING%"]:>7.1f}% {nt["TRRUST%"]:>7.1f}% {nt["time"]:>5.0f}s')
    print(f'{"":>4} {"DELTA":>12} {"":>6} {r["delta_STRING"]:+d} STRING edges')
    print()

print('Saved: multi_d_sweep.json')
