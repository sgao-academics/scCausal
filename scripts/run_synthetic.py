"""
Synthetic ZINB benchmark: generate data with known causal DAG,
compare scCausal vs NOTEARS. Ground truth F1 evaluation.
"""
import sys, os, json, time, gc
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sccausal_engine import scCausalModel, train_sccausal, extract_edges, dag_constraint
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(42)

def generate_zinb_dag(d, n_samples, n_edges, zero_frac=0.85):
    """Generate ZINB data from a known sparse DAG."""
    import networkx as nx
    # Create random DAG
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    edges_added = 0
    while edges_added < n_edges:
        i, j = np.random.randint(0, d, 2)
        if i < j and not G.has_edge(i, j):
            # Check no cycles
            G_temp = G.copy()
            G_temp.add_edge(i, j)
            if nx.is_directed_acyclic_graph(G_temp):
                G.add_edge(i, j)
                edges_added += 1
    
    W_true = np.zeros((d, d))
    for (i, j) in G.edges():
        W_true[i, j] = np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1])
    
    # Generate data: X = XW + noise, then ZINB
    X = np.random.exponential(2, (n_samples, d))
    for _ in range(3):
        X_new = X @ W_true.T
        X = X + X_new * 0.5
    X = np.clip(X, 0, None)
    X = np.round(X).astype(np.float32)
    
    # Zero-inflate
    mask = np.random.random((n_samples, d)) > zero_frac
    X = X * mask.astype(np.float32)
    
    print(f'Generated: {n_samples}x{d}, {n_edges} edges, {(X==0).sum()/X.size*100:.1f}% zeros')
    return X, W_true

def compute_f1(W_est, W_true):
    """Compute F1 score comparing estimated to true adjacency."""
    est_edges = set()
    true_edges = set()
    d = W_true.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j and abs(W_true[i, j]) > 1e-6:
                true_edges.add((i, j))
            if i != j and abs(W_est[i, j]) > 0.3:
                est_edges.add((i, j))
    
    tp = len(est_edges & true_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1, prec, rec, tp, fp, fn

results = []
for d, n_edges, n_samples in [
    (50, 40, 500),
    (100, 80, 800),
    (200, 160, 1500),
]:
    print(f'\n=== d={d}, edges={n_edges} ===')
    X, W_true = generate_zinb_dag(d, n_samples, n_edges)
    X_t = torch.tensor(X, dtype=torch.float32)
    
    # scCausal
    rank = min(32, d//3)
    model = scCausalModel(d, rank=rank, dropout=True)
    t0 = time.time()
    W_est = train_sccausal(model, X_t, n_outer=80, n_inner=300, lambda1=0.0005,
                           device=DEVICE, verbose=False, warmup=30)
    t_sc = time.time() - t0
    f1_sc, prec_sc, rec_sc, tp_sc, fp_sc, fn_sc = compute_f1(W_est.numpy(), W_true)
    n_edges_sc = int((abs(W_est) > 0.3).sum().item())
    
    # NOTEARS baseline (using Gaussian model with full rank)
    import torch.nn as nn
    class NoteARS(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.W_flat = nn.Parameter(torch.randn(d*d) * 0.01)
        def get_W(self):
            W = self.W_flat.view(d, d)
            return W - torch.diag(torch.diag(W))
    
    nt = NoteARS(d).to(DEVICE)
    X_cuda = X_t.to(DEVICE)
    rho = 1.0; alpha = torch.tensor(0.0, device=DEVICE)
    t0 = time.time()
    for outer in range(50):
        opt = torch.optim.Adam(nt.parameters(), lr=0.002)
        for _ in range(300):
            opt.zero_grad()
            W = nt.get_W()
            recon = X_cuda - X_cuda @ W
            mse = (recon**2).mean()
            l1 = 0.001 * W.abs().sum()
            h = dag_constraint(W)
            loss = mse + l1 + alpha*h + 0.5*rho*h**2
            loss.backward()
            opt.step()
        with torch.no_grad():
            h_new = dag_constraint(nt.get_W()).item()
        alpha = alpha + rho * h_new
        if h_new > 0.25 * h.item():
            rho = min(rho*10, 1e10)
        if h_new < 1e-8: break
    t_nt = time.time() - t0
    
    W_nt = nt.get_W().detach().cpu().numpy()
    f1_nt, prec_nt, rec_nt, tp_nt, fp_nt, fn_nt = compute_f1(W_nt, W_true)
    n_edges_nt = int((abs(W_nt) > 0.3).sum())
    
    res = {
        'd': d, 'n_edges_true': n_edges, 'n_samples': n_samples,
        'scCausal': {'F1': f1_sc, 'prec': prec_sc, 'rec': rec_sc, 
                     'edges': n_edges_sc, 'time': t_sc, 'tp': tp_sc, 'fp': fp_sc, 'fn': fn_sc},
        'NOTEARS': {'F1': f1_nt, 'prec': prec_nt, 'rec': rec_nt,
                    'edges': n_edges_nt, 'time': t_nt, 'tp': tp_nt, 'fp': fp_nt, 'fn': fn_nt}
    }
    results.append(res)
    print(f'  scCausal: F1={f1_sc:.3f} (P={prec_sc:.3f}, R={rec_sc:.3f}), {n_edges_sc} edges, {t_sc:.0f}s')
    print(f'  NOTEARS:  F1={f1_nt:.3f} (P={prec_nt:.3f}, R={rec_nt:.3f}), {n_edges_nt} edges, {t_nt:.0f}s')
    
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

with open(os.path.join(RESULTS_DIR, 'synthetic_benchmark.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {RESULTS_DIR}/synthetic_benchmark.json')
