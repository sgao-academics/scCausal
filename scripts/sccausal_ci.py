"""
scCausal: PC algorithm + NB likelihood ratio CI test for scRNA-seq.

Innovation: Replace Fisher's z-test (Gaussian assumption) with NB likelihood 
ratio test, which is the correct distribution for scRNA-seq count data.

Algorithm:
  1. Start with fully connected undirected graph
  2. For each edge (i,j): test X_i ⟂ X_j | S for conditioning sets S of size 0,1,2,...
  3. CI test: fit NB(X_j ~ S) vs NB(X_j ~ S + X_i), compute LR = 2*(LL1 - LL0)
  4. If LR < chi2_critical (df=1, alpha=0.05), remove edge
  5. Orient edges using standard PC rules

Computational optimizations:
  - Only test first-order CI (|S| <= 1) for efficiency
  - Cache NB model fits
  - Parallelize gene pairs
"""
import numpy as np
import torch
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, json, time, gzip

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def fit_nb_ll(X, y):
    """Fit Negative Binomial GLM: y ~ X, return log-likelihood.
    
    Uses gradient descent with Adam for speed.
    X: (n, p) predictor matrix
    y: (n,) response vector (counts)
    """
    n, p = X.shape
    eps = 1e-8
    
    # Convert to torch
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    
    # Parameters
    intercept = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    coef = torch.zeros(p, device=DEVICE, requires_grad=True)
    log_theta = torch.tensor(0.0, device=DEVICE, requires_grad=True)  # log dispersion
    
    optimizer = torch.optim.Adam([intercept, coef, log_theta], lr=0.01)
    
    best_ll = -float('inf')
    best_params = None
    
    for _ in range(200):
        optimizer.zero_grad()
        theta = torch.clamp(torch.exp(log_theta), min=1.0, max=1000.0)
        log_mu = intercept + X_t @ coef
        mu = torch.clamp(torch.exp(log_mu), min=eps)
        
        # NB log-likelihood
        t1 = torch.lgamma(y_t + theta + eps) - torch.lgamma(theta + eps)
        t2 = -torch.lgamma(y_t + 1.0)
        t3 = theta * torch.log(theta + eps)
        t4 = y_t * torch.log(mu + eps)
        t5 = -(theta + y_t) * torch.log(theta + mu + eps)
        ll = (t1 + t2 + t3 + t4 + t5).sum()
        
        (-ll).backward()
        optimizer.step()
        
        ll_val = ll.item()
        if ll_val > best_ll:
            best_ll = ll_val
            best_params = (intercept.item(), coef.detach().cpu().numpy(), theta.item())
    
    return best_ll


def nb_ci_test(X_data, i, j, condition_set, alpha=0.05):
    """NB likelihood ratio CI test.
    
    H0: gene_i ⟂ gene_j | condition_set
    Returns: (independent: bool, p_value: float, LR: float)
    """
    n = X_data.shape[0]
    
    # Build predictor matrices
    predictors_base = [X_data[:, k] for k in condition_set]
    
    if len(predictors_base) == 0:
        # No conditioning: just test correlation
        X_null = np.ones((n, 1))  # intercept only
        X_alt = np.column_stack([np.ones(n), X_data[:, i]])
    else:
        X_null = np.column_stack([np.ones(n)] + [X_data[:, k] for k in condition_set])
        X_alt = np.column_stack([np.ones(n)] + [X_data[:, k] for k in condition_set] + [X_data[:, i]])
    
    y = X_data[:, j]
    
    ll0 = fit_nb_ll(X_null, y)
    ll1 = fit_nb_ll(X_alt, y)
    
    LR = 2 * (ll1 - ll0)
    p_value = 1 - chi2.cdf(max(LR, 0), df=1)
    
    return p_value > alpha, p_value, max(LR, 0)


def pc_skeleton(X_data, gene_names, max_cond_size=1, alpha=0.05, verbose=True):
    """PC algorithm skeleton discovery with NB CI test.
    
    Args:
        X_data: (n_cells, d_genes) count matrix
        gene_names: list of gene names
        max_cond_size: max conditioning set size (1 = first-order PC)
        alpha: significance level
    
    Returns:
        adjacency: (d, d) binary matrix (1 = edge exists)
        edge_history: list of (i,j,cond_set_size,p_value) for removed edges
    """
    n, d = X_data.shape
    
    # Start with fully connected (upper triangle)
    adj = np.ones((d, d), dtype=bool)
    np.fill_diagonal(adj, False)  # no self-loops
    # Keep only upper triangle (undirected)
    adj = np.triu(adj, k=1)
    adj = adj + adj.T
    
    n_initial = int(adj.sum() / 2)
    if verbose:
        print(f'Initial edges: {n_initial}')
    
    edge_history = []
    
    # Conditioning size 0 (marginal independence)
    if verbose: print(f'Testing marginal independence (|S|=0)...')
    tested = 0
    for i in range(d):
        for j in range(i+1, d):
            if not adj[i, j]: continue
            indep, p_val, lr = nb_ci_test(X_data, i, j, set(), alpha)
            tested += 1
            if indep:
                adj[i, j] = adj[j, i] = False
                edge_history.append((i, j, 0, float(p_val), float(lr)))
            if tested % 20 == 0 and verbose:
                n_edges = int(adj.sum() / 2)
                print(f'  Tested {tested}/{n_initial}, remaining: {n_edges} edges')
    
    n_after_marginal = int(adj.sum() / 2)
    if verbose: print(f'After marginal CI: {n_after_marginal} edges')
    
    # Conditioning size 1 (first-order CI)
    if max_cond_size >= 1 and n_after_marginal > 0:
        if verbose: print(f'Testing first-order CI (|S|=1)...')
        tested = 0
        for i in range(d):
            neighbors = np.where(adj[i, :])[0]
            for j in neighbors:
                if j <= i: continue
                if not adj[i, j]: continue
                
                # Try conditioning on each common neighbor
                common = [k for k in neighbors if k != j and adj[k, i] or adj[k, j]]
                for k in common:
                    if not adj[i, j]: break
                    indep, p_val, lr = nb_ci_test(X_data, i, j, {k}, alpha)
                    tested += 1
                    if indep:
                        adj[i, j] = adj[j, i] = False
                        edge_history.append((i, j, 1, float(p_val), float(lr)))
                
                if tested % 50 == 0 and verbose:
                    n_edges = int(adj.sum() / 2)
                    print(f'  Tested {tested} CI tests, remaining: {n_edges} edges')
    
    n_final = int(adj.sum() / 2)
    if verbose: print(f'Final edges: {n_final}')
    
    return adj, edge_history


def extract_edges_from_adj(adj, gene_names, W_weights=None):
    """Convert adjacency matrix + optional weights to edge list."""
    d = adj.shape[0]
    edges = []
    for i in range(d):
        for j in range(i+1, d):
            if adj[i, j]:
                w = W_weights[i, j] if W_weights is not None else 1.0
                edges.append({'source': int(i), 'target': int(j), 'weight': float(w),
                              'source_name': gene_names[i], 'target_name': gene_names[j]})
    return edges


print('scCausal CI loaded: PC algorithm + NB likelihood ratio test')
