"""
scCausal v3: NB regression + L1 sparsity. NO DAG constraint.
Each gene's expression is regressed on all other genes using NB likelihood.
The fitted coefficients form the weighted adjacency matrix.

This is fundamentally different from NOTEARS: we optimize for predictive
accuracy with NB loss, not for DAG structure. Biological gene regulation
has feedback cycles - DAG is the wrong assumption.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os, json, time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NBRegressor(nn.Module):
    """Per-gene NB regression: X_j ~ NB(mu_j, theta_j), mu_j = exp(intercept + X_{-j} @ w_j).
    
    This learns W column by column. W[i,j] = effect of gene i on gene j.
    Diagonal is zero (no self-loops). 
    No DAG constraint - feedback cycles are biologically meaningful.
    """
    def __init__(self, d, symmetric=False):
        super().__init__()
        self.d = d
        # W[i,j] = effect of i on j. Column j is the vector of parents for gene j.
        # Store as (d, d) with zero diag
        self.W = nn.Parameter(torch.randn(d, d) * 0.1)
        self.log_theta = nn.Parameter(torch.zeros(d))  # per-gene dispersion
    
    def forward(self, X):
        """NB negative log-likelihood for all genes.
        
        X_j ~ NB(mu_j, theta_j) where log(mu_j) = intercept_j + sum_i X_i * W_ij
        """
        eps = 1e-8
        n, d = X.shape
        
        # Zero diag
        W = self.W - torch.diag(torch.diag(self.W))
        
        # Library-size normalized mean
        lib = X.sum(1, keepdim=True).float()
        linear = torch.matmul(X.float(), W)  # [n, d]
        log_lib = lib.log()
        
        # log(mu) = log(lib) + linear (no softplus cap, simpler)
        log_mu = log_lib + linear - 5.0  # intercept-like shrinkage
        mu = torch.clamp(torch.exp(log_mu), min=eps)
        
        theta = torch.clamp(torch.exp(self.log_theta), min=1.0, max=1000.0)  # [d]
        
        # NB log-prob
        x_f = X.float()
        t1 = torch.lgamma(x_f + theta + eps) - torch.lgamma(theta + eps)
        t2 = -torch.lgamma(x_f + 1.0)
        t3 = theta * torch.log(theta + eps)
        t4 = x_f * torch.log(mu + eps)
        t5 = -(theta + x_f) * torch.log(theta + mu + eps)
        
        ll = t1 + t2 + t3 + t4 + t5
        return -ll.mean(), W, theta


def train_sccausal_v3(X, d, lr=0.002, lambda1=0.001, n_iter=300, 
                       device='cuda', verbose=True):
    """Train NB regressor with L1 sparsity. No DAG constraint."""
    X = X.to(device)
    model = NBRegressor(d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    pbar = tqdm(range(n_iter), desc='scCausal v3') if verbose else range(n_iter)
    
    for it in pbar:
        optimizer.zero_grad()
        nll, W, theta = model(X)
        l1 = lambda1 * W.abs().sum()
        loss = nll + l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        if it % 20 == 0:
            with torch.no_grad():
                W_c = model.W - torch.diag(torch.diag(model.W))
                n_edges = (W_c.abs() > 0.3).sum().item()
                n_nz = (W_c.abs() > 1e-4).sum().item()
                if verbose:
                    pbar.set_postfix({
                        'nll': f'{nll.item():.1f}',
                        'edges': n_edges,
                        'nonzero': n_nz
                    })
    
    W_final = model.W - torch.diag(torch.diag(model.W))
    return W_final.detach()


def postprocess_acyclic(W):
    """Optional: remove cycles by keeping only top edges that form a DAG.
    Uses greedy edge removal (MES - minimal edge set for acyclicity).
    """
    W_np = W.cpu().numpy()
    d = W_np.shape[0]
    
    # Sort edges by absolute weight
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W_np[i, j]) > 0.3:
                edges.append((i, j, abs(W_np[i, j])))
    edges.sort(key=lambda x: -x[2])  # descending weight
    
    # Greedy DAG: add edges in order, skip if creates cycle
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    kept = []
    
    for (i, j, w) in edges:
        G.add_edge(i, j)
        try:
            nx.find_cycle(G)
            G.remove_edge(i, j)  # creates cycle, remove
        except nx.NetworkXNoCycle:
            kept.append((i, j, float(W_np[i, j])))
    
    return kept


def extract_edges(W, threshold=0.3):
    W_np = W.cpu().numpy()
    edges = []
    for i in range(W_np.shape[0]):
        for j in range(W_np.shape[1]):
            if i != j and abs(W_np[i, j]) > threshold:
                edges.append({'source': int(i), 'target': int(j), 'weight': float(W_np[i, j])})
    return edges


print('scCausal v3: NB regression + L1, NO DAG constraint')
