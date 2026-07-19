"""
scCausal v2: NB likelihood + Spectral Norm acyclicity + Low-rank.
Inspired by CASCADE's approach, completely rewritten from scratch.
No dependency on causalscale. No ZINB. No matrix_exp.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os, json, time, gc

# === NB Likelihood (no zero-inflation!) ===
class NBLikelihood(nn.Module):
    """Negative Binomial likelihood with library size normalization.
    
    X_ij ~ NB(mu_ij, theta_j) where:
      mu_ij = library_size_i * expression_rate_ij
      log(mu_ij) = log(library_size_i) + (XW)_ij + log(cap_j) - softplus(log_mu_j)
    
    theta_j: gene-specific dispersion, estimated empirically.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.log_theta = nn.Parameter(torch.zeros(d))  # log dispersion
    
    def forward(self, X, W, library_size=None):
        """Negative log-likelihood per entry.
        
        Args:
            X: [n, d] raw count matrix
            W: [d, d] weighted adjacency (zero diag)
            library_size: [n] total counts per cell, or None to compute
        """
        eps = 1e-8
        n, d = X.shape
        
        if library_size is None:
            library_size = X.sum(1, keepdim=True)  # [n, 1]
        elif library_size.ndim == 1:
            library_size = library_size.unsqueeze(1)
        
        # Library-size normalized mean
        # mu = library_size * exp(linear_predictor) * cap
        linear = torch.matmul(X.float(), W)  # [n, d], XW
        log_lib = library_size.float().log()
        
        # Softplus parameterization for mean (as in CASCADE)
        log_mu = log_lib + linear - F.softplus(linear + log_lib) + 0.1  # cap offset
        
        mu = torch.clamp(torch.exp(log_mu), min=eps)
        
        # Dispersion (from log_theta, ensure > 0)
        theta = torch.clamp(torch.exp(self.log_theta), min=1.0, max=1000.0)  # [d]
        
        # NB log-prob: NB(x | mu, theta) where Var = mu + mu^2/theta
        x_f = X.float()
        
        # log P(x | mu, theta) = lgamma(x + theta) - lgamma(theta) - lgamma(x+1)
        #   + theta * log(theta) + x * log(mu) - (theta + x) * log(theta + mu)
        t1 = torch.lgamma(x_f + theta + eps) - torch.lgamma(theta + eps)
        t2 = -torch.lgamma(x_f + 1.0)
        t3 = theta * torch.log(theta + eps)
        t4 = x_f * torch.log(mu + eps)
        t5 = -(theta + x_f) * torch.log(theta + mu + eps)
        
        ll = t1 + t2 + t3 + t4 + t5
        nll = -ll.mean()
        
        return nll, mu, theta


# === Spectral Norm Acyclicity ===
def spectral_norm_energy(W, n_iter=5):
    """Compute spectral norm energy for acyclicity constraint.
    
    Uses power iteration. W is acyclic iff spectral radius < limit.
    Returns energy >= 0; energy=0 means acyclic.
    
    Args:
        W: [d, d] weighted adjacency
        n_iter: power iteration steps
    
    Returns:
        energy: scalar, 0 if spectral radius < limit
    """
    d = W.shape[0]
    if d <= 1:
        return torch.tensor(0.0, device=W.device)
    
    # Power iteration for spectral radius
    with torch.no_grad():
        v = torch.randn(d, 1, device=W.device)
        v = v / v.norm()
        for _ in range(n_iter):
            v = W.T @ (W @ v)
            v_norm = v.norm()
            if v_norm > 0:
                v = v / v_norm
    
    # Rayleigh quotient estimate
    v_d = v.detach()
    Wv = W @ v_d
    spectral_rad = (v_d.T @ Wv).abs() / (v_d.T @ v_d + 1e-8)
    
    # Limit: 1/d (rough heuristic - complete DAG has spectral radius ~ sqrt(d))
    limit = 1.0 / max(d, 1)
    
    energy = F.relu(spectral_rad - limit)
    return energy.squeeze()


# === scCausal v2 Model ===
class scCausalV2(nn.Module):
    """NB loss + spectral norm + low-rank factorization."""
    def __init__(self, d, rank=32):
        super().__init__()
        self.d = d
        self.rank = min(rank, d)
        self.nb_loss = NBLikelihood(d)
        
        # Low-rank: W = U @ V^T
        self.U = nn.Parameter(torch.randn(d, self.rank) * 0.1)
        self.V = nn.Parameter(torch.randn(d, self.rank) * 0.1)
    
    def get_W(self):
        W = self.U @ self.V.T
        return W - torch.diag(torch.diag(W))
    
    def forward(self, X, library_size=None):
        W = self.get_W()
        nll, mu, theta = self.nb_loss(X, W, library_size)
        acyc_energy = spectral_norm_energy(W)
        return nll, acyc_energy, W, mu, theta


# === Training ===
def train_sccausal_v2(model, X, n_outer=100, n_inner=200, lr=0.002,
                       lambda1=0.0005, lambda_acyc=10.0, device='cuda',
                       verbose=True, library_size=None):
    """Train scCausal v2 with spectral norm acyclicity penalty."""
    X = X.to(device)
    model = model.to(device)
    if library_size is not None:
        library_size = library_size.to(device)
    
    best_loss = float('inf')
    pbar = tqdm(range(n_outer), desc='scCausal v2') if verbose else range(n_outer)
    
    for outer in pbar:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for inner in range(n_inner):
            optimizer.zero_grad()
            nll, acyc, W, mu, theta = model(X, library_size)
            l1 = lambda1 * W.abs().sum()
            loss = nll + l1 + lambda_acyc * acyc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        
        with torch.no_grad():
            _, acyc_val, W_final, _, _ = model(X, library_size)
            acyc_np = acyc_val.item()
            n_edges = (W_final.abs() > 0.3).sum().item()
            n_nonzero = (W_final.abs() > 1e-4).sum().item()
        
        if verbose:
            pbar.set_postfix({
                'acyc': f'{acyc_np:.2e}',
                'edges': n_edges,
                'nonzero': n_nonzero
            })
        
        if acyc_np < 1e-6 and n_edges > 0:
            if verbose: pbar.set_description('Converged!')
            break
    
    return model.get_W().detach()


# === Edge Extraction ===
def extract_edges(W, threshold=0.3):
    W_np = W.cpu().numpy()
    edges = []
    for i in range(W_np.shape[0]):
        for j in range(W_np.shape[1]):
            if i != j and abs(W_np[i, j]) > threshold:
                edges.append({'source': int(i), 'target': int(j), 'weight': float(W_np[i, j])})
    return edges


print('scCausal v2 loaded: NB + SpectralNorm + LowRank')
