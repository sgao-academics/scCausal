"""
scCausal: Zero-Inflated Negative Binomial Causal Discovery for Single-Cell RNA-seq
==========================================================================
Standalone implementation. No dependency on causalscale.
Key innovations:
  1. ZINB likelihood replacing NOTEARS MSE
  2. Low-rank W = UV^T factorization
  3. Joint dropout probability estimation
  4. GPU optimization with float32 stability
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json, os, time

# === ZINB Log-Likelihood Module ===
class ZINBLoss(nn.Module):
    """Zero-inflated negative binomial likelihood for scRNA-seq counts.
    
    X_ij ~ ZINB(mu_ij, theta_j, pi_ij):
      mu_ij = (XW)_ij (linear SEM mean)
      theta_j: gene-specific dispersion
      pi_ij: dropout prob = sigmoid(gamma_j * log(mu+1) + beta_j)
    """
    def __init__(self, d, dropout=True):
        super().__init__()
        self.d = d
        self.use_dropout = dropout
        self.log_theta = nn.Parameter(torch.zeros(d))
        self.gamma = nn.Parameter(torch.full((d,), -0.5))
        self.beta = nn.Parameter(torch.full((d,), -1.0))
    
    def _nb_log_prob(self, x, mu, theta):
        eps = 1e-8
        x_f = x.float()
        t1 = torch.lgamma(x_f + theta + eps) - torch.lgamma(theta + eps)
        t2 = -torch.lgamma(x_f + 1.0)
        t3 = theta * torch.log(theta + eps)
        t4 = x_f * torch.log(mu + eps)
        t5 = -(theta + x_f) * torch.log(theta + mu + eps)
        return t1 + t2 + t3 + t4 + t5
    
    def forward(self, X, W):
        """NLL mean per entry."""
        eps = 1e-8
        mu = torch.clamp(torch.matmul(X.float(), W), min=0.1)
        theta = torch.clamp(torch.exp(self.log_theta), min=0.01, max=1000.0)
        
        nb_ll = self._nb_log_prob(X, mu, theta)
        
        if self.use_dropout:
            log_mu_p1 = torch.log(mu + 1.0)
            pi = torch.sigmoid(self.gamma * log_mu_p1 + self.beta)
            nb_p0 = torch.exp(self._nb_log_prob(torch.zeros_like(mu), mu, theta))
            is_zero = (X == 0).float()
            p_zero = pi + (1 - pi) * nb_p0
            log_p_zero = torch.log(torch.clamp(p_zero, min=eps))
            log_p_nonzero = torch.log(torch.clamp(1 - pi, min=eps)) + nb_ll
            ll = is_zero * log_p_zero + (1 - is_zero) * log_p_nonzero
        else:
            ll = nb_ll
        
        return -ll.mean()


# === DAG Constraint ===
def dag_constraint(W):
    M = W * W
    expM = torch.matrix_exp(M)
    return torch.trace(expM) - W.shape[0]


# === Rank Estimation ===
def estimate_rank(X, variance_thresh=0.80):
    with torch.no_grad():
        X_c = X.float() - X.float().mean(0, keepdim=True)
        U, S, V = torch.svd_lowrank(X_c, q=min(200, min(X.shape)-1))
        total_var = (X_c ** 2).sum()
        cum_var = torch.cumsum(S ** 2, 0)
        ranks = torch.where(cum_var / total_var >= variance_thresh)[0]
        r = ranks[0].item() + 1 if len(ranks) > 0 else len(S)
        return min(max(r, 16), 128)


# === scCausal Model (nn.Module) ===
class scCausalModel(nn.Module):
    """Full scCausal model: ZINB loss + low-rank W + DAG constraint."""
    def __init__(self, d, rank=None, dropout=True):
        super().__init__()
        self.d = d
        self.rank = rank
        self.zinb = ZINBLoss(d, dropout=dropout)
        
        if rank is not None and rank < d:
            # Larger init so W = UV^T starts with non-trivial values
            self.U = nn.Parameter(torch.randn(d, rank) * 0.1)
            self.V = nn.Parameter(torch.randn(d, rank) * 0.1)
            self.use_lowrank = True
        else:
            self.W_flat = nn.Parameter(torch.randn(d * d) * 0.01)
            self.use_lowrank = False
    
    def get_W(self):
        if self.use_lowrank:
            W = self.U @ self.V.T
        else:
            W = self.W_flat.view(self.d, self.d)
        return W - torch.diag(torch.diag(W))
    
    def forward(self, X):
        W = self.get_W()
        nll = self.zinb(X, W)
        h = dag_constraint(W)
        return nll, h, W


# === Augmented Lagrangian Trainer ===
def train_sccausal(model, X, n_outer=200, n_inner=500, lr=0.002,
                   lambda1=0.0005, rho0=1.0, rho_factor=10.0, 
                   h_tol=1e-8, device='cuda', verbose=True, warmup=50):
    """Train scCausal with augmented Lagrangian.
    
    Uses MSE warmup for the first `warmup` outer iterations to escape the
    W=0 trivial solution, then switches to ZINB likelihood for fine-tuning.
    """
    X = X.to(device)
    model = model.to(device)
    
    rho = rho0
    alpha = torch.tensor(0.0, device=device)
    
    pbar = tqdm(range(n_outer), desc='scCausal') if verbose else range(n_outer)
    
    for outer in pbar:
        use_mse = (outer < warmup)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for inner in range(n_inner):
            optimizer.zero_grad()
            W = model.get_W()
            h_val = dag_constraint(W)
            
            if use_mse:
                # MSE warmup: strong gradients to move W away from zero
                recon = X - X @ W
                data_loss = (recon ** 2).mean()
            else:
                data_loss = model.zinb(X, W)
            
            l1 = lambda1 * W.abs().sum()
            loss = data_loss + l1 + alpha * h_val + 0.5 * rho * h_val ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            if inner % 100 == 50 and h_val.item() < h_tol * 10:
                break
        
        with torch.no_grad():
            W_final = model.get_W()
            h_val_np = dag_constraint(W_final).item()
            n_edges = (W_final.abs() > 0.3).sum().item()
        
        alpha = alpha + rho * h_val_np
        if h_val_np > 0.25 * h_val.item():
            rho = min(rho * rho_factor, 1e10)
        
        status = 'warmup' if use_mse else 'ZINB'
        if verbose:
            pbar.set_postfix({'h': f'{h_val_np:.2e}', 'edges': n_edges, 
                            'rho': f'{rho:.1f}', 'mode': status})
        
        if h_val_np < h_tol:
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
                edges.append({'source': i, 'target': j, 'weight': float(W_np[i, j])})
    return edges


print('scCausal v1.0 loaded: scCausalModel, train_sccausal, extract_edges, dag_constraint, estimate_rank')
