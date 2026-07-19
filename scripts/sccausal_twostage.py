"""
scCausal two-stage: MSE discovers skeleton, NB refines edge weights.
Novel contribution: two-stage causal discovery for count data.
Stage 1: MSE + matrix_exp → find which edges exist
Stage 2: NB loss → adjust edge weights to match count distribution
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class TwoStageCausal(nn.Module):
    def __init__(self, d, rank=32):
        super().__init__()
        self.d = d
        self.rank = min(rank, d)
        self.U = nn.Parameter(torch.randn(d, self.rank) * 0.1)
        self.V = nn.Parameter(torch.randn(d, self.rank) * 0.1)
        # NB parameters (for stage 2)
        self.log_theta = nn.Parameter(torch.zeros(d))
    
    def get_W(self):
        W = self.U @ self.V.T
        return W - torch.diag(torch.diag(W))
    
    def get_edge_mask(self, W_init, threshold=0.3):
        """Create binary mask from initial edge structure."""
        mask = (W_init.abs() > threshold).float()
        mask = mask - torch.diag(torch.diag(mask))  # zero diag
        return mask


def train_stage1(X, d, rank=32, n_outer=80, n_inner=300, device='cuda'):
    """Stage 1: MSE + matrix_exp DAG constraint. Find skeleton."""
    import torch.nn as nn
    
    class Stage1Model(nn.Module):
        def __init__(self, d, rank):
            super().__init__()
            self.U = nn.Parameter(torch.randn(d, rank, device=device) * 0.1)
            self.V = nn.Parameter(torch.randn(d, rank, device=device) * 0.1)
        def get_W(self):
            W = self.U @ self.V.T
            return W - torch.diag(torch.diag(W))
    
    X = X.to(device)
    model = Stage1Model(d, rank).to(device)
    
    rho, alpha = 1.0, torch.tensor(0.0, device=device)
    
    for outer in range(n_outer):
        opt = torch.optim.Adam(model.parameters(), lr=0.002)
        for _ in range(n_inner):
            opt.zero_grad()
            W = model.get_W()
            recon = X - X @ W
            mse = (recon ** 2).mean()
            M = W * W
            h = torch.trace(torch.matrix_exp(M)) - d
            loss = mse + 0.001 * W.abs().sum() + alpha*h + 0.5*rho*h**2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        with torch.no_grad():
            h_new = torch.trace(torch.matrix_exp(model.get_W() * model.get_W())) - d
        alpha = alpha + rho * h_new.item()
        if h_new > 0.25 * h: rho = min(rho*10, 1e10)
        if h_new < 1e-8: break
    
    return model.get_W().detach()


def train_stage2(W_init, X, d, rank=32, lr=0.001, lambda1=0.0001, 
                  n_iter=500, device='cuda'):
    """Stage 2: NB loss to refine edge weights, constrained by skeleton.
    
    Only edges found in Stage 1 are non-zero. The NB loss adjusts
    their weights to match the count distribution.
    """
    class Stage2Model(nn.Module):
        def __init__(self, d, rank, edge_mask):
            super().__init__()
            self.d = d
            self.rank = rank
            self.U = nn.Parameter(torch.randn(d, rank, device=device) * 0.1)
            self.V = nn.Parameter(torch.randn(d, rank, device=device) * 0.1)
            self.log_theta = nn.Parameter(torch.zeros(d, device=device))
            self.register_buffer('edge_mask', edge_mask)
        
        def get_W(self):
            W = self.U @ self.V.T
            W = W - torch.diag(torch.diag(W))
            # Apply skeleton mask: only edges from Stage 1 survive
            return W * self.edge_mask
    
    X = X.to(device)
    edge_mask = (W_init.abs() > 0.3).float().to(device)
    edge_mask = edge_mask - torch.diag(torch.diag(edge_mask))
    n_edges_stage1 = int(edge_mask.sum().item())
    print(f'  Stage 1 skeleton: {n_edges_stage1} edges')
    
    model = Stage2Model(d, rank, edge_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    eps = 1e-8
    pbar = tqdm(range(n_iter), desc='Stage2 NB')
    
    for it in pbar:
        optimizer.zero_grad()
        W = model.get_W()
        
        # NB likelihood
        lib = X.sum(1, keepdim=True).float()
        linear = X.float() @ W
        log_mu = lib.log() + linear - 5.0
        mu = torch.clamp(torch.exp(log_mu), min=eps)
        theta = torch.clamp(torch.exp(model.log_theta), min=1.0, max=1000.0)
        
        xf = X.float()
        t1 = torch.lgamma(xf + theta + eps) - torch.lgamma(theta + eps)
        t2 = -torch.lgamma(xf + 1.0)
        t3 = theta * torch.log(theta + eps)
        t4 = xf * torch.log(mu + eps)
        t5 = -(theta + xf) * torch.log(theta + mu + eps)
        ll = t1 + t2 + t3 + t4 + t5
        nll = -ll.mean()
        
        l1 = lambda1 * W.abs().sum()
        loss = nll + l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        if it % 50 == 0:
            with torch.no_grad():
                n_edges = (model.get_W().abs() > 0.3).sum().item()
                pbar.set_postfix({'nll': f'{nll.item():.1f}', 'edges': n_edges})
    
    return model.get_W().detach()


def extract_edges(W, threshold=0.3):
    W_np = W.cpu().numpy()
    edges = []
    for i in range(W_np.shape[0]):
        for j in range(W_np.shape[1]):
            if i != j and abs(W_np[i, j]) > threshold:
                edges.append({'source': int(i), 'target': int(j), 'weight': float(W_np[i, j])})
    return edges


print('Two-stage scCausal loaded: MSE skeleton + NB refinement')
