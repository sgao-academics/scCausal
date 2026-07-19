"""
scCausal CI: PC algorithm + fast NB CI test (statsmodels IRLS).
"""
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
import time

def nb_ci_test_fast(X_data, i, j, condition_set, alpha=0.05):
    """Fast NB CI test using statsmodels IRLS."""
    n = X_data.shape[0]
    
    if len(condition_set) == 0:
        X_null = np.ones((n, 1))
        X_alt = np.column_stack([np.ones(n), X_data[:, i]])
    else:
        cond_cols = [X_data[:, k] for k in condition_set]
        X_null = np.column_stack([np.ones(n)] + cond_cols)
        X_alt = np.column_stack([np.ones(n)] + cond_cols + [X_data[:, i]])
    
    y = X_data[:, j]
    
    fam = sm.families.NegativeBinomial(alpha=1.0)
    
    try:
        ll0 = sm.GLM(y, X_null, family=fam).fit(disp=0).llf
        ll1 = sm.GLM(y, X_alt, family=fam).fit(disp=0).llf
        LR = max(2 * (ll1 - ll0), 0)
        p_val = 1 - chi2.cdf(LR, df=1)
        return p_val > alpha, p_val, LR
    except:
        # If fit fails, assume dependent (keep edge)
        return False, 1.0, 0.0


def pc_skeleton_fast(X_data, gene_names, max_cond_size=1, alpha=0.05, 
                      prefilter_corr=0.1, verbose=True):
    """PC skeleton with NB CI test + correlation pre-filter for speed."""
    n, d = X_data.shape
    
    # Pre-filter: compute correlation on log1p data
    X_log = np.log1p(X_data)
    corr = np.corrcoef(X_log.T)
    
    # Start with edges where |corr| > prefilter_corr
    adj = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(i+1, d):
            if abs(corr[i, j]) > prefilter_corr:
                adj[i, j] = adj[j, i] = True
    
    n_prefilter = int(adj.sum() / 2)
    if verbose:
        print(f'After corr pre-filter (|r|>{prefilter_corr}): {n_prefilter} edges')
    
    # Marginal CI (|S|=0)
    if verbose: print('Marginal CI tests...')
    tested = 0
    for i in range(d):
        for j in range(i+1, d):
            if not adj[i, j]: continue
            indep, p, lr = nb_ci_test_fast(X_data, i, j, set(), alpha)
            tested += 1
            if indep:
                adj[i, j] = adj[j, i] = False
            if tested % 10 == 0 and verbose:
                n_edges = int(adj.sum() / 2)
                print(f'  {tested} tests, {n_edges} edges remaining')
    
    n_mar = int(adj.sum() / 2)
    if verbose: print(f'After marginal CI: {n_mar} edges')
    
    # First-order CI (|S|=1)
    if max_cond_size >= 1 and n_mar > 0:
        if verbose: print('First-order CI tests...')
        tested = 0
        for i in range(d):
            neighbors = np.where(adj[i, :])[0]
            for j in neighbors:
                if j <= i: continue
                if not adj[i, j]: continue
                common = [k for k in range(d) if k != i and k != j and 
                         (adj[i, k] or adj[k, i] or adj[j, k] or adj[k, j])]
                for k in common[:5]:  # limit conditioning candidates
                    if not adj[i, j]: break
                    indep, p, lr = nb_ci_test_fast(X_data, i, j, {k}, alpha)
                    tested += 1
                    if indep:
                        adj[i, j] = adj[j, i] = False
                
                if tested % 20 == 0 and verbose:
                    n_e = int(adj.sum() / 2)
                    print(f'  {tested} CI tests, {n_e} edges')
    
    n_final = int(adj.sum() / 2)
    if verbose: print(f'Final: {n_final} edges')
    return adj


def extract_edges(adj, gene_names):
    d = adj.shape[0]
    edges = []
    for i in range(d):
        for j in range(i+1, d):
            if adj[i, j]:
                edges.append({'source': int(i), 'target': int(j),
                              'source_name': gene_names[i], 'target_name': gene_names[j],
                              'weight': 1.0})
    return edges


print('scCausal CI (fast) loaded')
