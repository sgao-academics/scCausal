import config  # scCausal data path configuration
"""Full sweep: NB CI test on PBMC d=30/50/100/200 + update paper."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, time, gzip, json
from sccausal_ci_fast import pc_skeleton_fast, extract_edges
import scanpy as sc

RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# Load data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
adata = sc.read_h5ad(os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad'))
X_raw = np.array(adata.X.todense() if hasattr(adata.X,'todense') else adata.X, dtype=np.float32)
all_genes = list(adata.var_names.astype(str))

# STRING + TRRUST
alias_path = config.resolve('string_aliases', 'SC_CAUSAL_STRING_ALIASES')
ppi_path = config.resolve('string_ppi', 'SC_CAUSAL_STRING_PPI')
sym2str = {}
with gzip.open(alias_path,'rt',encoding='utf-8',errors='ignore') as f:
    for l in f:
        if l[0]=='#':continue
        p=l.strip().split('\t')
        if len(p)>=2 and not p[1].isdigit(): sym2str[p[1].upper()]=p[0]
ppi = set()
with gzip.open(ppi_path,'rt',encoding='utf-8',errors='ignore') as f:
    f.readline()
    for l in f:
        p=l.strip().split()
        try:
            if int(p[-1])>=700: ppi.add((p[0],p[1]))
        except:pass

trrust_pairs = set()
trrust_path = config.resolve('trrust', 'SC_CAUSAL_TRRUST')
with open(trrust_path) as f:
    for line in f:
        p=line.strip().split('\t')
        if len(p)>=2: trrust_pairs.add((p[0].upper(),p[1].upper()))

all_results = {}
for d_test in [30, 50, 100, 200]:
    top_idx = np.argsort(np.var(X_raw,0))[-d_test:]
    X_d = X_raw[:, top_idx]; genes_d = [all_genes[i] for i in top_idx]
    g2s = {g:sym2str.get(g.upper(),'') for g in genes_d}
    sparsity = 100*(X_d==0).sum()/X_d.size
    mapped = sum(1 for v in g2s.values() if v)
    
    print(f'\n{"="*50}')
    print(f'd={d_test}: {X_d.shape}, {sparsity:.1f}% zeros, STRING: {mapped}/{d_test}')
    
    t0 = time.time()
    adj = pc_skeleton_fast(X_d, genes_d, max_cond_size=1, alpha=0.05, 
                           prefilter_corr=[0.20,0.15,0.12,0.10][[30,50,100,200].index(d_test)],
                           verbose=('d' not in os.environ.get('QUIET','')))
    t = time.time() - t0
    
    edges = extract_edges(adj, genes_d)
    n = len(edges)
    
    s_val = sum(1 for e in edges if g2s.get(e['source_name'],'') and 
                g2s.get(e['target_name'],'') and 
                (g2s[e['source_name']], g2s[e['target_name']]) in ppi)
    sp = s_val/max(n,1)*100
    
    t_val = sum(1 for e in edges if (e['source_name'].upper(), e['target_name'].upper()) in trrust_pairs)
    tp = t_val/max(n,1)*100
    
    print(f'RESULT: {n} edges, STRING={s_val}({sp:.1f}%), TRRUST={t_val}({tp:.1f}%), {t:.0f}s')
    
    # Top STRING genes
    str_edges = [(e['source_name'], e['target_name']) for e in edges 
                 if g2s.get(e['source_name'],'') and g2s.get(e['target_name'],'') and
                 (g2s[e['source_name']], g2s[e['target_name']]) in ppi]
    
    all_results[f'd_{d_test}'] = {
        'd': d_test, 'edges': n, 'string_validated': s_val, 'string_precision': sp,
        'trrust_validated': t_val, 'trrust_precision': tp, 'time_s': t,
        'sparsity': sparsity, 'string_mapped': mapped,
        'top_string_edges': str_edges[:15]
    }

# Save
with open(os.path.join(RESULT_DIR, 'ci_sweep_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# Summary
print(f'\n{"="*60}')
print(f'NB CI SWEEP SUMMARY')
print(f'{"="*60}')
print(f'{"D":>4} {"Edges":>6} {"STRING#":>8} {"STRING%":>8} {"TRRUST#":>9} {"Time":>6}')
print(f'{"-"*45}')
for d in [30,50,100,200]:
    r = all_results[f'd_{d}']
    print(f'{d:>4} {r["edges"]:>6} {r["string_validated"]:>8} {r["string_precision"]:>7.1f}% {r["trrust_validated"]:>8} {r["time_s"]:>5.0f}s')

# Compare to v1
print(f'\nCOMPARISON TO v1 (NOTEARS-based):')
print(f'  v1 NOTEARS d=100:  246 edges, STRING=19.1%')
print(f'  v1 scCausal d=100:  258 edges, STRING=16.7%')
print(f'  v2 NB CI   d=100:  TBD above')
print(f'\nKey advantage of NB CI: statistically grounded, no DAG collapse, p-values for every edge.')

print(f'\nSaved: {RESULT_DIR}/ci_sweep_results.json')
