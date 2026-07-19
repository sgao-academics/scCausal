import config  # scCausal data path configuration
"""
Full sweep with checkpoint/resume. Runs in background.
Features:
  - d=30,50,100,200 (200 in background, rest foreground)
  - Checkpoint: saves after each d-value
  - Resume: skips completed d-values
  - Lock file: prevents duplicate runs
Usage: python sweep_full.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, time, gzip, json
from sccausal_ci_fast import pc_skeleton_fast, extract_edges
import scanpy as sc

RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CHECKPOINT = os.path.join(RESULT_DIR, '_sweep_checkpoint.json')
LOCKFILE = os.path.join(RESULT_DIR, '_sweep.lock')
os.makedirs(RESULT_DIR, exist_ok=True)

# Lock
if os.path.exists(LOCKFILE):
    print(f'Lock file exists. Another sweep may be running. Remove {LOCKFILE} if stale.')
    sys.exit(1)
with open(LOCKFILE, 'w') as f: f.write(str(os.getpid()))

# Load checkpoint
all_results = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f: all_results = json.load(f)
    print(f'Resumed from checkpoint: {list(all_results.keys())}')

# Load data once
print('Loading PBMC...')
adata = sc.read_h5ad(os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad'))
X_raw = np.array(adata.X.todense() if hasattr(adata.X, 'todense') else adata.X, dtype=np.float32)
all_genes = list(adata.var_names.astype(str))

# STRING (load once)
alias_path = r'config.get_path("string_aliases")'
ppi_path = r'config.get_path("string_ppi")'
sym2str = {}
with gzip.open(alias_path,'rt',encoding='utf-8', errors='ignore') as f:
    for l in f:
        if l[0]=='#': continue
        p=l.strip().split('\t')
        if len(p)>=2 and not p[1].isdigit(): sym2str[p[1].upper()]=p[0]
ppi = set()
with gzip.open(ppi_path,'rt',encoding='utf-8', errors='ignore') as f:
    f.readline()
    for l in f:
        p=l.strip().split()
        try:
            if int(p[-1])>=700: ppi.add((p[0],p[1]))
        except:pass
print(f'STRING: {len(ppi)//1000}K HC pairs')

# TRRUST
trrust_pairs = set()
with open(r'config.get_path("trrust")') as f:
    for l in f:
        p=l.strip().split('\t')
        if len(p)>=2: trrust_pairs.add((p[0].upper(),p[1].upper()))

# Run sweep
D_VALUES = [30, 50, 100, 200]
PRE_FILTER = {30: 0.20, 50: 0.15, 100: 0.12, 200: 0.10}

for d in D_VALUES:
    key = f'd_{d}'
    if key in all_results:
        r = all_results[key]
        print(f'\nSKIP d={d}: already done ({r["edges"]} edges, STRING={r["string_precision"]:.1f}%)')
        continue
    
    print(f'\n{"="*50}')
    print(f'D={d}')
    print(f'{"="*50}')
    
    top_idx = np.argsort(np.var(X_raw,0))[-d:]
    X_d = X_raw[:, top_idx]; genes_d = [all_genes[i] for i in top_idx]
    g2s = {g:sym2str.get(g.upper(),'') for g in genes_d}
    mapped = sum(1 for v in g2s.values() if v)
    sparsity = 100*(X_d==0).sum()/X_d.size
    print(f'  {X_d.shape}, {sparsity:.0f}% zeros, STRING: {mapped}/{d}')
    
    t0 = time.time()
    adj = pc_skeleton_fast(X_d, genes_d, max_cond_size=1, alpha=0.05,
                           prefilter_corr=PRE_FILTER[d], verbose=True)
    t = time.time() - t0
    
    edges = extract_edges(adj, genes_d)
    n = len(edges)
    
    s_val = sum(1 for e in edges if g2s.get(e['source_name'],'') and g2s.get(e['target_name'],'') and (g2s[e['source_name']], g2s[e['target_name']]) in ppi)
    sp = s_val/max(n,1)*100
    t_val = sum(1 for e in edges if (e['source_name'].upper(), e['target_name'].upper()) in trrust_pairs)
    tp = t_val/max(n,1)*100
    
    str_edges = [(e['source_name'], e['target_name']) for e in edges if g2s.get(e['source_name'],'') and g2s.get(e['target_name'],'') and (g2s[e['source_name']], g2s[e['target_name']]) in ppi]
    
    all_results[key] = {
        'd': d, 'edges': n, 'string_validated': s_val, 'string_precision': sp,
        'trrust_validated': t_val, 'trrust_precision': tp, 'time_s': t,
        'sparsity': sparsity, 'string_mapped': mapped,
        'top_string_edges': str_edges[:20]
    }
    
    print(f'  DONE: {n} edges, STRING={s_val}({sp:.1f}%), TRRUST={t_val}({tp:.1f}%), {t:.0f}s')
    
    # Save checkpoint
    with open(CHECKPOINT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'  Checkpoint saved')

# Final save
with open(os.path.join(RESULT_DIR, 'ci_full_sweep.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# Summary
print(f'\n{"="*60}')
print(f'FULL SWEEP COMPLETE')
print(f'{"="*60}')
print(f'{"D":>4} {"Edges":>6} {"STRING#":>8} {"STRING%":>8} {"TRRUST#":>9} {"Time":>6}')
print(f'{"-"*45}')
for d in D_VALUES:
    r = all_results.get(f'd_{d}', {})
    if r:
        print(f'{d:>4} {r["edges"]:>6} {r["string_validated"]:>8} {r["string_precision"]:>7.1f}% {r["trrust_validated"]:>8} {r["time_s"]:>5.0f}s')

# Cleanup
os.remove(LOCKFILE)
print(f'\nLock released. Results: {RESULT_DIR}/ci_full_sweep.json')
