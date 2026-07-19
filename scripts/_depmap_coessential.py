import config  # scCausal data path configuration
"""
scCausal × DepMap CRISPR Co-Essentiality Validation
=====================================================
Tests whether scCausal-predicted causal edges show significant
CRISPR co-essentiality in DepMap 23Q4 data (>1000 cancer cell lines).

Co-essentiality = Pearson correlation of CRISPR gene effect scores
across cell lines. If gene A and gene B are causally linked (A->B),
perturbing A should affect B's dependency profile across diverse
genetic backgrounds — producing correlated CRISPR scores.

This is COMPLETELY INDEPENDENT of STRING and transcriptomic co-expression.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, fisher_exact, mannwhitneyu
import json, os, time

DEPMAP = r"os.path.dirname(config.get_path("depmap_crispr"))"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ====================================================
# 1. LOAD DepMap CRISPR data
# ====================================================
print("Loading DepMap CRISPRGeneEffect.csv (441 MB)...")
t0 = time.time()
df = pd.read_csv(os.path.join(DEPMAP, "CRISPRGeneEffect.csv"), index_col=0)
# Strip Entrez ID suffix: "GZMB (3002)" -> "GZMB"
df.columns = [c.split(' (')[0] for c in df.columns]
print(f"  Loaded in {time.time()-t0:.1f}s: {df.shape[0]} cell lines x {df.shape[1]} genes")
print(f"  Sample columns: {list(df.columns[:5])}")

# ====================================================
# 2. scCausal predicted causal edges
# ====================================================
# These are the STRING-validated edges from scCausal d=50 PBMC network
# (Fig 3). We test ALL scCausal edges, plus specific predictions.
scCausal_predictions = [
    # HLA/MHC class II module
    ('HLA-DPA1','HLA-DRA'), ('HLA-DPA1','CD74'), ('HLA-DRA','CD74'),
    # Immune effector module
    ('GNLY','NKG7'), ('GZMB','CCL5'), ('CCL5','GNLY'), ('CCL5','NKG7'),
    # Ribosomal
    ('RPL8','RPS27A'),
    # Mitochondrial
    ('MT-CO1','MT-CO2'), ('MT-CO2','MT-CYB'), ('MT-CO1','MT-CYB'),
    # S100 calcium-binding
    ('S100A6','S100A11'), ('S100A8','S100A9'),
    # Cytoskeletal
    ('ACTB','ARPC1B'),
]

# Testable predictions from Discussion §4.4
key_predictions = [
    ('GZMB','CCL5'),
    ('S100A8','S100A9'),
    ('HLA-DPA1','HLA-DRA'),
    ('GNLY','NKG7'),
]

# ====================================================
# 3. Compute co-essentiality for all scCausal edges
# ====================================================
print("\nComputing co-essentiality for scCausal edges...")
results = []
genes_found = set()
genes_missing = set()

for g1, g2 in scCausal_predictions:
    # Check if genes are in DepMap
    in_depmap = True
    for g in [g1, g2]:
        if g not in df.columns:
            # Try with suffix
            alternatives = [c for c in df.columns if c.startswith(g + ' ')]
            if alternatives:
                in_depmap = False
                genes_missing.add(g)
            else:
                in_depmap = False
                genes_missing.add(g)
    
    if not in_depmap:
        # Try exact match
        if g1 in df.columns and g2 in df.columns:
            pass
        else:
            continue
    
    if g1 in df.columns and g2 in df.columns:
        g1_col = g1
        g2_col = g2
    else:
        continue
    
    genes_found.add(g1)
    genes_found.add(g2)
    
    # Compute Pearson correlation
    corr, pval = pearsonr(df[g1_col], df[g2_col])
    results.append({
        'gene1': g1, 'gene2': g2,
        'correlation': corr, 'p_value': pval,
        'significant': pval < 0.05,
    })

print(f"  Found in DepMap: {len(genes_found)} genes")
print(f"  Missing: {genes_missing}")
print(f"  Edges with both genes: {len(results)}/{len(scCausal_predictions)}")

# ====================================================
# 4. Background distribution (random gene pairs)
# ====================================================
print("\nComputing background co-essentiality distribution...")
# Sample random gene pairs from all DepMap genes
all_genes = list(df.columns)
np.random.seed(42)
n_random = 10000
bg_corrs = []
for _ in range(n_random):
    g1 = all_genes[np.random.randint(0, len(all_genes))]
    g2 = all_genes[np.random.randint(0, len(all_genes))]
    if g1 != g2:
        corr, _ = pearsonr(df[g1], df[g2])
        bg_corrs.append(corr)

bg_corrs = np.array(bg_corrs)
bg_mean = np.mean(bg_corrs)
bg_std = np.std(bg_corrs)

# ====================================================
# 5. Statistical tests
# ====================================================
print("\n=== STATISTICAL ANALYSIS ===")

sc_corrs = np.array([r['correlation'] for r in results])
sig_count = sum(1 for r in results if r['significant'])

print(f"\nscCausal edges:")
print(f"  Mean |correlation|: {np.mean(np.abs(sc_corrs)):.4f}")
print(f"  Significant (p<0.05): {sig_count}/{len(results)} ({100*sig_count/len(results):.1f}%)")

# Mann-Whitney U test: are scCausal |correlations| higher than background?
stat, p_mw = mannwhitneyu(np.abs(sc_corrs), np.abs(bg_corrs), alternative='greater')
print(f"\nMann-Whitney U (|corr| > background): p = {p_mw:.4f}")

# Fisher's exact test for enrichment of significant pairs
bg_sig = np.sum(np.abs(bg_corrs) > np.percentile(np.abs(bg_corrs), 95))
table = [[sig_count, len(results)-sig_count],
         [bg_sig, n_random-bg_sig]]
odds_ratio, p_fisher = fisher_exact(table, alternative='greater')
print(f"Fisher exact (sig enrichment): OR = {odds_ratio:.2f}, p = {p_fisher:.4f}")

# ====================================================
# 6. Key predictions
# ====================================================
print("\n=== KEY PREDICTIONS ===")
for g1, g2 in key_predictions:
    match = [r for r in results if (r['gene1']==g1 and r['gene2']==g2) or 
                                   (r['gene1']==g2 and r['gene2']==g1)]
    if match:
        r = match[0]
        sig_str = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
        print(f"  {g1}--{g2}: r = {r['correlation']:.3f}, p = {r['p_value']:.4f} {sig_str}")
    else:
        print(f"  {g1}--{g2}: NOT FOUND in DepMap")

# ====================================================
# 7. SAVE results
# ====================================================
output = {
    'n_cell_lines': int(df.shape[0]),
    'n_genes_depmap': int(df.shape[1]),
    'scCausal_edges_tested': len(results),
    'scCausal_significant': sig_count,
    'scCausal_mean_abs_corr': float(np.mean(np.abs(sc_corrs))),
    'background_mean_abs_corr': float(np.mean(np.abs(bg_corrs))),
    'background_n_samples': n_random,
    'mann_whitney_p': float(p_mw),
    'fisher_OR': float(odds_ratio),
    'fisher_p': float(p_fisher),
    'key_predictions': {f'{r["gene1"]}-{r["gene2"]}': {
        'correlation': float(r['correlation']),
        'p_value': float(r['p_value']),
        'significant': bool(r['significant']),
    } for r in results if (r['gene1'], r['gene2']) in key_predictions or 
                           (r['gene2'], r['gene1']) in key_predictions},
    'all_edges': [{'pair': f'{r["gene1"]}-{r["gene2"]}', 
                   'correlation': float(r['correlation']),
                   'p_value': float(r['p_value'])} for r in results],
}
with open(os.path.join(OUT_DIR, 'depmap_coessentiality.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to depmap_coessentiality.json")
print("Done!")
