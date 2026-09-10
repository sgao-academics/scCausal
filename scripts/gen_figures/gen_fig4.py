"""
Fig 4: Validation & Robustness -- 2x2 panel composite.
Colorblind-friendly palette (Paul Tol "bright").

Every panel is read from the FAIR-protocol result files (the same run that
produces Table 1); nothing is hardcoded.

DATA PROVENANCE
  (a) functional composition of the STRING-validated d=30 gene set
      -> results/fair_downstream.json  (key: go_d30)
  (b) STRING precision vs combined-score threshold (400-900)
      -> results/fair_downstream.json  (key: threshold)
  (c) library-size GLM offset ablation
      -> results/fair_supplementary.json (keys: offset_d*)
         baseline  -> results/checkpoints/fair_pbmc_d*.json (NB_moment)
  (d) cross-tissue NB-LR precision, PBMC vs Paul15
      -> results/checkpoints/fair_{pbmc,paul15}_d*.json (NB_moment)
"""
import os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
CKPT_DIR = os.path.join(RES_DIR, 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)

C = {'nb': '#EE7733', 'fz': '#0077BB', 'pau': '#EE3377', 'gn': '#009988',
     'bg': '#FFFFFF', 'grid': '#E8E8E8', 'grey': '#BBBBBB'}

plt.style.use(['science', 'no-latex', 'bright'])
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BDBDBD', 'axes.linewidth': 0.8,
})


def _j(p):
    with open(os.path.join(RES_DIR, p), encoding='utf-8') as fh:
        return json.load(fh)


def _ck(p):
    with open(os.path.join(CKPT_DIR, p), encoding='utf-8') as fh:
        return json.load(fh)


D = [30, 50, 100, 200]
ds = _j('fair_downstream.json')
supp = _j('fair_supplementary.json')
pbmc = {d: _ck('fair_pbmc_d%d.json' % d) for d in D}
paul = {d: _ck('fair_paul15_d%d.json' % d) for d in D}

# === FIGURE ===
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(C['bg'])

# ---- (a) Reactome pathway co-membership enrichment ----
# Literature-curated, independent of STRING's co-expression channel.
ax = axes[0, 0]
r30, r50 = ds['reactome']['by_d']['30'], ds['reactome']['by_d']['50']
groups = [f"$d=30$\n({r30['n_edges']} edges,\n{r30['n_genes']} genes)",
          f"$d=50$\n({r50['n_edges']} edges,\n{r50['n_genes']} genes)"]
edge_pct = [r30['edge_pct'], r50['edge_pct']]
bg_pct = [r30['bg_pct'], r50['bg_pct']]
xg = np.arange(2); wg = 0.34
ax.bar(xg - wg / 2, edge_pct, wg, color=C['nb'], edgecolor='white', lw=0.5,
       label='STRING-validated edges')
ax.bar(xg + wg / 2, bg_pct, wg, color=C['grey'], edgecolor='white', lw=0.5,
       label='Background (all gene pairs)')
for i, r in enumerate([r30, r50]):
    ax.text(i - wg / 2, edge_pct[i] + 1.2, f"{r['edge_pct']:.1f}%",
            ha='center', fontsize=7, fontweight='bold', color=C['nb'])
    ax.text(i + wg / 2, bg_pct[i] + 1.2, f"{r['bg_pct']:.1f}%",
            ha='center', fontsize=7, fontweight='bold', color='#616161')
    ax.text(i - wg / 2, edge_pct[i] * 0.5,
            f"OR$\\,={r['odds_ratio']:.2f}$\n$p={r['fisher_p']:.1e}$",
            ha='center', va='center', fontsize=7.4, fontweight='bold', color='white')
ax.set_xticks(xg); ax.set_xticklabels(groups, fontsize=7)
ax.set_ylabel('Pathway co-membership of edge (%)')
ax.set_title('(a) Reactome co-membership (PBMC)', fontweight='bold')
ax.legend(fontsize=6.8, framealpha=0.9, loc='upper right')
ax.set_ylim(0, 122)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# ---- (b) STRING precision vs combined-score threshold ----
ax = axes[0, 1]
thr_axis = [400, 500, 600, 700, 800, 900]
cmap = {30: C['nb'], 50: C['fz'], 100: C['pau']}
for d in [30, 50, 100]:
    if str(d) not in ds['threshold']:
        continue
    rec = ds['threshold'][str(d)]
    nb = [rec['NB_moment'][str(t)]['precision'] for t in thr_axis]
    fz = [rec['Fisher_z'][str(t)]['precision'] for t in thr_axis]
    ax.plot(thr_axis, nb, 'o-', color=cmap[d], lw=2, ms=6, label=f'NB-LR, $d={d}$')
    ax.plot(thr_axis, fz, 's--', color=cmap[d], lw=1.6, ms=5, mfc='white',
            label=r"Fisher's $z$, $d=%d$" % d)
ax.set_xlabel('STRING combined-score threshold')
ax.set_ylabel('Validation precision (%)')
ax.set_title('(b) STRING threshold sensitivity ($d{=}30,50,100$)', fontweight='bold')
ax.legend(fontsize=6, framealpha=0.9, ncol=2)
ax.invert_xaxis()
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# ---- (c) library-size GLM offset ablation ----
ax = axes[1, 0]
no_off = [pbmc[d]['NB_moment']['precision'] for d in D]
with_off = [supp['offset_d%d' % d]['precision'] for d in D]
xl = np.arange(len(D)); wl = 0.36
ax.bar(xl - wl / 2, no_off, wl, color=C['fz'], edgecolor='white', lw=0.5,
       label=r'NB-LR, no offset')
ax.bar(xl + wl / 2, with_off, wl, color=C['nb'], edgecolor='white', lw=0.5,
       label=r'NB-LR, $+$ log(library size) offset')
for i in range(len(D)):
    dl = with_off[i] - no_off[i]
    clr = C['pau'] if dl > 0 else C['fz']
    ax.text(i, max(no_off[i], with_off[i]) + 0.4, f'{dl:+.2f}pp',
            ha='center', fontsize=7.2, fontweight='bold', color=clr)
ax.set_xticks(xl); ax.set_xticklabels([f'$d={d}$' for d in D])
ax.set_ylabel('STRING precision (%)')
ax.set_title('(c) Library-size offset (PBMC)', fontweight='bold')
ax.legend(fontsize=7, framealpha=0.9, loc='upper right')
ax.set_ylim(0, 18)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# ---- (d) cross-tissue NB-LR precision, PBMC vs Paul15 ----
ax = axes[1, 1]
pb = [pbmc[d]['NB_moment']['precision'] for d in D]
pa = [paul[d]['NB_moment']['precision'] for d in D]
x4 = np.arange(len(D)); w4 = 0.36
ax.bar(x4 - w4 / 2, pb, w4, color=C['fz'], edgecolor='white', lw=0.5,
       label='PBMC (peripheral blood)')
ax.bar(x4 + w4 / 2, pa, w4, color=C['pau'], edgecolor='white', lw=0.5,
       label='Paul15 (bone marrow)')
for i in range(len(D)):
    dd = pa[i] - pb[i]
    ax.text(i, max(pb[i], pa[i]) + 0.4, f'{dd:+.2f}pp', ha='center',
            fontsize=7.2, fontweight='bold', color=C['pau'] if dd > 0 else C['fz'])
ax.set_xticks(x4); ax.set_xticklabels([f'$d={d}$' for d in D])
ax.set_ylabel('NB-LR STRING precision (%)')
ax.set_title('(d) Cross-tissue NB-LR precision', fontweight='bold')
ax.legend(fontsize=7, framealpha=0.9, loc='upper right')
ax.set_ylim(0, 18)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

plt.tight_layout(pad=2.5, h_pad=2.2, w_pad=2.2)
for fmt in ['pdf', 'png']:
    plt.savefig(os.path.join(FIG_DIR, f'fig4_validation.{fmt}'), dpi=300,
                bbox_inches='tight', facecolor=C['bg'], edgecolor='none')
plt.close()

print('Fig 4 done. Source values:')
print('  (a) reactome d=30 / d=50      :',
      (r30['edge_pct'], r30['bg_pct'], r30['odds_ratio'], r30['fisher_p']),
      (r50['edge_pct'], r50['bg_pct'], r50['odds_ratio'], r50['fisher_p']))
print('      validated genes in universe: %d / %d'
      % (len(ds['go_d30']['validated_genes']), ds['go_d30']['universe_size']))
print('  (b) thresholds                :', thr_axis)
print('  (c) no-offset                 :', no_off)
print('      with-offset               :', with_off)
print('  (d) PBMC  / Paul15 NB precision:', pb, pa)
for fmt in ['pdf', 'png']:
    p = os.path.join(FIG_DIR, f'fig4_validation.{fmt}')
    print('  file %-4s %8d bytes' % (fmt, os.path.getsize(p)))
