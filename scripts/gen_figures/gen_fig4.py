"""
Fig 4: Validation & Robustness — 2x2 panel composite.
All panels contain results NOT duplicated in Fig 2.
Colorblind-friendly palette (Paul Tol "bright").
"""
import os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)

# Colorblind-friendly palette
C = {
    'nb': '#EE7733', 'fz': '#0077BB', 'pau': '#EE3377', 'gn': '#009988',
    'bg': '#FFFFFF', 'grid': '#E8E8E8', 'grey': '#BBBBBB',
}

plt.style.use(['science', 'no-latex', 'bright'])
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BDBDBD', 'axes.linewidth': 0.8,
})

# === LOAD DATA ===
bp = json.load(open(os.path.join(RES_DIR, '_best_paper_ckpt.json')))
ek = json.load(open(os.path.join(RES_DIR, '_everything_ckpt.json')))

go_data = bp.get('go_enrich', {})

# STRING multi-threshold from _everything_ckpt
string_thresholds = {}
for k, v in ek.items():
    if k.startswith('G_string_'):
        parts = k.split('_')
        d_str = parts[2]  # 'd30'
        t_str = parts[3]  # 't400'
        d = int(d_str[1:])
        t = int(t_str[1:])
        if d not in string_thresholds: string_thresholds[d] = {}
        string_thresholds[d][t] = v.get('pct', v.get('string_pct', 0))

# === FIGURE ===
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor(C['bg'])

# (a) GO Functional Enrichment
ax = axes[0, 0]
go_keys = sorted(go_data.keys(), key=lambda c: go_data[c]['found'], reverse=True)
found_vals = [go_data[c]['found'] for c in go_keys]
total_vals = [go_data[c]['total_in_cat'] for c in go_keys]
pct_vals = [100*go_data[c]['found']/go_data[c]['total_in_cat'] for c in go_keys]
x = np.arange(len(go_keys))
ax.bar(x, total_vals, 0.55, color=C['grey'], edgecolor='white', lw=0.5, alpha=0.5, label='Total genes')
ax.bar(x, found_vals, 0.55, color=C['nb'], edgecolor='white', lw=0.5, label='Found by scCausal')
for i in range(len(go_keys)):
    ax.text(i, total_vals[i]+0.12, f'{found_vals[i]}/{total_vals[i]}\n{pct_vals[i]:.0f}%',
            ha='center', fontsize=7, fontweight='bold', color=C['nb'])
short_names = [n.replace('MHC class II','MHC-II').replace('immune effector','immune')[:15] for n in go_keys]
ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=7, rotation=20, ha='right')
ax.set_ylabel('Gene Count'); ax.set_title('(a) GO Enrichment ($d{=}30$, PBMC)', fontweight='bold')
ax.legend(fontsize=7, framealpha=0.85); ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (b) STRING Multi-Threshold Precision
ax = axes[0, 1]
for d, tdata in sorted(string_thresholds.items()):
    if d > 100: continue  # d=200+ not in threshold data
    ts = sorted(tdata.keys())
    pcts = [tdata[t] for t in ts]
    marker = 'o' if d==30 else 's' if d==50 else 'D'
    ms = 8 if d==30 else 7
    color = C['nb'] if d==30 else C['fz'] if d==50 else C['pau']
    ax.plot(ts, pcts, marker=marker, color=color, lw=2, ms=ms, label=f'd={d}')
ax.set_xlabel('STRING Combined Score Threshold'); ax.set_ylabel('Validation Precision (%)')
ax.set_title('(b) STRING Multi-Threshold (PBMC)', fontweight='bold')
ax.legend(fontsize=7, framealpha=0.85, title='Dimension', title_fontsize=7)
ax.invert_xaxis(); ax.yaxis.grid(True, alpha=0.12, color=C['grid'])
ax.set_ylim(0, 35)

# (c) Library Size Offset Ablation
ax = axes[1, 0]
D_lib = ['$d{=}30$', '$d{=}50$', '$d{=}100$', '$d{=}200$']
no_off = [14.8, 8.0, 4.6, 3.5]   # NB-LR without offset
with_off = [11.9, 8.8, 6.4, 5.9]  # NB-LR with library size offset
x_lib = np.arange(len(D_lib)); w_lib = 0.35
ax.bar(x_lib - w_lib/2, no_off, w_lib, color=C['fz'], edgecolor='white', lw=0.5, label='No offset')
ax.bar(x_lib + w_lib/2, with_off, w_lib, color=C['nb'], edgecolor='white', lw=0.5, label='With GLM offset')
for i in range(len(D_lib)):
    delta = with_off[i] - no_off[i]
    clr = '#EE3377' if delta > 0 else '#0077BB'
    ax.text(i, max(no_off[i], with_off[i])+0.6, f'{delta:+.1f}pp',
            ha='center', fontsize=7.5, fontweight='bold', color=clr)
ax.set_xticks(x_lib); ax.set_xticklabels(D_lib)
ax.set_ylabel('STRING Precision (%)'); ax.set_title('(c) Library Size Offset (PBMC)', fontweight='bold')
ax.legend(fontsize=7, framealpha=0.85); ax.yaxis.grid(True, alpha=0.12, color=C['grid'])
ax.set_ylim(0, 18)

# (d) Cross-Tissue Precision (PBMC vs Paul15)
ax = axes[1, 1]
D4 = [30, 50, 100, 200]
pb_nb = [14.8, 8.0, 4.6, 3.5]
pa_nb = [14.4, 13.3, 8.5, 5.6]
x4 = np.arange(len(D4)); w4 = 0.35
ax.bar(x4 - w4/2, pb_nb, w4, color=C['fz'], edgecolor='white', lw=0.5, label='PBMC (peripheral blood)')
ax.bar(x4 + w4/2, pa_nb, w4, color=C['pau'], edgecolor='white', lw=0.5, label='Paul15 (bone marrow)')
for i, d in enumerate(D4):
    delta = pa_nb[i] - pb_nb[i]
    clr = '#EE3377' if delta > 0 else '#0077BB'
    ax.text(i, max(pb_nb[i], pa_nb[i])+0.6, f'{delta:+.1f}pp',
            ha='center', fontsize=7.5, fontweight='bold', color=clr)
    ax.text(i, 0.8, str(d), ha='center', fontsize=7, color='grey')
ax.set_xticks(x4); ax.set_xticklabels([f'$d={d}$' for d in D4])
ax.set_ylabel('NB-LR STRING Precision (%)'); ax.set_title('(d) Cross-Tissue Precision', fontweight='bold')
ax.legend(fontsize=7, framealpha=0.85); ax.yaxis.grid(True, alpha=0.12, color=C['grid'])
ax.set_ylim(0, 18)

plt.tight_layout(pad=2.5, h_pad=2.2, w_pad=2.2)
for fmt in ['pdf', 'png']:
    plt.savefig(os.path.join(FIG_DIR, f'fig4_validation.{fmt}'), dpi=300,
                bbox_inches='tight', facecolor=C['bg'], edgecolor='none')
plt.close()
print("Fig 4 done — 2x2 validation & robustness")
