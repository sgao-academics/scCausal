"""
Fig 5: validation and robustness -- 2 x 2 panel composite.

DESIGN CONTRACT (identical to fig1_pipeline.tex and gen_fig3.py)
  * sans-serif Helvetica/Arial throughout
  * body text 5-7 pt AT FINAL PRINT SIZE -> the canvas is drawn at 1:1, i.e.
    figsize x 72 pt equals the printed size, so savefig must NOT be given
    bbox_inches (that would let the canvas drift away from the target width)
  * no coloured text: colour lives only in fills, edges and markers
  * flat design: hairline spines, no drop shadow, no in-figure title
    (the descriptive title lives in the caption)
  * colour semantics are FIXED across the whole figure:
        blue   = scCausal / NB-LR
        orange = Fisher's z
        teal   = NB-LR with the library-size GLM offset
        pink   = Paul15 (cross-tissue)
        grey   = background / reference
    marker shape, not colour, encodes the dimension in panel (b)

DATA PROVENANCE (nothing is hardcoded; every plotted number is recomputed from
the fair-protocol result files -- the same run that produces Table 1)
  (a) Reactome pathway co-membership of the STRING-validated gene set
      -> results/fair_downstream.json              (key: reactome.by_d)
  (b) STRING precision vs combined-score threshold (400-900)
      -> results/fair_downstream.json              (key: threshold)
  (c) library-size GLM offset ablation
      -> results/fair_supplementary.json           (keys: offset_d*)
         baseline -> results/checkpoints/fair_pbmc_d*.json (NB_moment)
  (d) cross-tissue NB-LR precision, PBMC vs Paul15
      -> results/checkpoints/fair_pbmc_d*.json and fair_paul15_d*.json
"""
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
CKPT_DIR = os.path.join(RES_DIR, 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)

# ---- canvas: printed at \textwidth of cas-sc = 466.60 pt -> scale exactly 1.0
FIG_W_IN = 466.60 / 72.0
FIG_H_IN = 5.10

C = {
    'nb':   '#0072B2',   # scCausal / NB-LR
    'fz':   '#EE7733',   # Fisher's z
    'alt':  '#009988',   # NB-LR with the library-size offset
    'pau':  '#EE3377',   # Paul15 (cross-tissue)
    'ref':  '#BBBBBB',   # background / reference
    'ink':  '#1F2933',   # every piece of text
    'grid': '#DDDDDD',
}

TITLE_FS = 7.5
AXLAB_FS = 7.0
TICK_FS = 6.5
LEG_FS = 6.0
DATA_FS = 6.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': TICK_FS,
    'axes.linewidth': 0.7,
    'axes.edgecolor': C['ink'],
    'axes.labelcolor': C['ink'],
    'axes.labelsize': AXLAB_FS,
    'axes.titlesize': TITLE_FS,
    'axes.titlecolor': C['ink'],
    'axes.titlepad': 3.0,
    'axes.axisbelow': True,
    'xtick.labelsize': TICK_FS,
    'ytick.labelsize': TICK_FS,
    'xtick.color': C['ink'],
    'ytick.color': C['ink'],
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'legend.fontsize': LEG_FS,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#DDDDDD',
    'legend.facecolor': 'white',
    'legend.borderpad': 0.35,
    'legend.labelspacing': 0.30,
    'legend.handlelength': 1.7,
    'legend.handletextpad': 0.45,
    'legend.columnspacing': 1.0,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.dpi': 300,
})


def _j(name):
    with open(os.path.join(RES_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


def _ck(name):
    with open(os.path.join(CKPT_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


def wilson(k, n, z=1.959964):
    """95% Wilson score interval, in per cent."""
    if n == 0:
        return 0.0, 0.0
    p = float(k) / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z / d * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return 100.0 * (c - h), 100.0 * (c + h)


D = [30, 50, 100, 200]
ds = _j('fair_downstream.json')
supp = _j('fair_supplementary.json')
pbmc = {d: _ck('fair_pbmc_d%d.json' % d) for d in D}
paul = {d: _ck('fair_paul15_d%d.json' % d) for d in D}

# all four panels are driven by (hits, edges) counts, never by stored decimals
r30, r50 = ds['reactome']['by_d']['30'], ds['reactome']['by_d']['50']

# === FIGURE ==================================================================
fig, axes = plt.subplots(2, 2, figsize=(FIG_W_IN, FIG_H_IN))

# ---- (a) Reactome pathway co-membership -------------------------------------
ax = axes[0, 0]
recs = [r30, r50]
n_edge = [r['n_edges'] for r in recs]
k_edge = [r['co_member_edges'] for r in recs]
n_bg = [r['n_bg_pairs'] for r in recs]
k_bg = [int(round(r['bg_pct'] / 100.0 * r['n_bg_pairs'])) for r in recs]
p_edge = [100.0 * k_edge[i] / n_edge[i] for i in range(2)]
p_bg = [100.0 * k_bg[i] / n_bg[i] for i in range(2)]
e_edge = [wilson(k_edge[i], n_edge[i]) for i in range(2)]
e_bg = [wilson(k_bg[i], n_bg[i]) for i in range(2)]

xg = np.arange(2)
wg = 0.34
for i, (px, err, col, lab) in enumerate([
        (p_edge, e_edge, C['nb'], 'STRING-validated edges'),
        (p_bg, e_bg, C['ref'], 'Background (all gene pairs)')]):
    off = -wg / 2 if i == 0 else wg / 2
    lo = [px[j] - err[j][0] for j in range(2)]
    hi = [err[j][1] - px[j] for j in range(2)]
    ax.bar(xg + off, px, wg, color=col, edgecolor='white', lw=0.5,
           yerr=[lo, hi], label=lab,
           error_kw=dict(ecolor=C['ink'], lw=0.7, capsize=1.6, capthick=0.7))
for j in range(2):
    ax.text(xg[j] - wg / 2, e_edge[j][1] + 1.2, '%.1f%%' % p_edge[j],
            ha='center', va='bottom', fontsize=DATA_FS, color=C['ink'])
    ax.text(xg[j] + wg / 2, e_bg[j][1] + 1.2, '%.1f%%' % p_bg[j],
            ha='center', va='bottom', fontsize=DATA_FS, color=C['ink'])
ax.set_xticks(xg)
# the odds ratio and Fisher p are per-group facts, so they sit in the group's
# own tick label rather than in a floating annotation that the d=30 whisker
# (whose Wilson top is 95.3%) would collide with
ax.set_xticklabels(['$d$=%d\n%d edges, %d genes\nOR $=$ %.2f, $p$ $=$ %.4f'
                    % (d, r['n_edges'], r['n_genes'],
                       r['odds_ratio'], r['fisher_p'])
                    for d, r in zip([30, 50], recs)])
for t in ax.get_xticklabels():
    t.set_linespacing(1.30)
ax.set_ylim(0, 104)
ax.set_ylabel('Pathway co-membership of edge (%)')
ax.set_title('(a) Reactome co-membership (PBMC)')
ax.yaxis.grid(True, lw=0.5, color=C['grid'])
ax.legend(loc='upper right')

# ---- (b) STRING precision vs combined-score threshold -----------------------
ax = axes[0, 1]
thr_axis = [400, 500, 600, 700, 800, 900]
STYLE = {30: ('o', 3.4), 50: ('s', 3.2), 100: ('^', 3.6)}
for d in [30, 50, 100]:
    rec = ds['threshold'][str(d)]
    mk, ms = STYLE[d]
    nb = [rec['NB_moment'][str(t)]['precision'] for t in thr_axis]
    fz = [rec['Fisher_z'][str(t)]['precision'] for t in thr_axis]
    ax.plot(thr_axis, nb, '-', color=C['nb'], lw=1.1, marker=mk, ms=ms,
            mfc=C['nb'], mec=C['nb'], mew=0.4)
    ax.plot(thr_axis, fz, '--', color=C['fz'], lw=1.0, marker=mk, ms=ms,
            mfc='white', mec=C['fz'], mew=0.7)
ax.set_xlabel('STRING combined-score threshold')
ax.set_ylabel('Validation precision (%)')
ax.set_title('(b) STRING threshold sensitivity')
ax.set_xlim(950, 350)
ax.set_ylim(0, 29)
ax.yaxis.grid(True, lw=0.5, color=C['grid'])
leg_m = ax.legend(handles=[
    Line2D([], [], color=C['nb'], lw=1.1, label='NB-LR'),
    Line2D([], [], color=C['fz'], lw=1.0, ls='--', label="Fisher's $z$")],
    loc='upper left')
ax.add_artist(leg_m)
ax.legend(handles=[Line2D([], [], color=C['ink'], marker=STYLE[d][0], ls='none',
                          ms=STYLE[d][1], label='$d$=%d' % d)
                   for d in [30, 50, 100]],
          loc='lower right')

# ---- (c) library-size GLM offset ablation ----------------------------------
ax = axes[1, 0]
no_off = [pbmc[d]['NB_moment']['precision'] for d in D]
with_off = [supp['offset_d%d' % d]['precision'] for d in D]
xl = np.arange(len(D))
wl = 0.36
ax.bar(xl - wl / 2, no_off, wl, color=C['nb'], edgecolor='white', lw=0.5,
       label='NB-LR, no offset')
ax.bar(xl + wl / 2, with_off, wl, color=C['alt'], edgecolor='white', lw=0.5,
       label='NB-LR, + log(library size) offset')
for i in range(len(D)):
    dl = with_off[i] - no_off[i]
    ax.text(i, max(no_off[i], with_off[i]) + 0.4, '%+.2f pp' % dl,
            ha='center', va='bottom', fontsize=DATA_FS, color=C['ink'])
ax.set_xticks(xl)
ax.set_xticklabels(['$d$=%d' % d for d in D])
ax.set_ylim(0, 17.5)
ax.set_ylabel('STRING precision (%)')
ax.set_title('(c) Library-size offset (PBMC)')
ax.yaxis.grid(True, lw=0.5, color=C['grid'])
ax.legend(loc='upper right')

# ---- (d) cross-tissue NB-LR precision, PBMC vs Paul15 ----------------------
ax = axes[1, 1]
pb = [pbmc[d]['NB_moment']['precision'] for d in D]
pa = [paul[d]['NB_moment']['precision'] for d in D]
ax.bar(xl - wl / 2, pb, wl, color=C['nb'], edgecolor='white', lw=0.5,
       label='PBMC (peripheral blood)')
ax.bar(xl + wl / 2, pa, wl, color=C['pau'], edgecolor='white', lw=0.5,
       label='Paul15 (bone marrow)')
for i in range(len(D)):
    dd = pa[i] - pb[i]
    ax.text(i, max(pb[i], pa[i]) + 0.4, '%+.2f pp' % dd,
            ha='center', va='bottom', fontsize=DATA_FS, color=C['ink'])
ax.set_xticks(xl)
ax.set_xticklabels(['$d$=%d' % d for d in D])
ax.set_ylim(0, 17.5)
ax.set_ylabel('NB-LR STRING precision (%)')
ax.set_title('(d) Cross-tissue precision')
ax.yaxis.grid(True, lw=0.5, color=C['grid'])
ax.legend(loc='upper right')

fig.tight_layout(pad=0.40, h_pad=0.90, w_pad=1.10)

for fmt in ['pdf', 'png']:
    out = os.path.join(FIG_DIR, 'fig5_validation.%s' % fmt)
    fig.savefig(out, dpi=300, facecolor='white', edgecolor='none')
plt.close(fig)

# === SOURCE VALUES ===========================================================
print('Fig 5 done. Source values:')
print('  (a) d=30: %d/%d = %.3f%%  bg %d/%d = %.3f%%  OR=%.3f p=%.6g'
      % (k_edge[0], n_edge[0], p_edge[0], k_bg[0], n_bg[0], p_bg[0],
         r30['odds_ratio'], r30['fisher_p']))
print('      d=50: %d/%d = %.3f%%  bg %d/%d = %.3f%%  OR=%.3f p=%.6g'
      % (k_edge[1], n_edge[1], p_edge[1], k_bg[1], n_bg[1], p_bg[1],
         r50['odds_ratio'], r50['fisher_p']))
print('      Wilson 95%% edges d30 [%.1f, %.1f]  d50 [%.1f, %.1f]'
      % (e_edge[0][0], e_edge[0][1], e_edge[1][0], e_edge[1][1]))
print('      Wilson 95%% bg    d30 [%.1f, %.1f]  d50 [%.1f, %.1f]'
      % (e_bg[0][0], e_bg[0][1], e_bg[1][0], e_bg[1][1]))
print('  (b) thresholds:', thr_axis)
for d in [30, 50, 100]:
    rec = ds['threshold'][str(d)]
    print('      d=%-3d NB: %s' % (d, [rec['NB_moment'][str(t)]['precision'] for t in thr_axis]))
    print('      d=%-3d Fz: %s' % (d, [rec['Fisher_z'][str(t)]['precision'] for t in thr_axis]))
print('  (c) no offset  :', no_off)
print('      with offset:', with_off)
print('      deltas     :', ['%+.2f' % (with_off[i] - no_off[i]) for i in range(4)])
print('  (d) PBMC       :', pb)
print('      Paul15     :', pa)
print('      deltas     :', ['%+.2f' % (pa[i] - pb[i]) for i in range(4)])
for fmt in ['pdf', 'png']:
    p = os.path.join(FIG_DIR, 'fig5_validation.%s' % fmt)
    print('  file %-4s %8d bytes' % (fmt, os.path.getsize(p)))
