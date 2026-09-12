"""
Fig 2: Comprehensive benchmark -- 3x3 panel composite.

DRAWN AT FINAL PRINT SIZE.  The canvas is set to the cas-sc body-text width
(466.6 pt = 16.46 cm), so the figure is placed at width=\textwidth with a
scale factor of 1.00 and every font size below is its true printed size
(5.2-7.0 pt, within the 5-7 pt band required for journal figures).
bbox_inches='tight' is deliberately NOT used, because it would silently change
the canvas size; the internal margins are set by tight_layout.

REPORTING CONVENTION
  Precision is a ratio, so a method that emits fewer edges can report a higher
  precision without recovering more validated edges.  Accordingly:
    * every precision bar carries its validated-STRING-edge count as a bar
      label, so the numerator is visible next to the ratio;
    * no bar is tinted to signal a winner;
    * panel (i) plots the raw precision gap and the gap in validated edges on
      twin axes -- the gap in validated edges is what a matched-edge-budget
      comparison measures, and it stays at or below zero throughout.
  Panels (a), (b), (d), (e) and (i) show raw, unmatched differences.  Their
  matched-edge-budget counterparts are reported in Table (eqcount) of the
  manuscript.
  Panel (h) reports the estimated dispersion as the end points of each curve in
  the legend rather than as a label on every marker: one d-group column is only
  ~37 px wide, so it cannot also hold four numeric labels, and the Paul15 curve
  (alpha-hat ~ 0.64) maps below the Paul15 bars, leaving no free space for a
  label.  Labelling every point tested 45.9%-57.6% collisions and four labels
  outside the axes frame.

DATA PROVENANCE
  (a,b,c) STRING precision, validated-edge counts and total edge counts, PBMC
          and Paul15 -> results/checkpoints/fair_{pbmc,paul15}_d{30,50,100,200}.json
  (d)     PBMC cell-type networks  -> results/fair_celltype.json
  (e)     Paul15 cell-type networks-> results/fair_celltype_paul15.json
          (scored against the MOUSE STRING v11 network, taxid 10090)
  (f)     Simulation, paired over seeds
          -> results/sim_multiseed_d*_n*.json  (sample-size sweep)
  (g)     NOTEARS zero-edge collapse -> results/_final_push_ckpt.json
  (h)     Raw gain vs overdispersion
  (i)     Threshold robustness     -> results/fair_supplementary.json
"""
import os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import scienceplots  # noqa: F401 -- enables plt.style.use('science')
import numpy as np

# thin white halo so a bar label stays legible if it grazes a neighbouring bar
STROKE = [pe.withStroke(linewidth=1.1, foreground='white')]

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
CKPT_DIR = os.path.join(RES_DIR, 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)

TEXTWIDTH_PT = 466.6               # cas-sc body text width, measured in the PDF
FIG_W_IN = 6.420                   # -> 466.6 pt after the backend's ~1% pad
FIG_H_IN = 6.190                   # figure + caption must fit one text page


def _j(name):
    with open(os.path.join(RES_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


def _ck(name):
    with open(os.path.join(CKPT_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


# ==== colourblind-safe palette (Paul Tol "bright", muted) ====
C = {
    'nb':   '#1565C0',   # scCausal, NB-LR with moment dispersion
    'fzn':  '#8E24AA',   # scCausal, fixed alpha = 1.0 control
    'fz':   '#E65100',   # Fisher's z
    'nt':   '#C62828',   # NOTEARS
    'pm':   '#78909C',   # permuted control
    'ink':  '#1F2933',   # all text
    'grid': '#E9E9E9',
    'grey': '#9AA5B1',
}

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 6.0,
    'axes.titlesize': 7.0,
    'axes.labelsize': 6.0,
    'xtick.labelsize': 5.8,
    'ytick.labelsize': 5.8,
    'legend.fontsize': 5.4,
    'text.color': C['ink'],
    'axes.labelcolor': C['ink'],
    'xtick.color': C['ink'],
    'ytick.color': C['ink'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#C4CDD5',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'grid.color': C['grid'],
    'grid.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

D = [30, 50, 100, 200]
INK = C['ink']
LAB_FS = 5.2          # bar-label size (printed pt)
NOTE_FS = 5.2

METHODS = [('NB_moment', 'scCausal (NB-LR, moment)', 'nb'),
           ('NB_fixed', r'scCausal (NB-LR, $\alpha{=}1.0$)', 'fzn'),
           ('Fisher_z', r"Fisher's $z$", 'fz')]

# ================================================================
# 1. LOAD ALL DATA (fair protocol only)
# ================================================================
pbmc = {d: _ck('fair_pbmc_d%d.json' % d) for d in D}
paul = {d: _ck('fair_paul15_d%d.json' % d) for d in D}

P, PE, PH = {}, {}, {}
for m, _, _ in METHODS:
    P[m] = [pbmc[d][m]['precision'] for d in D]
    PE[m] = [pbmc[d][m]['edges'] for d in D]
    PH[m] = [pbmc[d][m]['hits'] for d in D]

Q, QE, QH = {}, {}, {}
for m, _, _ in METHODS:
    Q[m] = [paul[d][m]['precision'] for d in D]
    QE[m] = [paul[d][m]['edges'] for d in D]
    QH[m] = [paul[d][m]['hits'] for d in D]

a_hat_pbmc = [pbmc[d]['meta']['alpha_hat'] for d in D]
a_hat_paul = [paul[d]['meta']['alpha_hat'] for d in D]

ct = _j('fair_celltype.json')['types']
ctp = _j('fair_celltype_paul15.json')['types']
GLOBAL_PBMC_D30 = pbmc[30]['NB_moment']['precision']


def _sim_sweep():
    rows = []
    for fn in sorted(os.listdir(RES_DIR)):
        if not (fn.startswith('sim_multiseed_d') and fn.endswith('.json')):
            continue
        b = _j(fn)['agg']
        pa = b['paired'][1]
        lo, hi = pa.get('boot_ci95', (pa['mean_diff'], pa['mean_diff']))
        rows.append({'d': b['d'], 'n': b['n'], 'seeds': b['seeds'],
                     'diff': pa['mean_diff'], 'lo': lo, 'hi': hi})
    return sorted(rows, key=lambda r: (r['n'], r['d']))


sim_sweep = _sim_sweep()
notears = _j('_final_push_ckpt.json')['notears_d30_50']['d_30']
supp = _j('fair_supplementary.json')

# ================================================================
# 2. FIGURE
# ================================================================
fig, axes = plt.subplots(3, 3, figsize=(FIG_W_IN, FIG_H_IN))
fig.patch.set_facecolor('white')

x = np.arange(len(D))
w = 0.26
wc = 0.34
hb = 0.36


def _style(ax):
    ax.tick_params(length=2, width=0.5)
    ax.yaxis.grid(True, which='major')
    ax.set_axisbelow(True)


# ---- (a) PBMC STRING precision -------------------------------------
ax = axes[0, 0]
for k, (m, lab, col) in enumerate(METHODS):
    ax.bar(x + (k - 1) * w, P[m], w, color=C[col], edgecolor='white', lw=0.4)
for i in range(len(D)):
    for k, (m, _, _) in enumerate(METHODS):
        ax.text(i + (k - 1) * w, P[m][i] + 0.3, '%d' % PH[m][i],
                ha='center', fontsize=LAB_FS, color=INK, path_effects=STROKE)
ax.set_xticks(x); ax.set_xticklabels(['$d{=}%d$' % d for d in D])
ax.set_ylabel('STRING precision (%)')
ax.set_title('(a) PBMC 3K', fontweight='bold', loc='left', pad=2.5)
ax.set_ylim(0, 18.6)
ax.text(0.98, 0.95, 'bar labels:\nvalidated edges', transform=ax.transAxes,
        ha='right', va='top', fontsize=NOTE_FS, color=C['grey'], linespacing=1.3)
_style(ax)

# ---- (b) Paul15 STRING precision -----------------------------------
ax = axes[0, 1]
for k, (m, lab, col) in enumerate(METHODS):
    ax.bar(x + (k - 1) * w, Q[m], w, color=C[col], edgecolor='white', lw=0.4)
for i in range(len(D)):
    for k, (m, _, _) in enumerate(METHODS):
        ax.text(i + (k - 1) * w, Q[m][i] + 0.3, '%d' % QH[m][i],
                ha='center', fontsize=LAB_FS, color=INK, path_effects=STROKE)
ax.set_xticks(x); ax.set_xticklabels(['$d{=}%d$' % d for d in D])
ax.set_ylabel('STRING precision (%)')
ax.set_title('(b) Paul15 (mouse STRING)', fontweight='bold', loc='left', pad=2.5)
ax.set_ylim(0, 18.6)
ax.text(0.98, 0.95, 'bar labels:\nvalidated edges', transform=ax.transAxes,
        ha='right', va='top', fontsize=NOTE_FS, color=C['grey'], linespacing=1.3)
_style(ax)

# ---- (c) Edge count (PBMC) -----------------------------------------
ax = axes[0, 2]
for k, (m, lab, col) in enumerate(METHODS):
    ax.bar(x + (k - 1) * w, PE[m], w, color=C[col], edgecolor='white', lw=0.4)
for i in range(len(D)):
    ax.text(i, PE['NB_moment'][i] + 130, '{:,}'.format(PE['NB_moment'][i]),
            ha='center', fontsize=LAB_FS, color=INK, path_effects=STROKE)
ax.set_xticks(x); ax.set_xticklabels(['$d{=}%d$' % d for d in D])
ax.set_ylabel('Recovered edges (#)')
ax.set_title('(c) Total edges (PBMC)', fontweight='bold', loc='left', pad=2.5)
ax.set_ylim(0, 7200)
ax.text(0.98, 0.95, 'labels: scCausal', transform=ax.transAxes, ha='right',
        va='top', fontsize=NOTE_FS, color=C['grey'])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, p: '0' if v == 0 else ('%.0fk' % (v / 1000.0))))
_style(ax)

# ---- (d) PBMC cell types, d = 30 -----------------------------------
ax = axes[1, 0]
ct_names = sorted([c for c in ct if 'd30' in ct[c]],
                  key=lambda c: ct[c]['d30']['NB_moment']['precision'], reverse=True)
nb_v = [ct[c]['d30']['NB_moment']['precision'] for c in ct_names]
fz_v = [ct[c]['d30']['Fisher_z']['precision'] for c in ct_names]
nb_h = [ct[c]['d30']['NB_moment']['hits'] for c in ct_names]
fz_h = [ct[c]['d30']['Fisher_z']['hits'] for c in ct_names]
xc = np.arange(len(ct_names))
ax.bar(xc - wc / 2, nb_v, wc, color=C['nb'], edgecolor='white', lw=0.4,
       label='scCausal (NB-LR)')
ax.bar(xc + wc / 2, fz_v, wc, color=C['fz'], edgecolor='white', lw=0.4,
       label=r"Fisher's $z$")
for i in range(len(ct_names)):
    ax.text(i - wc / 2, nb_v[i] + 0.6, '%d' % nb_h[i], ha='center',
            fontsize=LAB_FS, color=INK, path_effects=STROKE)
    ax.text(i + wc / 2, fz_v[i] + 0.6, '%d' % fz_h[i], ha='center',
            fontsize=LAB_FS, color=INK, path_effects=STROKE)
ax.axhline(y=GLOBAL_PBMC_D30, color=C['grey'], linestyle='--', lw=0.8)
ax.text(1.5, GLOBAL_PBMC_D30 - 2.9, 'pooled %.2f%%' % GLOBAL_PBMC_D30,
        fontsize=NOTE_FS, color=C['grey'], ha='center', path_effects=STROKE)
short = [n.replace('+ ', '+').replace('Monocyte', 'Mono') for n in ct_names]
ax.set_xticks(xc); ax.set_xticklabels(short, fontsize=5.4, rotation=15, ha='right')
ax.set_ylabel('STRING precision (%)')
ax.set_title(r'(d) PBMC cell types ($d{=}30$)', fontweight='bold', loc='left', pad=2.5)
ax.legend(loc='upper right', frameon=False, handlelength=1.0, handletextpad=0.35,
          borderpad=0.1, labelspacing=0.28, fontsize=5.2)
ax.set_ylim(0, 38)
_style(ax)

# ---- (e) Paul15 cell types, d = 30 (horizontal) --------------------
ax = axes[1, 1]
LIN = {'10GMP': 'GMP', '9GMP': 'GMP', '12Baso': 'Baso', '13Baso': 'Baso',
       '14Mo': 'Mono', '15Mo': 'Mono', '16Neu': 'Neut', '2Ery': 'Ery',
       '3Ery': 'Ery', '4Ery': 'Ery', '5Ery': 'Ery', '6Ery': 'Ery',
       '7MEP': 'MEP', '8Mk': 'Mk'}
cp_names = sorted(list(ctp.keys()),
                  key=lambda c: ctp[c]['d30']['NB_moment']['precision'])
cp_nb = [ctp[c]['d30']['NB_moment']['precision'] for c in cp_names]
cp_fz = [ctp[c]['d30']['Fisher_z']['precision'] for c in cp_names]
cp_nh = [ctp[c]['d30']['NB_moment']['hits'] for c in cp_names]
cp_fh = [ctp[c]['d30']['Fisher_z']['hits'] for c in cp_names]
yp = np.arange(len(cp_names))
ax.barh(yp - hb / 2, cp_nb, hb, color=C['nb'], edgecolor='white', lw=0.4,
        label='scCausal (NB-LR)')
ax.barh(yp + hb / 2, cp_fz, hb, color=C['fz'], edgecolor='white', lw=0.4,
        label=r"Fisher's $z$")
for i in range(len(cp_names)):
    ax.text(cp_nb[i] + 0.45, i, '%d' % cp_nh[i], va='center',
            fontsize=5.3, color=INK, path_effects=STROKE)
ax.set_yticks(yp)
ax.set_yticklabels(['%s (%s)' % (c, LIN.get(c, '?')) for c in cp_names], fontsize=5.3)
ax.set_xlabel('STRING precision (%)')
ax.set_title(r'(e) Paul15 cell types ($d{=}30$)', fontweight='bold', loc='left', pad=2.5)
ax.legend(loc='upper right', frameon=False, handlelength=1.0, handletextpad=0.35,
          borderpad=0.1, labelspacing=0.28, fontsize=5.2)
ax.set_xlim(0, 30)
ax.xaxis.grid(True, which='major'); ax.set_axisbelow(True)
ax.invert_yaxis()

# ---- (f) Simulation vs sample size ---------------------------------
ax = axes[1, 2]
s_n = [r['n'] for r in sim_sweep]
s_d = [r['diff'] for r in sim_sweep]
ax.axhline(0, color=C['grey'], lw=0.8, zorder=1)
ax.errorbar(s_n, s_d,
            yerr=[[r['diff'] - r['lo'] for r in sim_sweep],
                  [r['hi'] - r['diff'] for r in sim_sweep]],
            fmt='none', ecolor='#8A8A8A', elinewidth=0.9, capsize=2.0, zorder=2)
ax.scatter(s_n, s_d, s=24, zorder=3, edgecolor='white', linewidth=0.7,
           c=[C['nb'] if r['diff'] > 0 else C['fz'] for r in sim_sweep])
ax.set_xscale('log')
ax.set_xticks([300, 500, 800, 1500])
ax.set_xticklabels(['300', '500', '800', '1500'])
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel('Cells ($n$, log scale)')
ax.set_ylabel(r'$\Delta$F1  (NB-LR $-$ Fisher)')
ax.set_title('(f) Simulation vs sample size', fontweight='bold', loc='left', pad=2.5)
ax.text(0.5, 0.05, r'sign follows $n$, not $d$', transform=ax.transAxes,
        ha='center', fontsize=NOTE_FS, color=C['grey'])
ax.set_ylim(-0.042, 0.040)
_style(ax)

# ---- (g) Method comparison, d = 30, PBMC ---------------------------
ax = axes[2, 0]
m_lab = ['NB-LR mom.', 'NB-LR ' + r'$\alpha{=}1$', 'Fisher', 'NOTEARS', 'Permuted']
m_val = [P['NB_moment'][0], P['NB_fixed'][0], P['Fisher_z'][0], 0.0, 0.0]
m_edg = [PE['NB_moment'][0], PE['NB_fixed'][0], PE['Fisher_z'][0], notears['edges'], 0]
m_col = [C['nb'], C['fzn'], C['fz'], C['nt'], C['pm']]
bars = ax.bar(m_lab, m_val, color=m_col, edgecolor='white', lw=0.5)
for k, (b, v, ne) in enumerate(zip(bars, m_val, m_edg)):
    cx = b.get_x() + b.get_width() / 2
    if v > 0:
        ax.text(cx, v + 0.4, '%.2f%%\n%d' % (v, ne), ha='center',
                fontsize=LAB_FS, color=INK, linespacing=1.3, path_effects=STROKE)
    else:
        ax.text(cx, 0.45, '0\nedges', ha='center', fontsize=LAB_FS,
                color=m_col[k], fontweight='bold', linespacing=1.3,
                path_effects=STROKE)
ax.set_ylabel('STRING precision (%)')
ax.set_title(r'(g) Method comparison ($d{=}30$)', fontweight='bold', loc='left', pad=2.5)
ax.set_ylim(0, 18.6)
ax.tick_params(axis='x', labelsize=5.2, pad=1.0)
for t in ax.get_xticklabels():
    t.set_rotation(45)
    t.set_ha('right')
    t.set_rotation_mode('anchor')
ax.text(0.98, 0.95, '2nd line:\nedges', transform=ax.transAxes, ha='right',
        va='top', fontsize=NOTE_FS, color=C['grey'], linespacing=1.3)
_style(ax)

# ---- (h) Raw gain vs overdispersion --------------------------------
ax = axes[2, 1]
d_pbmc = [P['NB_moment'][i] - P['Fisher_z'][i] for i in range(len(D))]
d_paul = [Q['NB_moment'][i] - Q['Fisher_z'][i] for i in range(len(D))]
wd = 0.34
ax.bar(x - wd / 2, d_pbmc, wd, color=C['nb'], edgecolor='white', lw=0.4,
       label=r'$\Delta$ precision, PBMC')
ax.bar(x + wd / 2, d_paul, wd, color=C['fz'], edgecolor='white', lw=0.4,
       label=r'$\Delta$ precision, Paul15')
for i in range(len(D)):
    ax.text(i - wd / 2, d_pbmc[i] + 0.07, '%+.2f' % d_pbmc[i], ha='center',
            fontsize=LAB_FS, color=INK, path_effects=STROKE)
    ax.text(i + wd / 2, d_paul[i] + 0.07, '%+.2f' % d_paul[i], ha='center',
            fontsize=LAB_FS, color=INK, path_effects=STROKE)
ax.set_xticks(x); ax.set_xticklabels(['$d{=}%d$' % d for d in D])
ax.set_ylabel(r'$\Delta$ precision (pp)')
ax.set_title('(h) Raw gain vs overdispersion', fontweight='bold', loc='left', pad=2.5)
ax.set_ylim(0, 3.3)
ax2 = ax.twinx()
# The dispersion is summarised by its end points in the legend instead of being
# annotated at every d.  A single d-group column is only ~37 px wide and already
# carries two bar labels; four numeric labels per group do not fit, and at d=30
# the PBMC alpha-hat (2.40) lands within 0.01 of the Paul15 bar top (0.74 on the
# left scale), so any placement of the two labels collides.  The Paul15 curve is
# worse: alpha-hat ~ 0.64 maps to 0.20 on the left scale, i.e. inside the Paul15
# bars, leaving no free space above, below or beside its markers.  Measured with
# a bounding-box audit, the previous per-point annotations produced two text
# collisions (45.9% and 57.6% coverage) and four labels hanging outside the axes
# frame.  The end points below are computed from the same arrays, so the printed
# range cannot drift from the plotted curve.
# Two decimals on both series, so the printed range matches the Results sentence
# verbatim: "alpha-hat from 2.40 at d=30 to 7.78 at d=200" (PBMC).
ax2.plot(x, a_hat_pbmc, 'o--', color=C['nb'], lw=1.2, ms=3.0, mfc='white',
         mew=0.9, label=r'$\hat{\alpha}$, PBMC (%.2f$\to$%.2f)'
                        % (a_hat_pbmc[0], a_hat_pbmc[-1]))
ax2.plot(x, a_hat_paul, 's--', color=C['fz'], lw=1.2, ms=3.0, mfc='white',
         mew=0.9, label=r'$\hat{\alpha}$, Paul15 (%.2f$\to$%.2f)'
                        % (a_hat_paul[0], a_hat_paul[-1]))
ax2.tick_params(axis='x', labelbottom=False)   # x ticks belong to the left axes
ax2.set_ylabel(r'$\hat{\alpha}$')
ax2.set_ylim(0, 10.6)
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color('#C4CDD5')
ax2.tick_params(axis='y', length=2, width=0.5)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False, handlelength=1.2,
          handletextpad=0.35, borderpad=0.1, labelspacing=0.3, fontsize=5.2)
_style(ax)

# ---- (i) Threshold robustness: raw precision vs validated edges ----
ax = axes[2, 2]
a_keys = sorted([k for k in supp if k.startswith('alpha_')], key=lambda k: float(k[6:]))
t_keys = sorted([k for k in supp if k.startswith('tau_')], key=lambda k: float(k[4:]))
a_x = [float(k[6:]) for k in a_keys]
t_x = [float(k[4:]) for k in t_keys]
a_y = [supp[k]['nb_precision'] - supp[k]['fz_precision'] for k in a_keys]
t_y = [supp[k]['nb_precision'] - supp[k]['fz_precision'] for k in t_keys]
a_dh = [supp[k]['nb_hits'] - supp[k]['fz_hits'] for k in a_keys]
t_dh = [supp[k]['nb_hits'] - supp[k]['fz_hits'] for k in t_keys]

ax.axhline(0, color=C['grey'], lw=0.7, ls=':')
ax.plot(a_x, a_y, 'o-', color=C['nb'], lw=1.3, ms=3.2, zorder=3)
ax.plot(t_x, t_y, 's--', color=C['fz'], lw=1.3, ms=3.2, zorder=3)
ax.set_xlabel('Threshold value')
ax.set_ylabel(r'$\Delta$ precision (pp)')
ax.set_ylim(-2.15, 2.45)
ax.set_title(r'(i) Threshold robustness ($d{=}30$)', fontweight='bold', loc='left',
             pad=2.5)
ax3 = ax.twinx()
ax3.plot(a_x, a_dh, 'o-', color=C['nb'], lw=1.0, ms=2.4, alpha=0.40, mfc='none')
ax3.plot(t_x, t_dh, 's--', color=C['fz'], lw=1.0, ms=2.4, alpha=0.40, mfc='none')
ax3.set_ylabel(r'$\Delta$ validated edges', labelpad=1.0)
ax3.set_ylim(-2.15, 2.45)
ax3.set_yticks([-2, -1, 0, 1, 2])
ax3.spines['right'].set_visible(True)
ax3.spines['right'].set_color('#C4CDD5')
ax3.tick_params(axis='y', length=2, width=0.5)
handles = [plt.Line2D([], [], color=C['nb'], marker='o', lw=1.3, ms=3.2,
                      label=r'$\Delta$ precision, $\alpha$'),
           plt.Line2D([], [], color=C['fz'], marker='s', lw=1.3, ms=3.2, ls='--',
                      label=r'$\Delta$ precision, $\tau$'),
           plt.Line2D([], [], color=C['grey'], marker='o', lw=1.0, ms=2.4, alpha=0.45,
                      mfc='none', label=r'$\Delta$ validated edges (faded)')]
ax.legend(handles=handles, loc='lower center', frameon=False, handlelength=1.3,
          handletextpad=0.35, borderpad=0.1, labelspacing=0.3, fontsize=5.3,
          bbox_to_anchor=(0.5, -0.02))
_style(ax)

# ---- shared legend for the three tests -----------------------------
fig.legend(handles=[Patch(facecolor=C['nb'], label='scCausal (NB-LR, moment)'),
                    Patch(facecolor=C['fzn'], label=r'scCausal (NB-LR, $\alpha{=}1.0$)'),
                    Patch(facecolor=C['fz'], label=r"Fisher's $z$")],
           loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
           frameon=False, fontsize=6.0, handlelength=1.1, handletextpad=0.4,
           columnspacing=1.6)

# ================================================================
# 3. SAVE
# ================================================================
fig.tight_layout(pad=0.28, h_pad=1.15, w_pad=0.8, rect=[0, 0, 1, 0.955])
for fmt in ['pdf', 'png']:
    plt.savefig(os.path.join(FIG_DIR, 'fig2_benchmark.%s' % fmt), dpi=400,
                facecolor='white', edgecolor='none')
plt.close()

# ---- provenance echo, so every printed number can be checked ----
print('Fig 2 done (canvas %.3f x %.3f in; target print scale %.4f)'
      % (FIG_W_IN, FIG_H_IN, TEXTWIDTH_PT / (FIG_W_IN * 72.0)))
print('  (a) PBMC prec  NB_moment/NB_fixed/Fisher:', P['NB_moment'], P['NB_fixed'], P['Fisher_z'])
print('      PBMC hits                        :', PH['NB_moment'], PH['NB_fixed'], PH['Fisher_z'])
print('      PBMC edges                       :', PE['NB_moment'], PE['NB_fixed'], PE['Fisher_z'])
print('  (b) Paul15 prec NB_moment/Fisher    :', Q['NB_moment'], Q['Fisher_z'])
print('      Paul15 hits NB_moment/Fisher    :', QH['NB_moment'], QH['Fisher_z'])
print('      Paul15 edges NB_moment/Fisher   :', QE['NB_moment'], QE['Fisher_z'])
print('  (d) PBMC ct hits  NB / Fisher       :', nb_h, fz_h)
print('  (e) Paul15 ct hits NB / Fisher      :', cp_nh, cp_fh)
print('  (g) d=30 values / edges             :', m_val, m_edg)
print('  (h) delta PBMC / Paul15             :', [round(v, 2) for v in d_pbmc],
      [round(v, 2) for v in d_paul])
print('      alpha-hat PBMC / Paul15         :', a_hat_pbmc, a_hat_paul)
print('  (i) alpha d-prec / d-hits           :', [round(v, 2) for v in a_y], a_dh)
print('      tau   d-prec / d-hits           :', [round(v, 2) for v in t_y], t_dh)
for fmt in ['pdf', 'png']:
    p = os.path.join(FIG_DIR, 'fig2_benchmark.%s' % fmt)
    print('  file %-8s %9d bytes' % (fmt, os.path.getsize(p)))
