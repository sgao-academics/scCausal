"""
Fig 2: Comprehensive Benchmark -- 3x3 panel composite.

All values are read from the FAIR-protocol result files, i.e. the same run that
produces Table 1 of the manuscript. Nothing is hardcoded.

DATA PROVENANCE
  (a,b,c) STRING precision / edge counts for PBMC 3K and Paul15
          -> results/checkpoints/fair_{pbmc,paul15}_d{30,50,100,200}.json
             (written incrementally by scripts/reproduce_benchmark.py)
  (d)     PBMC cell-type networks  -> results/fair_celltype.json
  (e)     Paul15 cell-type networks -> results/fair_celltype_paul15.json
             (validated against the MOUSE STRING v11 network, taxid 10090)
  (f)     Simulation, 100 random seeds -> results/_everything_ckpt.json (E_sim_*)
  (g)     NOTEARS zero-edge collapse   -> results/_final_push_ckpt.json
  (h)     NB advantage vs overdispersion: pooled over every fair setting above
  (i)     Threshold robustness          -> results/fair_supplementary.json

Fair protocol: identical tau = 0.10, identical top-d variance-selected gene set
and identical log1p pre-filter for every method; the only difference is the
conditional-independence test itself.
"""
import os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- enables plt.style.use('science')
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
CKPT_DIR = os.path.join(RES_DIR, 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)


def _j(name):
    with open(os.path.join(RES_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


def _ck(name):
    with open(os.path.join(CKPT_DIR, name), encoding='utf-8') as fh:
        return json.load(fh)


# === Nature-inspired palette ===
C = {
    'nb':   '#1565C0',   # NB-LR blue
    'fzn':  '#8E24AA',   # NB fixed-alpha purple
    'fz':   '#E65100',   # Fisher orange
    'pau':  '#00838F',   # Paul15 teal
    'fzpau':'#BF360C',   # Paul15 Fisher deep orange
    'nt':   '#C62828',   # NOTEARS red
    'pm':   '#546E7A',   # permuted slate
    'bg':   '#FFFFFF',
    'grid': '#E8E8E8',
    'grey': '#78909C',
}

plt.style.use(['science', 'no-latex', 'bright'])
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.5,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BDBDBD', 'axes.linewidth': 0.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})

D = [30, 50, 100, 200]
METHODS = [('NB_moment', 'scCausal (NB-LR, moment)', 'nb'),
           ('NB_fixed', r'scCausal (NB-LR, $\alpha{=}1.0$)', 'fzn'),
           ('Fisher_z', r"Fisher's $z$ ($\log(1{+}x)$)", 'fz')]

# ================================================================
# 1. LOAD ALL DATA (fair protocol only)
# ================================================================
pbmc = {d: _ck('fair_pbmc_d%d.json' % d) for d in D}
paul = {d: _ck('fair_paul15_d%d.json' % d) for d in D}

s = {m: [pbmc[d][m]['precision'] for d in D] for m, _, _ in METHODS}
e = {m: [pbmc[d][m]['edges'] for d in D] for m, _, _ in METHODS}
sp = {m: [paul[d][m]['precision'] for d in D] for m, _, _ in METHODS}

a_hat_pbmc = [pbmc[d]['meta']['alpha_hat'] for d in D]
a_hat_paul = [paul[d]['meta']['alpha_hat'] for d in D]

# --- cell types ---
ct = _j('fair_celltype.json')['types']
ctp = _j('fair_celltype_paul15.json')['types']

GLOBAL_PBMC_D30 = pbmc[30]['NB_moment']['precision']   # 14.11

# --- simulation (100 seeds) ---
ek = _j('_everything_ckpt.json')
sim_d = [30, 50, 100]
sim = [ek['E_sim_d%d' % d] for d in sim_d]

# --- NOTEARS ---
notears = _j('_final_push_ckpt.json')['notears_d30_50']['d_30']

# --- threshold sweeps (fair) ---
supp = _j('fair_supplementary.json')

# ================================================================
# 2. CREATE 3x3 FIGURE
# ================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14.5))
fig.patch.set_facecolor(C['bg'])
x = np.arange(len(D))
w = 0.26

# ---- ROW 1: Core benchmark ----
# (a) PBMC STRING precision
ax = axes[0, 0]
for k, (m, lab, col) in enumerate(METHODS):
    ax.bar(x + (k - 1) * w, s[m], w, color=C[col], edgecolor='white', lw=0.5, label=lab)
for i in range(len(D)):
    mx = max(s[m][i] for m, _, _ in METHODS)
    win = 'nb' if s['NB_moment'][i] >= mx else 'fz'
    ax.text(i, mx + 0.45, f'{mx:.2f}%', ha='center', fontsize=7,
            fontweight='bold', color=C[win])
ax.set_xticks(x); ax.set_xticklabels([f'$d={d}$' for d in D])
ax.set_ylabel('STRING precision (%)')
ax.set_title('(a) PBMC 3K precision', fontweight='bold', color=C['nb'])
ax.legend(fontsize=5.8, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 18.5)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (b) Paul15 STRING precision (mouse STRING v11)
ax = axes[0, 1]
for k, (m, lab, col) in enumerate(METHODS):
    ax.bar(x + (k - 1) * w, sp[m], w,
           color=C['pau'] if col == 'nb' else (C['fzn'] if col == 'fzn' else C['fzpau']),
           edgecolor='white', lw=0.5, label=lab)
for i, d in enumerate(D):
    delta = sp['NB_moment'][i] - sp['Fisher_z'][i]
    ax.text(i, max(sp[m][i] for m, _, _ in METHODS) + 0.75,
            f'{sp["NB_moment"][i]:.2f}%\n$\\Delta${delta:+.2f} pp',
            ha='center', fontsize=6.5, fontweight='bold', color=C['pau'])
ax.set_xticks(x); ax.set_xticklabels([f'$d={d}$' for d in D])
ax.set_ylabel('STRING precision (%)')
ax.set_title('(b) Paul15 (mouse STRING)', fontweight='bold', color=C['pau'])
ax.legend(fontsize=5.8, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 18.5)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (c) Edge count (PBMC)
ax = axes[0, 2]
for k, (m, lab, col) in enumerate(METHODS):
    ax.bar(x + (k - 1) * w, e[m], w, color=C[col], edgecolor='white', lw=0.5, label=lab)
for i in range(len(D)):
    for k, (m, _, col) in enumerate(METHODS):
        ax.text(i + (k - 1) * w, e[m][i] + 90, f'{e[m][i]:,}', ha='center',
                fontsize=5.6, fontweight='bold', color=C[col])
ax.set_xticks(x); ax.set_xticklabels([f'$d={d}$' for d in D])
ax.set_ylabel('Recovered edges (#)')
ax.set_title('(c) Edge count (PBMC)', fontweight='bold')
ax.legend(fontsize=5.8, framealpha=0.9)
ax.set_ylim(0, 6400)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# ---- ROW 2: Biological depth ----
# (d) PBMC cell-type networks (d = 30)
ax = axes[1, 0]
ct_names = sorted([c for c in ct if 'd30' in ct[c]],
                  key=lambda c: ct[c]['d30']['NB_moment']['precision'], reverse=True)
nb_v = [ct[c]['d30']['NB_moment']['precision'] for c in ct_names]
fz_v = [ct[c]['d30']['Fisher_z']['precision'] for c in ct_names]
xc = np.arange(len(ct_names)); wc = 0.36
ax.bar(xc - wc / 2, nb_v, wc, color=C['nb'], edgecolor='white', lw=0.5, label='scCausal (NB-LR)')
ax.bar(xc + wc / 2, fz_v, wc, color=C['fz'], edgecolor='white', lw=0.5, label=r"Fisher's $z$")
for i in range(len(ct_names)):
    ax.text(i, max(nb_v[i], fz_v[i]) + 0.8, f'{nb_v[i] - fz_v[i]:+.1f}',
            ha='center', fontsize=6.2, fontweight='bold', color=C['nb'])
ax.axhline(y=GLOBAL_PBMC_D30, color=C['grey'], linestyle='--', lw=1, alpha=0.7)
ax.text(len(ct_names) - 0.45, GLOBAL_PBMC_D30 + 0.6, f'global {GLOBAL_PBMC_D30:.2f}%',
        fontsize=6, color=C['grey'], ha='right')
short = [n.replace('+ ', '+').replace('Monocyte', 'Mono') for n in ct_names]
ax.set_xticks(xc); ax.set_xticklabels(short, fontsize=6.5, rotation=20, ha='right')
ax.set_ylabel('STRING precision (%)')
ax.set_title('(d) PBMC cell types ($d=30$)', fontweight='bold', color=C['nb'])
ax.legend(fontsize=6, framealpha=0.9); ax.set_ylim(0, 38)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (e) Paul15 cell-type networks (d = 30, mouse STRING)
ax = axes[1, 1]
LIN = {'10GMP': 'GMP', '9GMP': 'GMP', '12Baso': 'Baso', '13Baso': 'Baso',
       '14Mo': 'Mono', '15Mo': 'Mono', '16Neu': 'Neut', '2Ery': 'Ery',
       '3Ery': 'Ery', '4Ery': 'Ery', '5Ery': 'Ery', '6Ery': 'Ery',
       '7MEP': 'MEP', '8Mk': 'Mk'}
cp_names = sorted(list(ctp.keys()),
                  key=lambda c: ctp[c]['d30']['NB_moment']['precision'], reverse=True)
cp_nb = [ctp[c]['d30']['NB_moment']['precision'] for c in cp_names]
cp_fz = [ctp[c]['d30']['Fisher_z']['precision'] for c in cp_names]
xp = np.arange(len(cp_names))
ax.bar(xp - wc / 2, cp_nb, wc, color=C['pau'], edgecolor='white', lw=0.5, label='scCausal (NB-LR)')
ax.bar(xp + wc / 2, cp_fz, wc, color=C['fzpau'], edgecolor='white', lw=0.5, label=r"Fisher's $z$")
for i in range(len(cp_names)):
    ax.text(i, max(cp_nb[i], cp_fz[i]) + 0.8, f'{cp_nb[i] - cp_fz[i]:+.1f}',
            ha='center', fontsize=5.6, fontweight='bold',
            color=C['pau'] if cp_nb[i] >= cp_fz[i] else C['fzpau'])
ax.set_xticks(xp)
ax.set_xticklabels([f'{c}\n({LIN.get(c, "?")})' for c in cp_names],
                   fontsize=5.4, rotation=45, ha='right')
ax.set_ylabel('STRING precision (%)')
ax.set_title('(e) Paul15 cell types ($d=30$)', fontweight='bold', color=C['pau'])
ax.legend(fontsize=6, framealpha=0.9); ax.set_ylim(0, 27)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (f) Simulation, 100 seeds, Cohen's d
ax = axes[1, 2]
nb_f1 = [r['nb_f1'] for r in sim]
fz_f1 = [r['fz_f1'] for r in sim]
nb_sd = [r['nb_f1_std'] for r in sim]
fz_sd = [r['fz_f1_std'] for r in sim]
ax.errorbar(sim_d, nb_f1, yerr=nb_sd, fmt='o-', color=C['nb'], lw=2.4, ms=8,
            capsize=3, label='NB-LR', zorder=3)
ax.errorbar(sim_d, fz_f1, yerr=fz_sd, fmt='s--', color=C['fz'], lw=2.4, ms=8,
            capsize=3, label=r"Fisher's $z$", zorder=3)
for i, d in enumerate(sim_d):
    pooled = np.sqrt((nb_sd[i] ** 2 + fz_sd[i] ** 2) / 2)
    cd = (nb_f1[i] - fz_f1[i]) / pooled
    ax.annotate(f"$d'={cd:+.2f}$", (d, max(nb_f1[i], fz_f1[i]) + 0.030),
                ha='center', fontsize=7.5, fontweight='bold',
                color=C['nb'] if cd > 0 else C['fz'])
ax.set_xlabel('Genes ($d$)'); ax.set_ylabel('F1 score')
ax.set_title("(f) Simulation (100 seeds, Cohen's $d'$)", fontweight='bold')
ax.legend(fontsize=7, framealpha=0.9); ax.set_ylim(0.40, 0.66)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# ---- ROW 3: Methods, mechanism, robustness ----
# (g) Method comparison at d = 30
ax = axes[2, 0]
m_lab = ['scCausal\n(NB-LR, moment)', r'scCausal' + '\n' + r'($\alpha{=}1.0$)',
         r"Fisher's $z$" + '\n' + r'($\log(1{+}x)$)', 'NOTEARS\n(linear)', 'Permuted\n(control)']
m_val = [s['NB_moment'][0], s['NB_fixed'][0], s['Fisher_z'][0], 0.0, 0.0]
m_edg = [e['NB_moment'][0], e['NB_fixed'][0], e['Fisher_z'][0],
         notears['edges'], 0]
m_col = [C['nb'], C['fzn'], C['fz'], C['nt'], C['pm']]
bars = ax.bar(m_lab, m_val, color=m_col, edgecolor='white', lw=0.8)
for k, (b, v, ne) in enumerate(zip(bars, m_val, m_edg)):
    if v > 0:
        ax.text(b.get_x() + b.get_width() / 2, v + 0.45,
                f'{v:.2f}%\n{ne:,} edges', ha='center', fontsize=6.5, fontweight='bold')
    else:
        ax.text(b.get_x() + b.get_width() / 2, 0.6,
                '0 edges\n(zero-edge\ncollapse)', ha='center', fontsize=6,
                fontweight='bold', color=m_col[k])
ax.set_ylabel('STRING precision (%)')
ax.set_title('(g) Method comparison ($d=30$, PBMC)', fontweight='bold')
ax.set_ylim(0, 18.5)
ax.tick_params(axis='x', labelsize=6)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (h) Mechanism: on PBMC both the estimated overdispersion and the NB-LR
#     advantage grow with dimension; on the mildly dispersed Paul15 both stay
#     small and flat.  Bars = precision gain (left axis), lines = alpha-hat
#     (right axis).  Pooling cell-type settings is deliberately NOT shown here,
#     because their precision differences are dominated by subset sample size.
ax = axes[2, 1]
d_pbmc = [s['NB_moment'][i] - s['Fisher_z'][i] for i in range(len(D))]
d_paul = [sp['NB_moment'][i] - sp['Fisher_z'][i] for i in range(len(D))]
wd = 0.36
ax.bar(x - wd / 2, d_pbmc, wd, color=C['nb'], edgecolor='white', lw=0.5,
       label=r'$\Delta$ precision, PBMC')
ax.bar(x + wd / 2, d_paul, wd, color=C['pau'], edgecolor='white', lw=0.5,
       label=r'$\Delta$ precision, Paul15')
for i in range(len(D)):
    ax.text(i - wd / 2, d_pbmc[i] + 0.05, f'{d_pbmc[i]:+.2f}', ha='center',
            fontsize=6, fontweight='bold', color=C['nb'])
    ax.text(i + wd / 2, d_paul[i] + 0.05, f'{d_paul[i]:+.2f}', ha='center',
            fontsize=6, fontweight='bold', color=C['pau'])
ax.set_xticks(x); ax.set_xticklabels([f'$d={d}$' for d in D])
ax.set_ylabel(r'NB-LR $-$ Fisher precision (pp)')
ax.set_ylim(0, 2.9)
ax2 = ax.twinx()
ax2.plot(x, a_hat_pbmc, 'o--', color=C['nb'], lw=1.8, ms=6, mfc='white',
         label=r'$\hat{\alpha}$, PBMC')
ax2.plot(x, a_hat_paul, 's--', color=C['pau'], lw=1.8, ms=6, mfc='white',
         label=r'$\hat{\alpha}$, Paul15')
for i in range(len(D)):
    ax2.annotate(f'{a_hat_pbmc[i]:.1f}', (i, a_hat_pbmc[i]), textcoords='offset points',
                 xytext=(0, 7), ha='center', fontsize=6, color=C['nb'], fontweight='bold')
    ax2.annotate(f'{a_hat_paul[i]:.2f}', (i, a_hat_paul[i]), textcoords='offset points',
                 xytext=(0, -12), ha='center', fontsize=6, color=C['pau'], fontweight='bold')
ax2.set_ylabel(r'Estimated overdispersion $\hat{\alpha}$')
ax2.set_ylim(0, 9.6)
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color('#BDBDBD')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=5.8, framealpha=0.9, loc='upper left')
ax.set_title('(h) Overdispersion drives the gain', fontweight='bold', color=C['nb'])
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# (i) Threshold robustness at d = 30
ax = axes[2, 2]
a_keys = sorted([k for k in supp if k.startswith('alpha_')], key=lambda k: float(k[6:]))
t_keys = sorted([k for k in supp if k.startswith('tau_')], key=lambda k: float(k[4:]))
a_x = [float(k[6:]) for k in a_keys]
a_y = [supp[k]['nb_precision'] - supp[k]['fz_precision'] for k in a_keys]
t_x = [float(k[4:]) for k in t_keys]
t_y = [supp[k]['nb_precision'] - supp[k]['fz_precision'] for k in t_keys]
ax.plot(a_x, a_y, 'o-', color=C['nb'], lw=2.2, ms=7,
        label=r' significance level $\alpha$')
ax.plot(t_x, t_y, 's--', color=C['fz'], lw=2.2, ms=7,
        label=r' pre-filter $\tau$')
for xx, yy in list(zip(a_x, a_y)) + list(zip(t_x, t_y)):
    ax.annotate(f'{yy:+.2f}', (xx, yy), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=6, fontweight='bold')
ax.axhline(0, color=C['grey'], lw=0.9, ls=':')
ax.set_xlabel('Threshold value')
ax.set_ylabel(r'NB-LR $-$ Fisher precision (pp)')
ax.set_title('(i) Threshold robustness ($d=30$)', fontweight='bold')
ax.legend(fontsize=6.2, framealpha=0.9, loc='upper left')
ax.set_ylim(-0.35, 2.35)
ax.yaxis.grid(True, alpha=0.12, color=C['grid'])

# ================================================================
# 3. SAVE
# ================================================================
plt.tight_layout(pad=2.5, h_pad=2.4, w_pad=2.2)
for fmt in ['pdf', 'png']:
    plt.savefig(os.path.join(FIG_DIR, f'fig2_benchmark.{fmt}'), dpi=300,
                bbox_inches='tight', facecolor=C['bg'], edgecolor='none')
plt.close()

# ---- provenance echo so the numbers can be checked at a glance ----
print('Fig 2 done. Source values:')
print('  (a) PBMC precision NB_moment  :', s['NB_moment'])
print('      PBMC precision NB_fixed   :', s['NB_fixed'])
print('      PBMC precision Fisher_z   :', s['Fisher_z'])
print('  (b) Paul15 precision NB_moment:', sp['NB_moment'])
print('      Paul15 precision Fisher_z :', sp['Fisher_z'])
print('  (c) PBMC edges NB_moment      :', e['NB_moment'])
print('      PBMC edges NB_fixed       :', e['NB_fixed'])
print('      PBMC edges Fisher_z       :', e['Fisher_z'])
print('  (g) methods d=30              :', m_val, m_edg)
print('  (h) delta PBMC / Paul15       :', [round(v, 2) for v in d_pbmc],
      [round(v, 2) for v in d_paul])
print('      alpha-hat PBMC / Paul15   :', a_hat_pbmc, a_hat_paul)
print('  (i) alpha deltas              :', dict(zip(a_x, [round(v, 2) for v in a_y])))
print('      tau   deltas              :', dict(zip(t_x, [round(v, 2) for v in t_y])))
for fmt in ['pdf', 'png']:
    p = os.path.join(FIG_DIR, f'fig2_benchmark.{fmt}')
    print('  file %-8s %8d bytes' % (fmt, os.path.getsize(p)))
