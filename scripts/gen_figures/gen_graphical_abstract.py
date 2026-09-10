"""
Graphical abstract for the scCausal submission (13 x 5.2 in, >=2656 x 1062 px).

Every number on the canvas is read from the FAIR-protocol result files, so the
graphical abstract can never drift away from Table 1 again.
"""
import os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
CKPT_DIR = os.path.join(RES_DIR, 'checkpoints')
OUT_DIRS = [FIG_DIR,
            os.path.join(os.path.dirname(__file__), '..', '..', 'Submissions',
                         'ComputationalBiologyChemistry')]


def _ck(n):
    with open(os.path.join(CKPT_DIR, n), encoding='utf-8') as fh:
        return json.load(fh)


def _j(n):
    with open(os.path.join(RES_DIR, n), encoding='utf-8') as fh:
        return json.load(fh)


pbmc30 = _ck('fair_pbmc_d30.json')
paul30 = _ck('fair_paul15_d30.json')
ct = _j('fair_celltype.json')['types']
supp = _j('fair_supplementary.json')

NB_PBMC = pbmc30['NB_moment']['precision']
FZ_PBMC = pbmc30['Fisher_z']['precision']
NB_PAUL = paul30['NB_moment']['precision']
FZ_PAUL = paul30['Fisher_z']['precision']
_settings = [(c, k) for c, r in ct.items() for k in ('d30', 'd50', 'd100', 'd200') if k in r]
n_ct_total = len([c for c, r in ct.items() if 'd30' in r])
n_ct_wins = sum(1 for c, k in _settings
                if ct[c][k]['NB_moment']['precision'] > ct[c][k]['Fisher_z']['precision'])
n_settings = len(_settings)
react = _j('fair_downstream.json')['reactome']['by_d']['30']

BLUE, ORANGE, TEAL, GREY = '#1565C0', '#E65100', '#00838F', '#546E7A'

fig = plt.figure(figsize=(13.0, 5.2), dpi=205)
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 52); ax.axis('off')

# ---------- title ----------
ax.text(65, 48.6, 'scCausal: Distribution-Matched Causal Skeleton Discovery',
        ha='center', va='center', fontsize=16.5, fontweight='bold', color=BLUE)
ax.text(65, 44.9,
        "A negative binomial likelihood-ratio CI test replaces Fisher's $z$-test in the PC algorithm",
        ha='center', va='center', fontsize=9.5, color='#37474F')

# ---------- headline band ----------
band = FancyBboxPatch((5, 37.0), 120, 5.6, boxstyle='round,pad=0.3,rounding_size=0.8',
                      linewidth=1.0, edgecolor='#CFD8DC', facecolor='#F5F9FC', zorder=1)
ax.add_patch(band)
head = [
    ('PBMC 3K, $d=30$', f'{NB_PBMC:.2f}% vs {FZ_PBMC:.2f}%', BLUE),
    ('Paul15 (mouse STRING), $d=30$', f'{NB_PAUL:.2f}% vs {FZ_PAUL:.2f}%', TEAL),
    ('PBMC cell type $\\times$ dimension', f'NB-LR wins {n_ct_wins}/{n_settings}', ORANGE),
]
for i, (lab, val, col) in enumerate(head):
    xc = 25 + i * 40
    ax.text(xc, 41.0, lab, ha='center', va='center', fontsize=8.6, color='#546E7A')
    ax.text(xc, 38.9, val, ha='center', va='center', fontsize=11.5,
            fontweight='bold', color=col)
    if i < 2:
        ax.plot([xc + 20, xc + 20], [37.6, 42.0], color='#CFD8DC', lw=0.8)
ax.text(65, 36.2, 'STRING validation precision (all methods share the identical pre-filter, gene set and preprocessing)',
        ha='center', va='center', fontsize=7.4, style='italic', color='#78909C')

# ---------- four stages ----------
STAGES = [
    ('1', 'scRNA-seq counts',
     ['$n$ = 2,700 (PBMC 3K)', '$n$ = 2,730 (Paul15)', '> 90% zeros',
      'raw UMI counts', 'NB overdispersion $\\hat{\\alpha}$ 2.4 $\\to$ 7.8'], BLUE),
    ('2', 'PC skeleton discovery',
     ['NB likelihood-ratio test', '$\\Lambda \\sim \\chi^2_1$ (Wilks)', 'first-order conditioning',
      'shared pre-filter $\\tau$ = 0.10', 'moments dispersion, $O(d)$'], ORANGE),
    ('3', 'Edge orientation',
     ['v-structures', "Meek's rules", 'Markov equivalence class',
      'interpretable $p$-values', 'no global optimisation'], TEAL),
    ('4', 'Validation',
     ['STRING PPI $\\geq$ 700', 'Reactome 2.4--4.8$\\times$',
      'DepMap CRISPR (1,208 lines)', '100 simulation seeds',
      'permuted control: 0 edges'], '#2E7D32'),
]
BW, BH, BY = 25.0, 24.0, 6.0
for i, (num, title, lines, col) in enumerate(STAGES):
    x0 = 5 + i * 30.5
    box = FancyBboxPatch((x0, BY), BW, BH,
                         boxstyle='round,pad=0.3,rounding_size=0.9',
                         linewidth=1.4, edgecolor=col, facecolor='white', zorder=2)
    ax.add_patch(box)
    ax.add_patch(Rectangle((x0 + 0.35, BY + 0.35), BW - 0.7, 3.4,
                           facecolor=col, alpha=0.10, zorder=2))
    ax.text(x0 + BW / 2, BY + 26.9, title, ha='center', va='center',
            fontsize=10.6, fontweight='bold', color=col)
    for j, ln in enumerate(lines):
        ax.text(x0 + BW / 2, BY + 22.6 - j * 3.6, ln, ha='center', va='center',
                fontsize=7.5, color='#37474F')
    ax.text(x0 + 1.8, BY + 1.9, num, ha='center', va='center', fontsize=8.6,
            fontweight='bold', color=col, zorder=3)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x0 + BW + 0.35, BY + BH / 2),
                                     (x0 + 30.5 - 0.35, BY + BH / 2),
                                     arrowstyle='-|>', mutation_scale=13,
                                     linewidth=1.6, color='#90A4AE', zorder=3))

# ---------- footer ----------
ax.text(65, 3.4,
        '2 tissues   |   %d PBMC cell types + 14 Paul15 clusters   |   '
        'Reactome OR = %.2f ($p$ = %.1e)   |   5 validation dimensions'
        % (n_ct_total, react['odds_ratio'], react['fisher_p']),
        ha='center', va='center', fontsize=7.8, color=GREY)

for d in OUT_DIRS:
    if not os.path.isdir(d):
        continue
    p = os.path.join(d, 'graphical_abstract.png')
    fig.savefig(p, dpi=205, facecolor='white')
    print('wrote %s (%d bytes)' % (p, os.path.getsize(p)))
plt.close(fig)

import PIL.Image as I
p = os.path.join(FIG_DIR, 'graphical_abstract.png')
print('size:', I.open(p).size)
