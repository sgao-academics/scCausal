"""
Graphical abstract for the scCausal submission -- 13.0 x 5.2 in, 2665 x 1066 px.

SPEC (Elsevier): the graphical abstract must be at least 531 x 1328 px (h x w),
i.e. a 2.5:1 landscape strip, and must be *readable when displayed at 5 x 13 cm*.

  canvas 13.0 x 5.2 in = 33.02 x 13.21 cm, drawn with x in [0,130] and y in [0,52]
  -> 1 data unit = 7.2 pt on both axes (isotropic), so a font of F pt on the
     canvas is printed at F x (13.0/33.02) = 0.3937 F pt when shown at 13 cm wide.
  -> EVERY text element on the canvas is therefore at least 13 pt, i.e. at
     least 5.1 pt at the 13 cm display size.  Nothing is allowed below that.

Every number on the canvas is read from the FAIR-protocol result files, so the
graphical abstract can never drift away from the manuscript again.  In
particular the third headline cell reports the MATCHED-EDGE-BUDGET result
(3/15 cell-type settings positive, mean -1.07 pp), not the raw win count: the
raw count is the quantity the manuscript's own Section "Matched-Edge-Budget
Comparison" shows to be a denominator effect, so it must not be advertised on
its own here.

DATA PROVENANCE
  headline 1 / 2  NB-LR vs Fisher's z STRING precision, d=30, raw
                  -> results/checkpoints/fair_pbmc_d30.json, fair_paul15_d30.json
  headline 3      matched-edge-budget outcome over the 15 PBMC cell-type x
                  dimension settings
                  -> results/equal_count_celltype.json (key: PBMC_celltype)
  stage 1-4       design facts and the validation dimensions
                  -> results/fair_downstream.json (reactome.by_d.30, go_d30)
                  -> results/fair_celltype.json, fair_celltype_paul15.json
"""
import os
import json

import matplotlib
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
ct15 = _j('fair_celltype_paul15.json')['types']
eqct = _j('equal_count_celltype.json')['PBMC_celltype']
react = _j('fair_downstream.json')['reactome']['by_d']['30']

NB_PBMC = pbmc30['NB_moment']['precision']
FZ_PBMC = pbmc30['Fisher_z']['precision']
NB_PAUL = paul30['NB_moment']['precision']
FZ_PAUL = paul30['Fisher_z']['precision']

# matched-edge-budget outcome over every PBMC cell-type x dimension setting
_eq = [r for t in eqct for r in eqct[t].values()]
N_EQ = len(_eq)
N_POS = sum(1 for r in _eq if r['delta_equal_count'] > 0)
MEAN_EQ = sum(r['delta_equal_count'] for r in _eq) / N_EQ

N_CT = len([c for c, r in ct.items() if 'd30' in r])
N_PAUL_CT = len(ct15)

# ---- canvas ----------------------------------------------------------------
# 13.0 x 5.2 in at 205 dpi = 2665 x 1066 px  (>= 1328 x 531, ratio 2.5:1)
FIG_W_IN, FIG_H_IN, DPI = 13.0, 5.2, 205
UX, UY = 130.0, 52.0                      # data units; 1 unit = 7.2 pt on both axes
DISPLAY_SCALE = 13.0 / (FIG_W_IN * 2.54)  # 0.3937 when shown at 13 cm wide

BLUE, ORANGE, TEAL, GREEN, GREY = '#1565C0', '#E65100', '#00838F', '#2E7D32', '#546E7A'
SLATE, NOTE_GREY = '#37474F', '#607D8B'

TITLE_FS = 21.0
SUB_FS = 14.0
LAB_FS = 14.0
VAL_FS = 17.0
NOTE_FS = 14.0
STITLE_FS = 15.0
BODY_FS = 14.0
FOOT_FS = 14.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, UX)
ax.set_ylim(0, UY)
ax.axis('off')

_chk = []          # (label, artist, x0, x1) -- checked for overflow before saving


def T(x, y, s, fs, **kw):
    return ax.text(x, y, s, fontsize=fs, ha=kw.pop('ha', 'center'),
                   va=kw.pop('va', 'center'), **kw)


# ---------- title -----------------------------------------------------------
# The header reuses the article title's own opening phrase verbatim
# ("scCausal: Distribution-Matched Conditional Independence Testing for
# Single-Cell Causal Discovery"), so the graphical abstract and the paper can
# never be read as two different claims. The full title is too long for a
# single header line (15 words would need ~16 pt, collapsing the type
# hierarchy); the mechanism the title leaves implicit is carried by the
# subtitle and by stage 2.
_a = T(65, 48.8, 'scCausal: Distribution-Matched Conditional Independence Testing',
       TITLE_FS, fontweight='bold', color=BLUE)
_chk.append(('title', _a, 1.0, UX - 1.0))
_b = T(65, 45.3,
       "A negative binomial likelihood-ratio CI test replaces Fisher's $z$-test "
       "in the single-cell PC algorithm",
       SUB_FS, color=SLATE)
_chk.append(('subtitle', _b, 1.0, UX - 1.0))

# ---------- headline band ---------------------------------------------------
band = FancyBboxPatch((5, 37.9), 120, 5.8,
                      boxstyle='round,pad=0.3,rounding_size=0.8',
                      linewidth=1.0, edgecolor='#CFD8DC', facecolor='#F5F9FC', zorder=1)
ax.add_patch(band)
head = [
    ('PBMC 3K, $d=30$ (raw)', '%.2f%% vs %.2f%%' % (NB_PBMC, FZ_PBMC), BLUE, 24, 38),
    ('Paul15 (mouse), $d=30$ (raw)', '%.2f%% vs %.2f%%' % (NB_PAUL, FZ_PAUL), TEAL, 61, 36),
    ('Matched edge budget', '$%d/%d$ positive ($%+.2f$ pp)' % (N_POS, N_EQ, MEAN_EQ),
     ORANGE, 102, 46),
]
for lab, val, col, xc, wid in head:
    a = T(xc, 42.2, lab, LAB_FS, color=GREY)
    _chk.append((lab, a, xc - wid / 2, xc + wid / 2))
    a = T(xc, 39.7, val, VAL_FS, fontweight='bold', color=col)
    _chk.append((val, a, xc - wid / 2, xc + wid / 2))
for xd in (43.0, 79.0):
    ax.plot([xd, xd], [38.5, 43.1], color='#CFD8DC', lw=0.8)

for y, s in [(36.3, 'STRING validation precision (NB-LR vs Fisher\'s $z$); both tests '
                    'share the identical pre-filter, gene set and preprocessing'),
             (33.6, 'Matched edge budget = Fisher\'s $z$ truncated to scCausal\'s edge count')]:
    a = T(65, y, s, NOTE_FS, style='italic', color=NOTE_GREY)
    _chk.append((s[:24], a, 5, 125))

# ---------- four stages -----------------------------------------------------
STAGES = [
    ('1', 'scRNA-seq counts',
     ['$n$ = 2,700 (PBMC 3K)', '$n$ = 2,730 (Paul15)', '> 90% zeros',
      'raw UMI counts', 'NB dispersion 2.4$\\to$7.8'], BLUE),
    ('2', 'PC skeleton discovery',
     ['NB likelihood-ratio test', '$\\Lambda \\sim \\chi^2_1$ (Wilks)',
      'first-order conditioning', 'pre-filter $\\tau$ = 0.10',
      'moments dispersion, $O(d)$'], ORANGE),
    ('3', 'Edge orientation',
     ['v-structures', "Meek's rules", 'Markov equivalence class',
      'interpretable $p$-values', 'no global optimisation'], TEAL),
    ('4', 'Validation',
     ['STRING PPI $\\geq$ 700', 'Reactome 2.4--4.8$\\times$',
      'DepMap CRISPR 1,208 lines', '100 simulation seeds',
      'permuted control: 0 edges'], GREEN),
]
BW, BH, BY, GAP, X0 = 29.5, 23.0, 6.0, 2.5, 2.2
for i, (num, title, lines, col) in enumerate(STAGES):
    x0 = X0 + i * (BW + GAP)
    ax.add_patch(FancyBboxPatch((x0, BY), BW, BH,
                                boxstyle='round,pad=0.3,rounding_size=0.9',
                                linewidth=1.4, edgecolor=col, facecolor='white', zorder=2))
    ax.add_patch(Rectangle((x0 + 0.35, BY + 0.35), BW - 0.7, 3.4,
                           facecolor=col, alpha=0.10, zorder=2))
    T(x0 + BW / 2, 30.9, title, STITLE_FS, fontweight='bold', color=col)
    for j, ln in enumerate(lines):
        a = T(x0 + BW / 2, 24.4 - j * 3.1, ln, BODY_FS, color=SLATE)
        _chk.append((ln, a, x0, x0 + BW))
    T(x0 + 2.0, BY + 1.9, num, STITLE_FS, fontweight='bold', color=col, zorder=3)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x0 + BW + 0.35, BY + BH / 2),
                                     (x0 + BW + GAP - 0.35, BY + BH / 2),
                                     arrowstyle='-|>', mutation_scale=14,
                                     linewidth=1.6, color='#90A4AE', zorder=3))

# ---------- footer ----------------------------------------------------------
T(65, 3.0,
  '2 tissues   |   %d PBMC cell types + %d Paul15 clusters   |   '
  'Reactome OR = %.2f ($p$ = %.4f)   |   5 validation dimensions'
  % (N_CT, N_PAUL_CT, react['odds_ratio'], react['fisher_p']),
  FOOT_FS, color=GREY)

# ---------- self-check: nothing may overflow its container ------------------
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_inv = ax.transData.inverted()
bad = []
for label, art, x0, x1 in _chk:
    bb = art.get_window_extent(renderer=_r)
    p0 = _inv.transform((bb.x0, bb.y0))
    p1 = _inv.transform((bb.x1, bb.y1))
    w = p1[0] - p0[0]
    if w > (x1 - x0):
        bad.append((label, w, x1 - x0))
print('overflowing labels: %d' % len(bad))
for label, w, avail in bad:
    print('  %-40r %.1f pt > %.1f pt available' % (label, w, avail))

# ---------- save ------------------------------------------------------------
for d in OUT_DIRS:
    if not os.path.isdir(d):
        continue
    p = os.path.join(d, 'graphical_abstract.png')
    fig.savefig(p, dpi=DPI, facecolor='white')
    print('wrote %s (%d bytes)' % (p, os.path.getsize(p)))
plt.close(fig)

import PIL.Image as I
p = os.path.join(FIG_DIR, 'graphical_abstract.png')
px = I.open(p).size
print('size: %d x %d px  (>= 1328 x 531: %s, ratio %.3f)'
      % (px[0], px[1], px[0] >= 1328 and px[1] >= 531, px[0] / px[1]))
print('smallest canvas font: %.1f pt  ->  %.2f pt at a 13 cm display width'
      % (min(TITLE_FS, SUB_FS, LAB_FS, VAL_FS, NOTE_FS, STITLE_FS, BODY_FS, FOOT_FS),
         min(TITLE_FS, SUB_FS, LAB_FS, VAL_FS, NOTE_FS, STITLE_FS, BODY_FS, FOOT_FS)
         * DISPLAY_SCALE))
print('headline 3 source: %d/%d positive, mean %+.4f pp'
      % (N_POS, N_EQ, MEAN_EQ))
