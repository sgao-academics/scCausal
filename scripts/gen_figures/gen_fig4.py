"""
Fig 4: Causal skeleton network -- TikZ circular module layout.

Data source (FAIR protocol, same run as Table 1):
    results/fair_supplementary.json  ->  edges_d50.nb_validated
    (60 STRING-validated NB-LR edges at d = 50, spanning 40 genes)

HLA- and MT- prefixes are dropped from the on-figure labels for legibility;
the caption states this.

DESIGN CONTRACT (same contract as fig1_pipeline.tex)
  * sans-serif Helvetica throughout, no coloured text -- colour lives only in
    fills and rules
  * every coordinate below is in FINAL PRINTED CENTIMETRES: the drawing is made
    at 1:1, so a 6 pt label really prints at 6 pt
  * the canvas is forced with \\path[use as bounding box], with
    16.26 cm + the 2 pt standalone border = 466.6 pt = \\textwidth of cas-sc
  * body text 6-6.5 pt, i.e. inside the 5-7 pt band required by the journals
  * flat design: hairline rules, no drop shadow, no gradient

What the figure has to say: the 60 validated edges are not spread uniformly.
53 of them stay inside one of six literature-defined functional modules and
only 7 cross module boundaries, so the modules are drawn as contiguous arcs
and each module's internal wiring is drawn in that module's colour while the
7 cross-module edges are dashed and neutral.
"""
import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)

supp = json.load(open(os.path.join(RES_DIR, "fair_supplementary.json"),
                      encoding="utf-8"))
top_edges = [tuple(p) for p in supp["edges_d50"]["nb_validated"]]
D_LABEL = 50
STR_THRESH = 700

# ---- functional modules (literature-defined, not learned from the edges) --
GROUPS = {
    'I': ['HLA-DPA1', 'HLA-DRA', 'HLA-DPB1', 'HLA-DRB1', 'CD74', 'B2M', 'CTSS'],
    'E': ['GNLY', 'NKG7', 'GZMB', 'CCL5', 'SRGN', 'IL32', 'PPBP'],
    'R': ['RPL8', 'RPL13', 'RPS3A', 'RPS4X', 'RPS5', 'RPS27A'],
    'S': ['S100A4', 'S100A6', 'S100A8', 'S100A9', 'S100A11', 'LGALS1'],
    'M': ['MT-CO1', 'MT-CO2', 'MT-CYB'],
    'O': ['ACTB', 'ARPC1B', 'AIF1', 'CST3', 'FCER1G', 'H3F3B', 'LST1',
          'TYROBP', 'UBB', 'FTL', 'FTH1'],
}
CAT = {g: k for k, v in GROUPS.items() for g in v}
MODS = ['I', 'E', 'R', 'S', 'M', 'O']
COL_HEX = {'I': 'EE7733', 'E': '882255', 'R': '0077BB',
           'S': 'EE3377', 'M': '33BBEE', 'O': 'BBBBBB'}
# the two lightest module colours are darkened for the wiring, otherwise a
# 0.65-opacity grey/cyan rule disappears on white paper
COL_EDGE = dict(COL_HEX, O='8C99A6')
MOD_NAMES = {'I': 'Immune/HLA', 'E': 'Effectors', 'R': 'Ribosomal',
             'S': 'S100/Ca', 'M': 'Mitochondrial', 'O': 'Other'}

INK = '1F2933'          # all text
MUTED = '5C6B76'        # secondary text / cross-module edges
RULE = 'D3DCE3'

# ---- canvas (cm) ----------------------------------------------------------
CANVAS_W = 16.32        # 16.32 cm = 461.61 pt, + 2 pt border each side = 466.6
CANVAS_H = 14.70        # pt = \textwidth of cas-sc, i.e. printed scale 1.000
NODE_R = 6.078          # node circle radius
LABEL_R = 7.079         # label circle radius
GAP_DEG = 7.0           # angular gap between modules
DOT_R = 0.170           # node dot radius (cm)
FS_LAB = r'\fontsize{6}{7}\selectfont'
FS_LEG = r'\fontsize{6.5}{8}\selectfont'


def short(g):
    for pre in ('HLA-', 'MT-'):
        if g.startswith(pre):
            return g[len(pre):]
    return g


# ---- gather nodes actually present in the validated edge set -------------
genes_all = set()
for g1, g2 in top_edges:
    genes_all.add(g1)
    genes_all.add(g2)
unknown = sorted(g for g in genes_all if g not in CAT)
if unknown:
    raise SystemExit('unmapped genes: %s' % unknown)

modules = {m: sorted(g for g in genes_all if CAT[g] == m) for m in MODS}
modules = {m: v for m, v in modules.items() if v}
total_nodes = sum(len(v) for v in modules.values())

# ---- layout: contiguous arcs, one arc per module -------------------------
total_gap = len(modules) * GAP_DEG
usable_deg = 360.0 - total_gap

positions = {}
all_nodes_ordered = []
ang_cur = 0.0
for m in MODS:
    if m not in modules:
        continue
    genes = modules[m]
    n = len(genes)
    mod_deg = (n / total_nodes) * usable_deg
    step = mod_deg / (n - 1) if n > 1 else 0
    for i, g in enumerate(genes):
        ang = ang_cur + (i * step if n > 1 else mod_deg / 2)
        rad = math.radians(ang - 90)          # first module starts at the bottom
        positions[g] = (NODE_R * math.cos(rad), NODE_R * math.sin(rad),
                        LABEL_R * math.cos(rad), LABEL_R * math.sin(rad),
                        ang, rad)
        all_nodes_ordered.append(g)
    ang_cur += mod_deg + GAP_DEG

# ---- edge bookkeeping ----------------------------------------------------
intra = [(a, b) for a, b in top_edges if CAT[a] == CAT[b]]
cross = [(a, b) for a, b in top_edges if CAT[a] != CAT[b]]

deg = {g: 0 for g in genes_all}
for a, b in top_edges:
    deg[a] += 1
    deg[b] += 1

# =================== generate TikZ ========================================
L = []
L.append(r'\documentclass[tikz,border=2pt]{standalone}')
L.append(r'\usepackage[T1]{fontenc}')
L.append(r'\usepackage{fix-cm}% CM at arbitrary sizes -> no font substitution')
L.append(r'\usepackage{helvet}')
L.append(r'\usepackage{xcolor}')
L.append(r'\usepackage{tikz}')
L.append(r'\renewcommand{\familydefault}{\sfdefault}')
L.append('')
for m in MODS:
    if m in modules:
        L.append(r'\definecolor{c%s}{HTML}{%s}' % (m, COL_HEX[m]))
        L.append(r'\definecolor{ce%s}{HTML}{%s}' % (m, COL_EDGE[m]))
L.append(r'\definecolor{ink}{HTML}{%s}' % INK)
L.append(r'\definecolor{muted}{HTML}{%s}' % MUTED)
L.append(r'\definecolor{rule}{HTML}{%s}' % RULE)
L.append(r'\definecolor{neut}{HTML}{FFFFFF}')
L.append('')
L.append(r'\begin{document}')
L.append(r'\begin{tikzpicture}')
L.append('')
L.append(r'% force the exact canvas so the printed scale is exactly 1.000')
L.append(r'\path[use as bounding box] (%.2f,%.2f) rectangle (%.2f,%.2f);'
         % (-CANVAS_W / 2, -CANVAS_H / 2, CANVAS_W / 2, CANVAS_H / 2))
L.append('')

# --- cross-module edges first, so the intra-module wiring sits on top ------
for g1, g2 in cross:
    xn1, yn1 = positions[g1][0], positions[g1][1]
    xn2, yn2 = positions[g2][0], positions[g2][1]
    mx, my = (xn1 + xn2) / 2.0, (yn1 + yn2) / 2.0
    dd = math.hypot(xn1 - xn2, yn1 - yn2)
    push = 1.29 if dd > 8.6 else 0.50
    if abs(mx) > 0.01:
        mx += push * (mx / abs(mx))
    if abs(my) > 0.01:
        my += push * (my / abs(my))
    L.append(r'\draw[muted,line width=0.7pt,opacity=0.85,'
             r'dash pattern=on 1.5pt off 1.2pt] '
             r'(%.3f,%.3f) .. controls (%.3f,%.3f) .. (%.3f,%.3f);'
             % (xn1, yn1, mx, my, xn2, yn2))
L.append('')

# --- intra-module edges, in the module colour -----------------------------
for g1, g2 in intra:
    m = CAT[g1]
    xn1, yn1 = positions[g1][0], positions[g1][1]
    xn2, yn2 = positions[g2][0], positions[g2][1]
    mx, my = (xn1 + xn2) / 2.0, (yn1 + yn2) / 2.0
    dd = math.hypot(xn1 - xn2, yn1 - yn2)
    push = 1.29 if dd > 8.6 else 0.50
    if abs(mx) > 0.01:
        mx += push * (mx / abs(mx))
    if abs(my) > 0.01:
        my += push * (my / abs(my))
    L.append(r'\draw[ce%s,line width=0.55pt,opacity=0.68] '
             r'(%.3f,%.3f) .. controls (%.3f,%.3f) .. (%.3f,%.3f);'
             % (m, xn1, yn1, mx, my, xn2, yn2))
L.append('')

# --- radial connectors from dot to label ----------------------------------
for g in all_nodes_ordered:
    xn, yn, xl, yl = positions[g][:4]
    m = CAT[g]
    L.append(r'\draw[c%s,line width=0.35pt,opacity=0.35] '
             r'(%.3f,%.3f) -- (%.3f,%.3f);' % (m, xn, yn, xl, yl))
L.append('')

# --- nodes ----------------------------------------------------------------
for g in all_nodes_ordered:
    xn, yn = positions[g][0], positions[g][1]
    m = CAT[g]
    L.append(r'\fill[c%s,draw=white,line width=0.9pt] (%.3f,%.3f) circle (%.3fcm);'
             % (m, xn, yn, DOT_R))
L.append('')

# --- labels (black, uniform size) -----------------------------------------
for g in all_nodes_ordered:
    xl, yl, rad = positions[g][2], positions[g][3], positions[g][5]
    lbl = short(g)
    anchor = 'west' if abs(rad) < math.pi / 2 else 'east'
    dx = 0.055 if anchor == 'west' else -0.055
    L.append(r'\node[font=%s\bfseries,text=ink,anchor=%s,inner sep=0pt] '
             r'at (%.3f,%.3f) {%s};' % (FS_LAB, anchor, xl + dx, yl, lbl))
L.append('')

# --- legend, centred inside the ring --------------------------------------
LX0, LX1 = -2.95, 0.30          # two columns of module entries
LY = [1.52, 0.92, 0.32]
L.append(r'% legend: opaque plate keeps the text clear of the chords behind it')
bx0, bx1 = -3.30, 3.05
by0, by1 = -1.40, 1.95
L.append(r'\fill[white] (%.2f,%.2f) rectangle (%.2f,%.2f);'
         % (bx0, by0, bx1, by1))
L.append(r'\draw[rule,line width=0.5pt,rounded corners=1.5pt] '
         r'(%.2f,%.2f) rectangle (%.2f,%.2f);' % (bx0, by0, bx1, by1))
for k, m in enumerate([m for m in MODS if m in modules]):
    col = LX0 + (k // 3) * 3.05
    row = LY[k % 3]
    L.append(r'\fill[c%s,draw=white,line width=0.5pt] (%.2f,%.2f) circle (0.115cm);'
             % (m, col, row))
    L.append(r'\node[font=%s,text=ink,anchor=west,inner sep=0pt] at (%.2f,%.2f) '
             r'{%s (%d)};' % (FS_LEG, col + 0.24, row, MOD_NAMES[m],
                              len(modules[m])))
# edge-style entries
L.append(r'\draw[ink,line width=0.7pt] (%.2f,%.2f) -- (%.2f,%.2f);'
         % (LX0 - 0.16, -0.32, LX0 + 0.16, -0.32))
L.append(r'\node[font=%s,text=ink,anchor=west,inner sep=0pt] at (%.2f,%.2f) '
         r'{edge within a module (%d)};' % (FS_LEG, LX0 + 0.30, -0.32, len(intra)))
L.append(r'\draw[muted,line width=0.7pt,dash pattern=on 1.5pt off 1.2pt] '
         r'(%.2f,%.2f) -- (%.2f,%.2f);'
         % (LX0 - 0.16, -0.92, LX0 + 0.16, -0.92))
L.append(r'\node[font=%s,text=ink,anchor=west,inner sep=0pt] at (%.2f,%.2f) '
         r'{edge between modules (%d)};' % (FS_LEG, LX0 + 0.30, -0.92, len(cross)))
L.append('')
L.append(r'\end{tikzpicture}')
L.append(r'\end{document}')

# ---- write + compile (one figure, one script) ----------------------------
tex = os.path.join(HERE, "fig4_network.tex")
with open(tex, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode",
                    "-output-directory", FIG_DIR, tex],
                   capture_output=True, cwd=HERE)


def _pdf_to_png(pdf_path, png_path, dpi=300):
    """Render the TikZ PDF to a preview PNG without needing poppler."""
    try:                                    # PyMuPDF (self-contained)
        import fitz
        d = fitz.open(pdf_path)
        d[0].get_pixmap(dpi=dpi).save(png_path)
        d.close()
        return "pymupdf"
    except Exception:
        pass
    try:                                    # pdf2image + poppler, if present
        from pdf2image import convert_from_path
        convert_from_path(pdf_path, dpi=dpi)[0].save(png_path, "PNG")
        return "pdf2image"
    except Exception as e:
        return "skipped (%s)" % e


print("png:", _pdf_to_png(os.path.join(FIG_DIR, "fig4_network.pdf"),
                          os.path.join(FIG_DIR, "fig4_network.png")))

for e in [".aux", ".log"]:
    f = os.path.join(FIG_DIR, "fig4_network" + e)
    if os.path.exists(f):
        os.remove(f)

pdf = os.path.join(FIG_DIR, "fig4_network.pdf")
print("Fig 4 done -- %d edges (%d intra-module, %d cross-module), %d genes, "
      "modules %s" % (len(top_edges), len(intra), len(cross), len(genes_all),
                      {m: len(modules[m]) for m in modules}))
print("  d=%d, STRING combined score >= %d, n=%s"
      % (D_LABEL, STR_THRESH, supp["edges_d50"].get("n")))
print("  degree: min %d, max %d, mean %.2f"
      % (min(deg.values()), max(deg.values()), sum(deg.values()) / len(deg)))
print("  canvas %.2f x %.2f cm, dot r=%.3f cm"
      % (CANVAS_W + 4 / 72 * 2.54, CANVAS_H + 4 / 72 * 2.54, DOT_R))
print("  pdf %d bytes" % os.path.getsize(pdf))
