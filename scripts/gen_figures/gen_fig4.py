"""
Fig 4: Causal skeleton network -- TikZ circular module layout.

Data source (FAIR protocol, same run as Table 1):
    results/fair_supplementary.json  ->  edges_d50.nb_validated
    (60 STRING-validated NB-LR edges at d = 50, spanning 40 genes)

HLA- and MT- prefixes are dropped from the on-figure labels for legibility;
the caption states this.
"""
import os, json, subprocess, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FIG_DIR = os.path.join(ROOT, 'figures')
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)

supp = json.load(open(os.path.join(RES_DIR, 'fair_supplementary.json'), encoding='utf-8'))
top_edges = [tuple(p) for p in supp['edges_d50']['nb_validated']]
D_LABEL = 50

# ---- functional modules -------------------------------------------------
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
MOD_NAMES = {'I': 'Immune/HLA', 'E': 'Effectors', 'R': 'Ribosomal',
             'S': 'S100/Ca', 'M': 'Mitochondrial', 'O': 'Other'}


def short(g):
    for pre in ('HLA-', 'MT-'):
        if g.startswith(pre):
            return g[len(pre):]
    return g


# ---- gather nodes actually present in the validated edge set ------------
genes_all = set()
for g1, g2 in top_edges:
    genes_all.add(g1); genes_all.add(g2)
unknown = sorted(g for g in genes_all if g not in CAT)
if unknown:
    print('WARNING: genes without a module assignment:', unknown)
    for g in unknown:
        CAT[g] = 'O'
    GROUPS['O'] = GROUPS['O'] + unknown

modules = {m: sorted([g for g in genes_all if CAT.get(g) == m]) for m in MODS}
modules = {m: v for m, v in modules.items() if v}

# === LAYOUT: modules grouped with gaps ===
NODE_R = 8.5      # node circle radius (cm)
LABEL_R = 9.9     # label circle radius (cm)
GAP_DEG = 7.0     # gap between modules (degrees); wide enough that the last
                  # label of one module cannot collide with the first of the next

mod_weights = {m: len(modules[m]) for m in modules}
total_nodes = sum(mod_weights.values())
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
        rad = math.radians(ang - 90)          # start from the top
        positions[g] = (NODE_R * math.cos(rad), NODE_R * math.sin(rad),
                        LABEL_R * math.cos(rad), LABEL_R * math.sin(rad),
                        ang, rad)
        all_nodes_ordered.append(g)
    ang_cur += mod_deg + GAP_DEG

# === GENERATE TIKZ ===
L = []
L.append(r'\documentclass[tikz,border=4pt]{standalone}')
L.append(r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{xcolor}')
L.append(r'\usepackage{tikz}')
L.append('')
for m in MODS:
    if m in modules:
        L.append(r'\definecolor{c%s}{HTML}{%s}' % (m, COL_HEX[m]))
L.append(r'\definecolor{eg}{HTML}{BDBDBD}')
L.append('')
L.append(r'\begin{document}')
L.append(r'\begin{tikzpicture}[scale=1.0]')
L.append('')

L.append(r'\node[font=\sffamily\bfseries\Large,align=center] at (0,11.2)')
L.append(r'  {scCausal causal skeleton \textemdash\ PBMC 3K ($d{=}%d$, STRING ${\geq}700$)};' % D_LABEL)
L.append(r'\node[font=\sffamily\itshape\footnotesize,text=gray] at (0,10.5)')
L.append(r'  {%d STRING-validated edges, %d genes in %d functional modules};'
         % (len(top_edges), len(genes_all), len(modules)))
L.append('')

# radial connector lines
for g in all_nodes_ordered:
    xn, yn, xl, yl, ang_deg, rad = positions[g]
    m = CAT.get(g, 'O')
    L.append(r'\draw[c%s,line width=0.3pt,opacity=0.25] (%.3f,%.3f) -- (%.3f,%.3f);'
             % (m, xn, yn, xl, yl))
L.append('')

# edges as Bezier curves
for g1, g2 in top_edges:
    xn1, yn1 = positions[g1][0], positions[g1][1]
    xn2, yn2 = positions[g2][0], positions[g2][1]
    mx = (xn1 + xn2) / 2; my = (yn1 + yn2) / 2
    dd = math.sqrt((xn1 - xn2) ** 2 + (yn1 - yn2) ** 2)
    push = 1.8 if dd > 12 else 0.7
    if abs(mx) > 0.01:
        mx += push * (mx / abs(mx))
    if abs(my) > 0.01:
        my += push * (my / abs(my))
    L.append(r'\draw[eg,line width=0.65pt,opacity=0.32] (%.3f,%.3f) .. controls (%.3f,%.3f) .. (%.3f,%.3f);'
             % (xn1, yn1, mx, my, xn2, yn2))
L.append('')

# nodes
for g in all_nodes_ordered:
    xn, yn = positions[g][0], positions[g][1]
    m = CAT.get(g, 'O')
    sz = '7pt' if m == 'I' else '5pt'
    L.append(r'\fill[c%s,draw=white,line width=1.1pt] (%.3f,%.3f) circle (%s);'
             % (m, xn, yn, sz))
L.append('')

# labels
for g in all_nodes_ordered:
    xl, yl, rad = positions[g][2], positions[g][3], positions[g][5]
    m = CAT.get(g, 'O')
    lbl = short(g)
    fs = r'\small' if len(lbl) <= 3 else (r'\footnotesize' if len(lbl) <= 5
                                          else r'\fontsize{6.6}{7.6}\selectfont')
    anchor = 'west' if abs(rad) < math.pi / 2 else 'east'
    L.append(r'\node[font=\sffamily\bfseries%s,text=c%s,anchor=%s,inner sep=0.8pt] at (%.3f,%.3f) {%s};'
             % (fs, m, anchor, xl, yl, lbl))
L.append('')

# legend (top-left)
lx0, ly0 = -11.5, 11.2
for m in MODS:
    if m in modules:
        L.append(r'\fill[c%s,draw=white,line width=0.5pt] (%.1f,%.1f) circle (4pt);'
                 % (m, lx0, ly0))
        L.append(r'\node[font=\sffamily\footnotesize,anchor=west] at (%.1f,%.1f) {%s (%d)};'
                 % (lx0 + 0.75, ly0, MOD_NAMES[m], len(modules[m])))
        ly0 -= 0.65

L.append(r'\end{tikzpicture}')
L.append(r'\end{document}')

# one figure, one script: the TikZ source lives beside this file, the rendered
# PDF/PNG land in figures/
tex = os.path.join(HERE, 'fig4_network.tex')
with open(tex, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')

for _ in range(2):
    subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', FIG_DIR, tex],
                   capture_output=True, cwd=HERE)

def _pdf_to_png(pdf_path, png_path, dpi=300):
    """Render the TikZ PDF to a preview PNG without needing poppler."""
    try:                                    # PyMuPDF (self-contained)
        import fitz
        d = fitz.open(pdf_path)
        d[0].get_pixmap(dpi=dpi).save(png_path)
        d.close()
        return 'pymupdf'
    except Exception:
        pass
    try:                                    # pdf2image + poppler, if present
        from pdf2image import convert_from_path
        convert_from_path(pdf_path, dpi=dpi)[0].save(png_path, 'PNG')
        return 'pdf2image'
    except Exception as e:
        return 'skipped (%s)' % e


print('png:', _pdf_to_png(os.path.join(FIG_DIR, 'fig4_network.pdf'),
                          os.path.join(FIG_DIR, 'fig4_network.png')))

# pdflatex wrote its intermediates into the output directory
for e in ['.aux', '.log']:
    f = os.path.join(FIG_DIR, 'fig4_network' + e)
    if os.path.exists(f):
        os.remove(f)

pdf = os.path.join(FIG_DIR, 'fig4_network.pdf')
print('Fig 4 done -- %d edges, %d genes, modules %s'
      % (len(top_edges), len(genes_all), {m: len(modules[m]) for m in modules}))
print('  pdf %d bytes' % os.path.getsize(pdf))
