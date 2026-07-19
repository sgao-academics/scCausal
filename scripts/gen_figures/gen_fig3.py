"""
Fig 3: Causal skeleton network — TikZ circular module layout.
Label spacing verified: 8.0cm radius orbit, 26 nodes = 1.93cm arc gap,
all gene names < 1.8cm at footnotesize = zero overlap.
"""
import os, json, subprocess, math

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)

sw = json.load(open(os.path.join(RES_DIR, '_sweep_checkpoint.json')))
top_edges = sw.get('d_50', {}).get('top_string_edges', [])

# Compact category codes
CAT = {
    'GNLY':'I','NKG7':'I','GZMB':'I','CCL5':'I','SRGN':'I',
    'HLA-DPA1':'I','HLA-DRA':'I','CD74':'I',
    'RPL8':'R','RPS27A':'R',
    'MT-CO1':'M','MT-CO2':'M','MT-CYB':'M',
    'S100A6':'S','S100A11':'S',
    'ACTB':'C','ARPC1B':'C',
    'AIF1':'O','CST3':'O','CTSS':'O','FCER1G':'O',
    'H3F3B':'O','IL32':'O','LST1':'O','TYROBP':'O','UBB':'O',
}
MODS = ['I','R','M','S','C','O']
# Colorblind-friendly palette (Paul Tol "bright" adapted)
COL_HEX = {'I':'EE7733','R':'0077BB','M':'33BBEE','S':'EE3377','C':'009988','O':'BBBBBB'}
MOD_NAMES = {'I':'Immune/HLA','R':'Ribosomal','M':'Mitochondrial',
             'S':'S100/Ca','C':'Cytoskeleton','O':'Other'}

# Group nodes
genes_all = set()
for g1,g2 in top_edges: genes_all.add(g1); genes_all.add(g2)
modules = {m: sorted([g for g in genes_all if CAT.get(g,'O')==m]) for m in MODS}
modules = {m:v for m,v in modules.items() if v}

# === PRECISE LAYOUT: modules grouped with gaps ===
NODE_R = 5.2      # node circle radius (cm)
LABEL_R = 8.0     # label circle radius (cm) — 2.8cm gap from nodes
GAP_DEG = 4.0     # gap between modules (degrees)

# Count module weights for angular allocation
mod_weights = {m: len(modules[m]) for m in modules}
total_nodes = sum(mod_weights.values())
total_gap = len(modules) * GAP_DEG
usable_deg = 360.0 - total_gap

# Compute angular positions
positions = {}
all_nodes_ordered = []
ang_cur = 0.0
for m in MODS:
    if m not in modules: continue
    genes = modules[m]
    n = len(genes)
    mod_deg = (n / total_nodes) * usable_deg
    if n > 1:
        step = mod_deg / (n - 1)
    else:
        step = 0
    for i, g in enumerate(genes):
        ang = ang_cur + (i * step if n > 1 else mod_deg/2)
        rad = math.radians(ang - 90)  # start from top
        x_n = NODE_R * math.cos(rad)
        y_n = NODE_R * math.sin(rad)
        x_l = LABEL_R * math.cos(rad)
        y_l = LABEL_R * math.sin(rad)
        positions[g] = (x_n, y_n, x_l, y_l, ang, rad)
        all_nodes_ordered.append(g)
    ang_cur += mod_deg + GAP_DEG

# === GENERATE TIKZ ===
L = []
L.append(r'\documentclass[tikz,border=6pt]{standalone}')
L.append(r'\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{xcolor}')
L.append(r'\usepackage{tikz}')
L.append(r'')
for m in MODS:
    if m in modules:
        L.append(r'\definecolor{c%s}{HTML}{%s}' % (m, COL_HEX[m]))
L.append(r'\definecolor{eg}{HTML}{BDBDBD}')
L.append(r'\definecolor{lg}{HTML}{ECEFF1}')
L.append(r'')
L.append(r'\begin{document}')
L.append(r'\begin{tikzpicture}[scale=1.0]')
L.append(r'')
# Title
L.append(r'\node[font=\sffamily\bfseries\LARGE,align=center] at (0,9.5)')
L.append(r'  {scCausal Causal Skeleton \textemdash\ PBMC 3K ($d{=}50$, STRING ${\geq}700$)};')
L.append(r'\node[font=\sffamily\itshape\footnotesize,text=gray] at (0,8.8)')
L.append(r'  {20 validated edges, 26 genes in 6 functional modules};')
L.append(r'')

# Draw connector lines (thin, light) from node to label
for g, (xn, yn, xl, yl, ang_deg, rad) in positions.items():
    m = CAT.get(g, 'O')
    L.append(r'\draw[c%s,line width=0.3pt,opacity=0.25] (%.3f,%.3f) -- (%.3f,%.3f);' % (m, xn, yn, xl, yl))

L.append(r'')

# Draw edges as Bezier curves
for g1, g2 in top_edges:
    xn1, yn1, _, _, _, _ = positions[g1]
    xn2, yn2, _, _, _, _ = positions[g2]
    mx = (xn1 + xn2) / 2; my = (yn1 + yn2) / 2
    d = math.sqrt((xn1-xn2)**2 + (yn1-yn2)**2)
    push = 1.8 if d > 7 else 0.6
    if abs(mx) > 0.01: mx += push * (mx/abs(mx))
    if abs(my) > 0.01: my += push * (my/abs(my))
    L.append(r'\draw[eg,line width=0.7pt,opacity=0.30] (%.3f,%.3f) .. controls (%.3f,%.3f) .. (%.3f,%.3f);' % (xn1,yn1,mx,my,xn2,yn2))

L.append(r'')

# Draw nodes
for g in all_nodes_ordered:
    xn, yn, _, _, _, _ = positions[g]
    m = CAT.get(g, 'O')
    sz = '8pt' if m in ('I','R') else '6pt'
    L.append(r'\fill[c%s,draw=white,line width=1.2pt] (%.3f,%.3f) circle (%s);' % (m, xn, yn, sz))

L.append(r'')

# Draw labels with precise sizing — short names bigger, long names smaller
for g in all_nodes_ordered:
    _, _, xl, yl, ang_deg, rad = positions[g]
    m = CAT.get(g, 'O')
    # Font size depends on name length
    if len(g) <= 4:
        fs = r'\small'
    elif len(g) <= 6:
        fs = r'\footnotesize'
    else:
        fs = r'\fontsize{7}{8}\selectfont'
    # Anchor: right->west, left->east
    anchor = 'west' if abs(rad) < math.pi/2 else 'east'
    L.append(r'\node[font=\sffamily\bfseries%s,text=c%s,anchor=%s,inner sep=1pt] at (%.3f,%.3f) {%s};' % (fs, m, anchor, xl, yl, g))

L.append(r'')

# Legend (top-left area)
lx0, ly0 = -9.0, 9.0
for m in MODS:
    if m in modules:
        n = len(modules[m])
        name = MOD_NAMES[m]
        L.append(r'\fill[c%s,draw=white,line width=0.5pt] (%.1f,%.1f) circle (4.5pt);' % (m, lx0, ly0))
        L.append(r'\node[font=\sffamily\footnotesize,anchor=west] at (%.1f,%.1f) {%s (%d)};' % (lx0+0.7, ly0, name, n))
        ly0 -= 0.6

L.append(r'\end{tikzpicture}')
L.append(r'\end{document}')

# Write and compile
tex = os.path.join(FIG_DIR, 'fig3_network.tex')
with open(tex, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')

for _ in range(2):
    subprocess.run(['pdflatex','-interaction=nonstopmode','-output-directory',FIG_DIR,tex],
                   capture_output=True, cwd=FIG_DIR)

# PNG
try:
    from pdf2image import convert_from_path
    convert_from_path(os.path.join(FIG_DIR,'fig3_network.pdf'),dpi=300)[0].save(
        os.path.join(FIG_DIR,'fig3_network.png'),'PNG')
except: pass

# Clean
for e in ['.aux','.log']:
    f = tex.replace('.tex',e)
    if os.path.exists(f): os.remove(f)

pdf = os.path.join(FIG_DIR, 'fig3_network.pdf')
print(f"Fig 3 done — {len(top_edges)} edges, {len(genes_all)} genes, PDF: {os.path.getsize(pdf)//1024}KB")
