"""Fig 1: Pipeline schematic — TikZ compilation + PNG conversion."""
import subprocess, os, sys

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
TEX = os.path.join(FIG_DIR, 'fig1_pipeline.tex')

# Compile TikZ -> PDF
print("Compiling TikZ...")
for _ in range(2):
    r = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-output-directory', FIG_DIR, TEX],
        capture_output=True, cwd=FIG_DIR
    )

# Convert PDF -> PNG via pdf2image (fallback: skip)
try:
    from pdf2image import convert_from_path
    images = convert_from_path(
        os.path.join(FIG_DIR, 'fig1_pipeline.pdf'), dpi=300
    )
    images[0].save(os.path.join(FIG_DIR, 'fig1_pipeline.png'), 'PNG')
    print("PNG converted (pdf2image)")
except ImportError:
    try:
        subprocess.run([
            'gswin64c', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m',
            '-r300', f'-sOutputFile={os.path.join(FIG_DIR, "fig1_pipeline.png")}',
            os.path.join(FIG_DIR, 'fig1_pipeline.pdf')
        ], capture_output=True, timeout=30)
        print("PNG converted (ghostscript)")
    except Exception:
        print("PNG not converted (install pdf2image or ghostscript)")

# Clean aux files
for ext in ['.aux', '.log']:
    f = TEX.replace('.tex', ext)
    if os.path.exists(f):
        os.remove(f)

print("Fig 1 done — TikZ pipeline schematic")
