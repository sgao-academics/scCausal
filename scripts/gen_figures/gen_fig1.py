"""Fig 1: framework schematic -- TikZ compilation + PNG preview.

One figure, one script: the TikZ source fig1_pipeline.tex sits beside this file
and is drawn at 1:1 print scale (canvas 16.30 x 7.10 cm vs a 16.46 cm textwidth),
so the 6-7.5 pt type in the source is the type that reaches the page.

The rendered PDF/PNG land in figures/. A non-zero pdflatex exit stops the script
instead of silently leaving last run's PDF in place.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FIG_DIR = os.path.join(ROOT, 'figures')
TEX = os.path.join(HERE, 'fig1_pipeline.tex')
PDF = os.path.join(FIG_DIR, 'fig1_pipeline.pdf')
PNG = os.path.join(FIG_DIR, 'fig1_pipeline.png')
LOG = os.path.join(FIG_DIR, 'fig1_pipeline.log')


def compile_tikz():
    """Compile twice (TikZ needs a second pass for the bounding box)."""
    src_mtime = os.path.getmtime(TEX)
    for _ in range(2):
        res = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-file-line-error',
             '-output-directory', FIG_DIR, TEX],
            capture_output=True, cwd=HERE)
    if res.returncode != 0:
        tail = ''
        if os.path.exists(LOG):
            with open(LOG, encoding='utf-8', errors='replace') as fh:
                tail = ''.join(l for l in fh if l.startswith('!'))[:600]
        sys.exit('pdflatex failed (rc=%d) on %s\n%s' % (res.returncode, TEX, tail))
    if not os.path.exists(PDF) or os.path.getmtime(PDF) < src_mtime:
        sys.exit('pdflatex reported success but %s was not regenerated' % PDF)


def pdf_to_png():
    """Try PyMuPDF, then pdf2image/poppler, then ghostscript."""
    try:
        import fitz
        doc = fitz.open(PDF)
        doc[0].get_pixmap(dpi=300).save(PNG)
        doc.close()
        return 'pymupdf'
    except Exception:
        pass
    try:
        from pdf2image import convert_from_path
        convert_from_path(PDF, dpi=300)[0].save(PNG, 'PNG')
        return 'pdf2image'
    except Exception as exc:
        try:
            subprocess.run(['gswin64c', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m',
                            '-r300', '-sOutputFile=%s' % PNG, PDF],
                           capture_output=True, timeout=60)
            return 'ghostscript'
        except Exception:
            return 'skipped (%s)' % exc


print('Compiling TikZ...')
compile_tikz()
print('PNG:', pdf_to_png())

# pdflatex wrote its intermediates into the output directory
for ext in ['.aux', '.log']:
    f = os.path.join(FIG_DIR, 'fig1_pipeline' + ext)
    if os.path.exists(f):
        os.remove(f)

print('Fig 1 done -- framework schematic (%d bytes PDF, %d bytes PNG)'
      % (os.path.getsize(PDF), os.path.getsize(PNG)))
