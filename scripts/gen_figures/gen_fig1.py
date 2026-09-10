"""Fig 1: Pipeline schematic -- TikZ compilation + PNG conversion."""
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

# Convert PDF -> PNG for the PNG preview only; the manuscript embeds the PDF.
PDF = os.path.join(FIG_DIR, 'fig1_pipeline.pdf')
PNG = os.path.join(FIG_DIR, 'fig1_pipeline.png')


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


print("PNG:", pdf_to_png())

# Clean aux files
for ext in ['.aux', '.log']:
    f = TEX.replace('.tex', ext)
    if os.path.exists(f):
        os.remove(f)

print("Fig 1 done -- TikZ pipeline schematic")
