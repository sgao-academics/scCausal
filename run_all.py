"""scCausal reproduction runner.

Runs the full fair-protocol pipeline and regenerates every figure.

    python run_all.py                 full pipeline (needs all external datasets)
    python run_all.py --pbmc-only     PBMC 3K only, no external datasets required
    python run_all.py --quick         smoke test: d=30 only, a few minutes
    python run_all.py --figs-only     regenerate figures from the shipped results
    python run_all.py --verify        check the shipped results are complete
    python run_all.py --download      print dataset download instructions

Every stage writes into results/ and is safe to re-run: reproduce_benchmark.py,
fair_supplementary.py and fair_downstream.py all resume from per-configuration
checkpoints, so an interrupted run continues where it stopped.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, 'scripts')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
LOG = os.path.join(RESULTS, 'run_all_log.txt')

PY = sys.executable
sys.path.insert(0, SCRIPTS)

# Legacy Windows consoles use a code page that cannot represent every
# character, and a captured child traceback may carry any byte sequence.
# Replace unrepresentable characters instead of raising, and force children
# to emit UTF-8 so the captured text decodes predictably.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except Exception:
        pass

CHILD_ENV = dict(os.environ, PYTHONIOENCODING='utf-8')

STATUS = []


def banner(text):
    print('\n' + '=' * 68)
    print('  ' + text)
    print('=' * 68)


def run(script, args=(), label=None):
    """Run one script from scripts/ and record its status."""
    label = label or script
    cmd = [PY, os.path.join(SCRIPTS, script)] + [str(a) for a in args]
    banner('%s\n  %s' % (label, ' '.join(os.path.basename(c) for c in cmd)))
    t0 = time.time()
    with open(LOG, 'a', encoding='utf-8') as log:
        log.write('\n%s\n%s\n' % ('=' * 68, ' '.join(cmd)))
        log.flush()
        proc = subprocess.run(cmd, cwd=SCRIPTS, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', env=CHILD_ENV)
        out = proc.stdout or ''
        log.write(out)
        print(out, end='' if out.endswith('\n') else '\n')
    dt = time.time() - t0
    ok = proc.returncode == 0
    STATUS.append((label, 'ok' if ok else 'FAILED (exit %d)' % proc.returncode, dt))
    if not ok:
        print('  !! %s exited with code %d -- see %s'
              % (label, proc.returncode, LOG))
    return ok


def gen_figures():
    figs = os.path.join(SCRIPTS, 'gen_figures')
    scripts = ['gen_fig1.py', 'gen_fig2.py', 'gen_fig3.py', 'gen_fig4.py',
               'gen_fig5.py', 'gen_graphical_abstract.py']
    for f in scripts:
        banner('Figure: %s' % f)
        t0 = time.time()
        proc = subprocess.run([PY, os.path.join(figs, f)], cwd=figs,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace',
                              env=CHILD_ENV)
        lay = proc.stdout or ''
        with open(LOG, 'a', encoding='utf-8') as log:
            log.write('\n=== %s ===\n%s' % (f, lay))
        print(lay, end='' if lay.endswith('\n') else '\n')
        ok = proc.returncode == 0
        STATUS.append(('figure %s' % f, 'ok' if ok else 'FAILED', time.time() - t0))
        if not ok:
            print('  !! %s exited with code %d -- see %s' % (f, proc.returncode, LOG))
    print('\n  Figures written to %s' % FIGURES)


def missing(external_only=False):
    """List external datasets that are not present."""
    import config
    need = [('paul15', 'Paul15 hematopoietic data'),
            ('string_aliases', 'human STRING v11 aliases'),
            ('string_ppi', 'human STRING v11 links'),
            ('string_aliases_mouse', 'mouse STRING v11 aliases'),
            ('string_ppi_mouse', 'mouse STRING v11 links'),
            ('trrust', 'TRRUST v2'),
            ('depmap_crispr', 'DepMap CRISPR gene effect')]
    out = []
    for key, name in need:
        p = config.get_path(key)
        if not os.path.exists(p):
            out.append((key, name, p))
    return out


def verify():
    """Check that every result the figures read is present and complete."""
    checks = [
        ('fair_pbmc.json', ['"30"', '"50"', '"100"', '"200"',
                            'NB_moment', 'Fisher_z']),
        ('fair_paul15.json', ['NB_moment', 'Fisher_z']),
        ('fair_celltype.json', ['B', 'CD4+ T', 'CD8+ T', 'CD14+ Monocyte']),
        ('fair_celltype_paul15.json', ['types']),
        ('fair_supplementary.json', ['edges_d30', 'edges_d50', 'alpha_0.05',
                                     'tau_0.1', 'offset_d200']),
        ('fair_downstream.json', ['threshold', 'reactome', 'depmap']),
        ('table1_fair_all.json', ['precision']),
        ('zinb_type1_v2_results.json', ['empirical_type1']),
        ('zinb_calib_v2_results.json', ['type1_calibrated']),
    ]
    ok = True
    print('\n%-34s %s' % ('result file', 'status'))
    print('-' * 68)
    for name, keys in checks:
        p = os.path.join(RESULTS, name)
        if not os.path.exists(p):
            p = os.path.join(SCRIPTS, name)
        if not os.path.exists(p):
            print('%-34s MISSING' % name)
            ok = False
            continue
        blob = open(p, encoding='utf-8', errors='replace').read()
        gone = [k for k in keys if k not in blob]
        if gone:
            print('%-34s INCOMPLETE (no %s)' % (name, ', '.join(gone)))
            ok = False
        else:
            print('%-34s ok' % name)
    print()
    print('  RESULT:', 'all result files complete' if ok else 'problems found')
    return ok


def main():
    ap = argparse.ArgumentParser(description='scCausal one-command reproduction')
    ap.add_argument('--pbmc-only', action='store_true',
                    help='PBMC 3K experiments only')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: d=30 only')
    ap.add_argument('--figs-only', action='store_true',
                    help='regenerate figures from the shipped results')
    ap.add_argument('--verify', action='store_true',
                    help='check the shipped results for completeness')
    ap.add_argument('--download', action='store_true',
                    help='print external dataset download instructions')
    ap.add_argument('--skip-downstream', action='store_true',
                    help='skip Reactome and DepMap validation')
    args = ap.parse_args()

    print('scCausal reproduction runner')
    print('  package root : %s' % ROOT)
    print('  python       : %s' % sys.version.split()[0])
    print('  interpreter  : %s' % PY)

    if args.download:
        import config
        config.print_download_instructions()
        return 0

    if args.verify:
        return 0 if verify() else 1

    if args.figs_only:
        os.makedirs(RESULTS, exist_ok=True)
        gen_figures()
        summary()
        return 0

    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as log:
        log.write('\n\n%s\nrun_all.py  %s\n%s\n'
                  % ('#' * 68, time.strftime('%Y-%m-%d %H:%M:%S'), '#' * 68))

    dims = ['30'] if args.quick else ['30', '50', '100', '200']
    dataset = 'pbmc' if args.pbmc_only else 'both'

    gone = missing()
    if gone:
        print('\n  External datasets not found:')
        for key, name, path in gone:
            print('    - %-24s expected at %s' % (name, path))
        print('\n  Run "python run_all.py --download" for download instructions,')
        print('  or "python run_all.py --pbmc-only" for the PBMC-only pipeline.')

    # 1. benchmark ---------------------------------------------------------
    run('reproduce_benchmark.py', ['--d'] + dims + ['--dataset', dataset],
        label='fair benchmark (%s, d=%s)' % (dataset, '/'.join(dims)))

    # 2. cell-type skeletons ----------------------------------------------
    run('reproduce_benchmark.py', ['--celltype', '--d'] + dims
        + ['--dataset', dataset],
        label='cell-type-resolved skeletons')

    # 3. Table 1 -----------------------------------------------------------
    run('reproduce_table1.py', label='Table 1 assembly')

    # 4. supplementary sweeps ---------------------------------------------
    run('fair_supplementary.py',
        label='alpha_sig / tau / library-size-offset sweeps')

    # 5. downstream validation --------------------------------------------
    if args.quick or args.skip_downstream:
        print('\n  skipping downstream validation (Reactome + DepMap)')
        STATUS.append(('downstream validation', 'skipped', 0.0))
    else:
        run('fair_downstream.py',
            label='STRING thresholds, Reactome, DepMap')

    # 6. figures -----------------------------------------------------------
    gen_figures()

    summary()
    return 0


def summary():
    banner('SUMMARY')
    print('  %-52s %-18s %s' % ('stage', 'status', 'minutes'))
    print('  ' + '-' * 84)
    for label, status, dt in STATUS:
        print('  %-52s %-18s %6.1f' % (label[:52], status, dt / 60.0))
    print()
    print('  results : %s' % RESULTS)
    print('  figures : %s' % FIGURES)
    print('  log     : %s' % LOG)


if __name__ == '__main__':
    sys.exit(main())
