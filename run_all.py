"""
scCausal Replication — One-Command Runner
=========================================
Reproduces all experiments and figures from:

  Gao, S. (2026). scCausal: Distribution-Matched Conditional Independence
  Testing for Single-Cell Causal Discovery. Submitted.

Usage:
  python run_all.py               # Full reproduction (requires STRING + Paul15 + DepMap)
  python run_all.py --pbmc-only   # PBMC 3K experiments only (no external data)
  python run_all.py --figs-only   # Regenerate figures from existing results
  python run_all.py --download    # Download required external datasets
  python run_all.py --validate    # Run DepMap + Reactome validation only
"""
import sys, os, subprocess, argparse, json, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
SCRIPTS = os.path.join(ROOT, 'scripts')
RESULTS = os.path.join(ROOT, 'results')
sys.path.insert(0, SCRIPTS)

PYTHON = sys.executable  # Ensure subprocess uses the same Python interpreter

def run(script, desc=""):
    """Run a Python script and report status."""
    name = desc or os.path.basename(script)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    script_path = os.path.join(SCRIPTS, script) if not script.startswith('scripts' + os.sep) else os.path.join(ROOT, script)
    r = subprocess.run([PYTHON, script_path], cwd=SCRIPTS)
    if r.returncode != 0:
        print(f"  WARNING: {name} returned code {r.returncode}")
    return r.returncode

def check_external_data():
    """Check if external datasets are available using config module."""
    import config
    missing = []
    dataset_keys = {
        'paul15': 'Paul15 hematopoietic data',
        'string_aliases': 'STRING aliases',
        'string_ppi': 'STRING PPI',
        'depmap_crispr': 'DepMap CRISPR',
        'trrust': 'TRRUST',
    }
    for key, name in dataset_keys.items():
        path = config.get_path(key)
        if not os.path.exists(path):
            missing.append(f'{name} (expected at: {path})')
    return missing

def run_pbmc_experiments():
    """Run PBMC 3K core experiments."""
    run('run_pbmc.py', 'PBMC 3K (NB-LR vs Fisher z)')
    run('run_synthetic.py', 'Synthetic Benchmarks (100 seeds)')
    run('run_baselines.py', 'Baselines (NOTEARS, GENIE3)')

def run_full_experiments():
    """Run all experiments including cross-tissue validation."""
    run_pbmc_experiments()
    run('run_everything.py', 'Full Sweep (all configs)')
    run('run_paul.py', 'Paul15 Cross-Tissue')
    run('run_fix30.py', 'Fixed d=30 + Cell Types')

def run_validation():
    """Run Reactome + DepMap validation."""
    run('_download_reactome.py', 'Download Reactome Pathways')
    run('_reactome_validate.py', 'Reactome Co-Membership Validation')
    run('_depmap_coessential.py', 'DepMap CRISPR Co-Essentiality')

def generate_figures():
    """Generate all figures from results."""
    figs_dir = os.path.join(SCRIPTS, 'gen_figures')
    for fig in ['gen_fig1.py', 'gen_fig2.py', 'gen_fig3.py', 'gen_fig4.py']:
        script_path = os.path.join(figs_dir, fig)
        print(f"\n{'='*60}\n  Figure: {fig}\n{'='*60}")
        r = subprocess.run([PYTHON, script_path], cwd=figs_dir)
        if r.returncode != 0:
            print(f"  WARNING: Figure: {fig} returned code {r.returncode}")
    print(f"\n  Figures saved to: {os.path.join(ROOT, 'figures')}")

def show_download_instructions():
    """Display dataset download instructions using config module."""
    import config
    config.print_download_instructions()

def verify_results():
    """Verify key result files exist and contain expected data."""
    checks = {
        '_sweep_checkpoint.json': ['d_30', 'd_50', 'd_100', 'd_200'],
        '_everything_ckpt.json': ['B_paul_d30', 'E_sim_d30'],
        'depmap_coessentiality.json': ['S100A8-S100A9'],
        'simulation_results.json': ['d_30', 'd_50', 'd_100'],
    }
    all_ok = True
    for fname, keys in checks.items():
        fpath = os.path.join(RESULTS, fname)
        if not os.path.exists(fpath):
            print(f"  MISSING: {fname}")
            all_ok = False
            continue
        with open(fpath) as f:
            data = json.load(f)
        missing_keys = [k for k in keys if k not in str(data)]
        if missing_keys:
            print(f"  INCOMPLETE: {fname} (missing keys: {missing_keys})")
            all_ok = False
        else:
            print(f"  OK: {fname}")
    return all_ok

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scCausal One-Command Replication')
    parser.add_argument('--pbmc-only', action='store_true', help='PBMC 3K experiments only')
    parser.add_argument('--figs-only', action='store_true', help='Regenerate figures only')
    parser.add_argument('--download', action='store_true', help='Show download instructions')
    parser.add_argument('--validate', action='store_true', help='Run external validation only')
    parser.add_argument('--verify', action='store_true', help='Verify existing results')
    args = parser.parse_args()

    print("scCausal Replication Runner")
    print(f"Package root: {ROOT}")
    print(f"Python: {sys.version}")

    if args.download:
        show_download_instructions()
        sys.exit(0)

    if args.verify:
        ok = verify_results()
        sys.exit(0 if ok else 1)

    if args.figs_only:
        generate_figures()
        print("\nDone. Figures regenerated.")
        sys.exit(0)

    if args.validate:
        run_validation()
        print("\nDone. Validation complete.")
        sys.exit(0)

    # Check external data availability
    missing = check_external_data()
    if missing and not args.pbmc_only:
        print("\nWARNING: Some external datasets are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun 'python run_all.py --download' for instructions.")
        print("Or use 'python run_all.py --pbmc-only' for PBMC-only reproduction.")
        if input("\nContinue anyway? (y/n): ").lower() != 'y':
            sys.exit(1)

    if args.pbmc_only:
        run_pbmc_experiments()
    else:
        run_full_experiments()
        run_validation()

    generate_figures()

    # Verify
    print("\n" + "=" * 60)
    print("  VERIFYING RESULTS")
    print("=" * 60)
    verify_results()

    print("\nDone. All experiments complete.")
    print(f"Results: {RESULTS}/")
    print(f"Figures: {os.path.join(ROOT, 'figures')}/")
