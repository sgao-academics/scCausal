"""reproduce_benchmark.py
Authoritative, FAIR reproduction of the scCausal benchmark (Table 1 + Table 3).

What it does
------------
1. Loads PBMC 3K and Paul15 directly with h5py + scipy.sparse (no scanpy
   dependency). Both are read from the local copy of the datasets.
2. Selects the top-d highly variable genes by variance.
3. Runs two PC conditional-independence skeletons on IDENTICAL data:
     - scCausal  : PC + NB-LR with data-driven (method-of-moments) dispersion.
     - Fisher's z: the standard Fisher-z PC algorithm on raw counts.
   Both use the SAME pre-filter threshold tau and SAME gene set, so the
   comparison is strictly fair (this removes the old confound where NB got
   tau=0.20 at d=30 while Fisher got tau=0.10).
4. Also reports a fixed-alpha (alpha=1.0) NB-LR transparency control.
5. Scores each recovered skeleton by STRING v11 precision (combined score
   >= 700, high confidence), exactly as in the paper.
6. Optionally runs cell-type-specific skeleton recovery for PBMC.

Usage
-----
    python reproduce_benchmark.py --d 30 50 100 200
    python reproduce_benchmark.py --celltype
"""
import sys, os, gzip, json, time, argparse, hashlib, pickle
import numpy as np
from scipy.stats import chi2, norm, pearsonr
from statsmodels.genmod.families import NegativeBinomial
import statsmodels.api as sm
import h5py
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402  (scCausal data path configuration)

ROOT = os.path.abspath(os.path.join(HERE, '..'))
RESULT_DIR = os.path.join(ROOT, 'results')
DATA_DIR = os.path.join(ROOT, 'data')
# Parsed-STRING caches (avoid re-parsing 70-160 MB gzip files on every run).
CACHE_DIR = os.path.join(RESULT_DIR, 'cache')
# Per-(dataset,d) incremental checkpoints for resumable runs.
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')


# ============================================================
# Logging: mirror stdout to results/run_log.txt for background runs
# ============================================================
class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, 'a', encoding='utf-8', errors='ignore', buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass


def _file_fingerprint(path):
    """Cheap, deterministic identity for an input file (size + mtime)."""
    try:
        st = os.stat(path)
        return {'path': path, 'size': int(st.st_size), 'mtime': int(st.st_mtime)}
    except Exception as e:
        return {'path': path, 'error': str(e)}


# ============================================================
# Checkpoint helpers (atomic write, resumable)
# ============================================================
def _ckpt_path(tag, d):
    return os.path.join(CKPT_DIR, '%s_d%d.json' % (tag, d))


def _ckpt_load(tag, d):
    p = _ckpt_path(tag, d)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as fh:
            obj = json.load(fh)
        return obj
    except Exception as e:
        print('  [ckpt] unreadable %s (%s); recomputing' % (p, e))
        return None


def _ckpt_save(tag, d, obj):
    os.makedirs(CKPT_DIR, exist_ok=True)
    p = _ckpt_path(tag, d)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, p)   # atomic: a crash never leaves a half-written checkpoint
    return p


def _ckpt_load_simple(tag):
    """Checkpoint keyed by tag alone (tag already encodes dataset and d)."""
    p = os.path.join(CKPT_DIR, '%s.json' % tag)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as e:
        print('  [ckpt] unreadable %s (%s); recomputing' % (p, e))
        return None


def _ckpt_save_simple(tag, obj):
    os.makedirs(CKPT_DIR, exist_ok=True)
    p = os.path.join(CKPT_DIR, '%s.json' % tag)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, p)
    return p

# --- Dataset paths ---
PBMC_H5AD = os.path.join(DATA_DIR, 'pbmc3k_filtered.h5ad')
PAUL_H5AD = config.get_path('paul15')
# --- STRING v11 resources ---
# Human STRING (taxid 9606) for PBMC, mouse STRING (taxid 10090) for Paul15.
# Paul15 is a mouse haematopoiesis dataset, so it MUST be validated against the
# mouse STRING network; validating it against human STRING yields zero hits and
# is methodologically invalid.
ALIAS = config.resolve('string_aliases', 'SC_CAUSAL_STRING_ALIASES')
PPI = config.resolve('string_ppi', 'SC_CAUSAL_STRING_PPI')
def _mouse_string(key, env_var, filename):
    """Locate a mouse STRING file: the configured data directory first, then
    the package's own data/ folder (where earlier releases expected it)."""
    p = config.resolve(key, env_var)
    if os.path.exists(p):
        return p
    return os.path.join(DATA_DIR, filename)


M_ALIAS = _mouse_string('string_aliases_mouse', 'SC_CAUSAL_STRING_ALIASES_MOUSE',
                        '10090.string_aliases.txt.gz')
M_PPI = _mouse_string('string_ppi_mouse', 'SC_CAUSAL_STRING_PPI_MOUSE',
                      '10090.string_ppi_full.txt.gz')


# ============================================================
# LOAD h5ad via h5py + scipy.sparse (no scanpy)
# ============================================================
def read_sparse_x(f, shape):
    """Read a CSR-encoded 'X' group into a dense float32 ndarray."""
    d = np.asarray(f['X/data'], dtype=np.float32)
    idx = np.asarray(f['X/indices'], dtype=np.int32)
    indptr = np.asarray(f['X/indptr'], dtype=np.int32)
    mat = sp.csr_matrix((d, idx, indptr), shape=shape)
    return np.asarray(mat.toarray(), dtype=np.float32)


def read_gene_names(f, var_index_key='index'):
    """Read gene names from var group. Handles both 'index' and '_index'."""
    if var_index_key in f['var']:
        raw = np.asarray(f['var'][var_index_key])
    elif '_index' in f['var']:
        raw = np.asarray(f['var']['_index'])
    else:
        # fallback: first column of a data field
        raw = np.asarray(f['var'].get('gene_ids',
                                     np.arange(shape_select(f))))
    return [_decode(x) for x in raw]


def _decode(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode('utf-8', errors='ignore')
    return str(x)


def shape_select(f):
    return int(len(f['var']['_index'])) if '_index' in f['var'] else int(len(f['var']['index']))


def load_pbmc():
    f = h5py.File(PBMC_H5AD, 'r')
    shape = tuple(int(v) for v in f['X'].attrs['shape'])
    X = read_sparse_x(f, shape)
    genes = read_gene_names(f, 'index')
    ct = None
    if 'cell_type' in f['obs']:
        ct = read_obs_categorical(f['obs']['cell_type'])
    n_cells = shape[0]
    f.close()
    return X, genes, ct, n_cells


def load_paul():
    f = h5py.File(PAUL_H5AD, 'r')
    X = np.asarray(f['X'], dtype=np.float32)
    genes = read_gene_names(f, '_index')
    ct = None
    if 'paul15_clusters' in f['obs']:
        ct = read_obs_categorical(f['obs']['paul15_clusters'])
    n_cells = X.shape[0]
    f.close()
    return X, genes, ct, n_cells


def read_obs_categorical(grp):
    """Read a scanpy categorical obs field (codes + categories)."""
    try:
        codes = np.asarray(grp['codes'], dtype=np.int64)
        cat = np.asarray(grp['categories'])
        cats = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c) for c in cat]
        return np.array([cats[c] if 0 <= c < len(cats) else 'NA' for c in codes])
    except Exception:
        try:
            raw = np.asarray(grp[:])
            return np.array([r.decode() if isinstance(r, (bytes, np.bytes_)) else str(r) for r in raw])
        except Exception:
            return None


# ============================================================
# STRING
# ============================================================
def _string_cache_key(alias_path, ppi_path):
    h = hashlib.md5(('%s|%s' % (alias_path, ppi_path)).encode('utf-8')).hexdigest()[:16]
    return h


def _parse_symbol_map(alias_path):
    symbol2string = {}
    with gzip.open(alias_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.strip().split('\t')
            if len(p) >= 2 and not p[1].isdigit():
                symbol2string[p[1].upper()] = p[0]
    return symbol2string


def _parse_ppi(ppi_path, min_score=700):
    ppi_set = set()
    with gzip.open(ppi_path, 'rt', encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                try:
                    if int(p[-1]) >= min_score:
                        ppi_set.add((p[0], p[1]))
                except ValueError:
                    continue
    return ppi_set


def _load_string_raw(alias_path, ppi_path):
    """Parse (or restore from cache) the symbol map and high-confidence PPI set.

    The human PPI file is ~160 MB gzip and the mouse one ~70 MB; parsing them on
    every invocation dominates runtime. We cache the parsed structures keyed by
    the *content identity* of both inputs, so a regenerated cache is used only
    when the inputs are unchanged. Cache files live in results/cache/.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _string_cache_key(alias_path, ppi_path)
    sym_fp = _file_fingerprint(alias_path)
    ppi_fp = _file_fingerprint(ppi_path)
    meta_path = os.path.join(CACHE_DIR, 'string_%s.meta.json' % key)
    sym_path = os.path.join(CACHE_DIR, 'string_%s.symbols.pkl' % key)
    ppi_path_c = os.path.join(CACHE_DIR, 'string_%s.ppi700.pkl' % key)

    if os.path.exists(meta_path) and os.path.exists(sym_path) and os.path.exists(ppi_path_c):
        try:
            with open(meta_path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
            if meta.get('alias') == sym_fp and meta.get('ppi') == ppi_fp:
                with open(sym_path, 'rb') as fh:
                    symbol2string = pickle.load(fh)
                with open(ppi_path_c, 'rb') as fh:
                    ppi_set = pickle.load(fh)
                print('[cache] STRING restored: %d symbols, %d PPI(>=700) edges'
                      % (len(symbol2string), len(ppi_set)))
                return symbol2string, ppi_set
            print('[cache] STRING inputs changed; rebuilding cache')
        except Exception as e:
            print('[cache] unreadable STRING cache (%s); rebuilding' % e)

    t0 = time.time()
    symbol2string = _parse_symbol_map(alias_path)
    ppi_set = _parse_ppi(ppi_path, 700)
    print('[cache] parsed STRING in %.1fs: %d symbols, %d PPI(>=700) edges'
          % (time.time() - t0, len(symbol2string), len(ppi_set)))

    for tmp_target, obj in [(sym_path, symbol2string), (ppi_path_c, ppi_set)]:
        tmp = tmp_target + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(obj, fh, protocol=4)
        os.replace(tmp, tmp_target)
    with open(meta_path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump({'alias': sym_fp, 'ppi': ppi_fp}, fh, indent=2)
    os.replace(meta_path + '.tmp', meta_path)
    return symbol2string, ppi_set


def load_string(alias_path, ppi_path, gene_names):
    symbol2string, ppi_set = _load_string_raw(alias_path, ppi_path)
    # Keys are stored upper-cased so that they match the upper-cased look-ups
    # used in score(). This is essential for mouse gene symbols (e.g. 'Mpo',
    # mixed case) and harmless for human symbols (already all-caps).
    gene2string = {g.upper(): symbol2string.get(g.upper(), '') for g in gene_names}
    return gene2string, ppi_set


# ============================================================
# NB-LR (moment dispersion) and Fisher-z
# ============================================================
def estimate_alpha_moment(X):
    """NB2 dispersion by the method of moments: Var = mu + alpha*mu^2.
    alpha_hat(g) = (s^2_g - xbar_g)/xbar_g^2, global = median over genes.
    """
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    valid = mu > 0
    alphas = (var[valid] - mu[valid]) / np.maximum(mu[valid] ** 2, 1e-8)
    alphas = np.clip(alphas, 1e-4, None)
    return float(np.median(alphas))


def nb_lr(X, i, j, cond, alpha=None):
    y = X[:, j]
    Xc = X[:, list(cond)] if len(cond) > 0 else None
    Xn = sm.add_constant(Xc) if Xc is not None and Xc.shape[1] > 0 else np.ones((len(y), 1))
    try:
        fam = NegativeBinomial(alpha=alpha) if alpha is not None else NegativeBinomial()
        m0 = sm.GLM(y, Xn, family=fam).fit(maxiter=50, disp=0)
        if m0.llf is None:
            return 1.0
        Xa = np.column_stack([Xn, X[:, i]]) if Xn.shape[1] > 0 else sm.add_constant(X[:, i].reshape(-1, 1))
        m1 = sm.GLM(y, Xa, family=fam).fit(maxiter=50, disp=0)
        if m1.llf is None:
            return 1.0
        return 1 - chi2.cdf(max(2 * (m1.llf - m0.llf), 0), 1)
    except Exception:
        return 1.0


def pc_nb(X, d, alpha=0.05, tau=0.10, max_k=1, use_moment=True):
    alpha_est = estimate_alpha_moment(X[:, :d]) if use_moment else None
    corr = np.corrcoef(np.log1p(X[:, :d]).T)
    edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(corr[i, j]) > tau}
    new_e = set()
    for (i, j) in edges:
        if nb_lr(X[:, :d], i, j, [], alpha=alpha_est) < alpha:
            new_e.add((i, j))
    edges = new_e
    if max_k >= 1:
        removed = set()
        for (i, j) in list(edges):
            nb_i, nb_j = set(), set()
            for (a, b) in edges:
                if a == i and b != j:
                    nb_i.add(b)
                elif b == i and a != j:
                    nb_i.add(a)
                if a == j and b != i:
                    nb_j.add(b)
                elif b == j and a != i:
                    nb_j.add(a)
            for k in list(nb_i & nb_j)[:5]:
                if nb_lr(X[:, :d], i, j, [k], alpha=alpha_est) >= alpha:
                    removed.add((i, j))
                    break
        edges -= removed
    return sorted(edges)


def pc_fz(X, d, alpha=0.05, tau=0.10, max_k=1):
    n = X.shape[0]
    logX = np.log1p(X[:, :d])
    corr = np.corrcoef(logX.T)
    edges = set()
    for i in range(d):
        for j in range(i + 1, d):
            if abs(corr[i, j]) > tau:
                z = 0.5 * np.log((1 + corr[i, j]) / max(1 - corr[i, j], 1e-10))
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 3)))) < alpha:
                    edges.add((i, j))
    if max_k >= 1:
        removed = set()
        for (i, j) in list(edges):
            nb_i, nb_j = set(), set()
            for (a, b) in edges:
                if a == i and b != j:
                    nb_i.add(b)
                elif b == i and a != j:
                    nb_i.add(a)
                if a == j and b != i:
                    nb_j.add(b)
                elif b == j and a != i:
                    nb_j.add(a)
            for k in list(nb_i & nb_j)[:5]:
                A = np.column_stack([np.ones(n), logX[:, k]])
                res_i = logX[:, i] - A @ np.linalg.lstsq(A, logX[:, i], rcond=None)[0]
                res_j = logX[:, j] - A @ np.linalg.lstsq(A, logX[:, j], rcond=None)[0]
                r, _ = pearsonr(res_i, res_j)
                z = 0.5 * np.log((1 + r) / max(1 - r, 1e-10))
                if 2 * (1 - norm.cdf(abs(z * np.sqrt(n - 4)))) >= alpha:
                    removed.add((i, j))
                    break
        edges -= removed
    return sorted(edges)


def score(edges, genes, gene2s, ppi):
    hits = 0
    for (i, j) in edges:
        sid = gene2s.get(genes[i].upper())
        tid = gene2s.get(genes[j].upper())
        if sid and tid and ((sid, tid) in ppi):
            hits += 1
    prec = 100.0 * hits / len(edges) if edges else 0.0
    return len(edges), hits, prec


# ============================================================
# Benchmark a single dataset across d
# ============================================================
def run_dataset(name, X, genes, d_list, tau, out_name, alias_path=None, ppi_path=None,
                force=False):
    var = X.var(axis=0)
    order = np.argsort(var)[::-1]
    alias_path = alias_path or ALIAS
    ppi_path = ppi_path or PPI
    # Validate the STRING resource actually maps the genes (catches the
    # human-on-mouse mismatch before wasting compute).
    gene2s, ppi = load_string(alias_path, ppi_path, genes)
    mapped = sum(1 for g in genes if gene2s.get(g.upper()))
    if mapped == 0:
        raise RuntimeError(
            f'{name}: STRING mapped 0/{len(genes)} genes using {alias_path}. '
            f'Using a mismatched STRING resource (e.g. human STRING on mouse '
            f'data) is methodologically invalid; aborting.')
    print(f'[ok] {name}: STRING resource maps {mapped}/{len(genes)} genes.')
    mapped = sum(1 for g in genes if gene2s.get(g.upper()))

    tag = os.path.splitext(out_name)[0]
    all_res = {}
    print(f'=== {name}: {X.shape[0]} cells x {X.shape[1]} genes, '
          f'STRING mapped {mapped}/{len(genes)} ({100*mapped/max(len(genes),1):.0f}%) ===')
    for d in d_list:
        cached = _ckpt_load(tag, d)
        if cached is not None and not force:
            all_res[str(d)] = cached
            print('  [resume] %s d=%d from checkpoint (%s)'
                  % (name, d, os.path.basename(_ckpt_path(tag, d))))
            continue
        idx = order[:d]
        Xd = X[:, idx]
        Gd = [genes[i] for i in idx]
        alpha_est = estimate_alpha_moment(Xd)
        print(f'  d={d}: alpha_hat={alpha_est:.4f}, zeros={100*(Xd==0).mean():.1f}%')
        res = {}
        for method, use_moment in [('NB_moment', True), ('NB_fixed', False)]:
            t0 = time.time()
            edges = pc_nb(Xd, d, alpha=0.05, tau=tau, use_moment=use_moment)
            el = time.time() - t0
            n_e, n_h, prec = score(edges, Gd, gene2s, ppi)
            res[method] = {'edges': n_e, 'hits': n_h, 'precision': round(prec, 2), 'time_s': round(el, 1)}
            print(f'    {method:<10} edges={n_e:5d} hits={n_h:4d} prec={prec:5.2f}% ({el:.0f}s)')
        t0 = time.time()
        edges = pc_fz(Xd, d, alpha=0.05, tau=tau)
        el = time.time() - t0
        n_e, n_h, prec = score(edges, Gd, gene2s, ppi)
        res['Fisher_z'] = {'edges': n_e, 'hits': n_h, 'precision': round(prec, 2), 'time_s': round(el, 1)}
        print(f'    {"Fisher_z":<10} edges={n_e:5d} hits={n_h:4d} prec={prec:5.2f}% ({el:.0f}s)')
        res['meta'] = {'d': d, 'n': Xd.shape[0], 'alpha_hat': round(alpha_est, 4),
                       'tau': tau, 'fair': True, 'name': name,
                       'string_fp': _file_fingerprint(ppi_path)}
        all_res[str(d)] = res
        _ckpt_save(tag, d, res)   # write incrementally so a crash loses nothing

    out = os.path.join(RESULT_DIR, out_name)
    with open(out, 'w') as fh:
        json.dump(all_res, fh, indent=2)
    print(f'  -> {out}')
    return all_res


def _slug(s):
    return ''.join(c if (c.isalnum() or c in '-_') else '_' for c in str(s))


def run_celltype(X, genes, celltype, d_list, tau=0.10, min_cells=200, force=False,
                 alias_path=None, ppi_path=None, prefix='ct', out_name='fair_celltype.json'):
    """Per cell-type skeleton recovery, fair NB(moment) vs Fisher.

    Only cell types with at least min_cells cells are used (smaller subsets
    have too little statistical power for a meaningful CI test), and only
    dimensions where n_cells >= 2*d are evaluated. Results are checkpointed per
    (cell-type, d) so the run resumes after any interruption. Pass alias_path/
    ppi_path to use a species-matched STRING resource (mouse for Paul15).
    """
    alias_path = alias_path or ALIAS
    ppi_path = ppi_path or PPI
    gene2s, ppi = load_string(alias_path, ppi_path, genes)
    mapped = sum(1 for g in genes if gene2s.get(g.upper()))
    print(f'[celltype:{prefix}] STRING maps {mapped}/{len(genes)} genes')

    order = np.argsort(X.var(axis=0))[::-1]
    types = [t for t in np.unique(celltype) if (celltype == t).sum() >= min_cells]
    res = {}
    for t in types:
        mask = celltype == t
        n_cells = int(mask.sum())
        row = {'n_cells': n_cells}
        for d in d_list:
            if n_cells < 50 or n_cells < 2 * d:
                continue
            tag = '%s_%s_d%d' % (prefix, _slug(t), d)
            cached = _ckpt_load_simple(tag)
            if cached is not None and not force:
                row['d%d' % d] = cached
                print('  [resume] celltype=%s d=%d from checkpoint' % (t, d))
                continue
            idx = order[:d]
            Xd_all = X[:, idx]
            Gd = [genes[i] for i in idx]
            Xt = Xd_all[mask]
            alpha_est = estimate_alpha_moment(Xt)
            dr = {'alpha_hat': round(alpha_est, 4)}
            for method, use_moment in [('NB_moment', True), ('NB_fixed', False)]:
                edges = pc_nb(Xt, d, alpha=0.05, tau=tau, use_moment=use_moment)
                n_e, n_h, prec = score(edges, Gd, gene2s, ppi)
                dr[method] = {'edges': n_e, 'hits': n_h, 'precision': round(prec, 2)}
            edges = pc_fz(Xt, d, alpha=0.05, tau=tau)
            n_e, n_h, prec = score(edges, Gd, gene2s, ppi)
            dr['Fisher_z'] = {'edges': n_e, 'hits': n_h, 'precision': round(prec, 2)}
            dr['delta_moment_fisher'] = round(dr['NB_moment']['precision']
                                              - dr['Fisher_z']['precision'], 2)
            row['d%d' % d] = dr
            _ckpt_save_simple(tag, dr)
            print(f'  {t:<18} d={d:<4} n={n_cells:5d} alpha={alpha_est:.3f} '
                  f'NB(m)={dr["NB_moment"]["precision"]:.2f}% '
                  f'Fz={dr["Fisher_z"]["precision"]:.2f}% '
                  f'delta={dr["delta_moment_fisher"]:+.2f}pp')
        res[t] = row

    out = os.path.join(RESULT_DIR, out_name)
    with open(out, 'w') as fh:
        json.dump({'d_list': d_list, 'tau': tau, 'min_cells': min_cells,
                   'types': res}, fh, indent=2)
    print(f'  -> {out}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, nargs='*', default=[30, 50, 100, 200])
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--dataset', choices=['pbmc', 'paul', 'both'], default='both')
    ap.add_argument('--celltype', action='store_true')
    ap.add_argument('--ct_min_cells', type=int, default=200,
                    help='minimum cells per type for the cell-type analysis')
    ap.add_argument('--force', action='store_true',
                    help='ignore existing checkpoints and recompute everything')
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Mirror all output to a log file so background runs are inspectable.
    log_path = os.path.join(RESULT_DIR, 'run_log.txt')
    sys.stdout = _Tee(sys.__stdout__, log_path)
    print('\n' + '=' * 70)
    print('reproduce_benchmark.py  dataset=%s  d=%s  tau=%.2f  celltype=%s  force=%s'
          % (args.dataset, args.d, args.tau, args.celltype, args.force))
    print('=' * 70)

    combined = {}
    if args.dataset in ('pbmc', 'both'):
        X, genes, ct, _ = load_pbmc()
        combined['PBMC'] = run_dataset('PBMC', X, genes, args.d, args.tau,
                                       'fair_pbmc.json', ALIAS, PPI, force=args.force)
        if args.celltype and ct is not None:
            combined['celltype'] = run_celltype(X, genes, ct, args.d, tau=args.tau,
                                                min_cells=args.ct_min_cells,
                                                force=args.force)
    if args.dataset in ('paul', 'both'):
        X, genes, ct, _ = load_paul()
        # Paul15 is mouse data: validate against mouse STRING, not human STRING.
        combined['Paul15'] = run_dataset('Paul15', X, genes, args.d, args.tau,
                                         'fair_paul15.json', M_ALIAS, M_PPI,
                                         force=args.force)
        if args.celltype and ct is not None:
            combined['celltype_paul15'] = run_celltype(
                X, genes, ct, args.d, tau=args.tau, min_cells=args.ct_min_cells,
                force=args.force, alias_path=M_ALIAS, ppi_path=M_PPI,
                prefix='ctpaul', out_name='fair_celltype_paul15.json')

    with open(os.path.join(RESULT_DIR, 'fair_benchmark.json'), 'w') as fh:
        json.dump({'by_dataset': combined, 'tau': args.tau, 'fair': True,
                   'd': args.d}, fh, indent=2, default=str)
    print(f'\nCombined -> results/fair_benchmark.json')
    print('DONE')


if __name__ == "__main__":
    main()
