"""
Random-pair background rate for the STRING >= 700 criterion.

Motivation
----------
The manuscript reports STRING-validated precision for discovered edge sets.
A precision figure is only interpretable next to the rate at which *random*
gene pairs of the same universe reach the criterion; without that baseline a
precision of, e.g., 14.11% cannot be separated from the annotation density of
the selected (highly expressed, co-expressed) genes.

This script computes the background rate over all gene pairs of the selected
gene universe, using the loader convention of the manuscript
(`run_pbmc.load_string`): the alias table maps symbol -> STRING protein id; the
links file has 16 whitespace-separated columns and the combined score is the
last one; a pair is high-confidence at combined score >= 700.

Paths are resolved through `config.resolve`, so no dataset location appears in
the source: set SC_CAUSAL_STRING_ALIASES / SC_CAUSAL_STRING_PPI, or place the
files in `data/` (see README).

Usage
-----
    python scripts/string_background_rate.py

Output (PBMC 3K, fair protocol, tau = 0.10)
-------------------------------------------
    d=30 : discovered 34/241 = 14.11% (paper)  vs  universe background 46/406 = 11.33%
           one-sided Fisher OR = 1.42, p = 0.097   -> 1.35x enrichment
    d=50 : discovered 60/596 = 10.07% (paper)  vs  universe background 76/1176 = 6.46%
           one-sided Fisher OR = 1.71, p = 0.0022  -> 1.63x enrichment

Note: MALAT1 has no STRING mapping, so an exact-match recomputation scores 222
(d=30) and 568 (d=50) of the discovered edges; the hit counts (34 and 60)
reproduce the manuscript exactly. The manuscript reports precision over all
discovered edges, hence 14.11% / 10.07% rather than 15.32% / 10.56%.
"""
import gzip
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402  (project data-path resolver)

SC_CAUSAL = os.path.abspath(os.path.join(HERE, ".."))
CASES = [
    ("supp_edges_d30.json", "d=30"),
    ("supp_edges_d50.json", "d=50"),
]
HC_THRESHOLD = 700


def load_symbol_to_string(path):
    """symbol (upper) -> STRING protein id, from the STRING alias table."""
    mapping = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.strip().split("\t")
            if len(p) >= 2 and not p[1].isdigit():
                mapping[p[1].upper()] = p[0]
    return mapping


def best_scores_within(universe_ids, ppi_path):
    """One pass over the links file, keeping the best score per intra-universe pair."""
    best = {}
    with gzip.open(ppi_path, "rt", encoding="utf-8", errors="ignore") as fh:
        fh.readline()  # header
        for line in fh:
            p = line.strip().split()
            if len(p) < 3:
                continue
            a, b = p[0], p[1]
            if a in universe_ids and b in universe_ids:
                try:
                    score = int(p[-1])
                except ValueError:
                    continue
                key = (a, b) if a < b else (b, a)
                if score > best.get(key, -1):
                    best[key] = score
    return best


def main():
    alias = config.resolve("string_aliases", "SC_CAUSAL_STRING_ALIASES")
    ppi = config.resolve("string_ppi", "SC_CAUSAL_STRING_PPI")
    for path in (alias, ppi):
        if not path or not os.path.exists(path):
            sys.exit("missing input: %s\n"
                     "Set SC_CAUSAL_STRING_ALIASES / SC_CAUSAL_STRING_PPI or place the "
                     "files in data/." % path)

    sym2id = load_symbol_to_string(alias)
    print("alias table entries: %d" % len(sym2id))

    try:
        from scipy.stats import fisher_exact
    except ImportError:
        fisher_exact = None
        print("scipy unavailable: enrichment test will be skipped")

    out = []
    for ckpt, label in CASES:
        with open(os.path.join(SC_CAUSAL, "results", "checkpoints", ckpt), encoding="utf-8") as fh:
            d = json.load(fh)
        genes, edges = d["genes"], d["nb_edges"]
        g2id = {g: sym2id.get(g.upper(), "") for g in genes}
        universe = sorted({v for v in g2id.values() if v})
        print("\n=== %s : %d genes, %d discovered edges, %d mapped ==="
              % (label, len(genes), len(edges), len(universe)))

        best = best_scores_within(set(universe), ppi)
        all_pairs = list(itertools.combinations(universe, 2))
        bg_hits = sum(1 for k in all_pairs if best.get(k, -1) >= HC_THRESHOLD)
        bg_rate = 100.0 * bg_hits / len(all_pairs)
        print("  universe C(%d,2) = %d pairs; >=%d: %d  ->  background rate = %.2f%%"
              % (len(universe), len(all_pairs), HC_THRESHOLD, bg_hits, bg_rate))

        discovered = []
        for a, b in edges:
            x, y = g2id[a], g2id[b]
            if x and y:
                discovered.append((x, y) if x < y else (y, x))
        hit = sum(1 for k in discovered if best.get(k, -1) >= HC_THRESHOLD)
        rate = 100.0 * hit / max(len(discovered), 1)
        print("  discovered testable: %d; >=%d: %d  ->  precision = %.2f%%"
              % (len(discovered), HC_THRESHOLD, hit, rate))
        print("  enrichment = %.2fx" % (rate / bg_rate))

        rec = {"case": label, "universe": len(universe), "pairs": len(all_pairs),
               "background_hits": bg_hits, "background_rate_pct": round(bg_rate, 2),
               "discovered_testable": len(discovered), "discovered_hits": hit,
               "precision_pct": round(rate, 2),
               "enrichment": round(rate / bg_rate, 2)}
        if fisher_exact is not None:
            orr, pv = fisher_exact(
                [[hit, len(discovered) - hit], [bg_hits, len(all_pairs) - bg_hits]],
                alternative="greater")
            rec["odds_ratio"] = round(float(orr), 2)
            rec["p_one_sided"] = float(pv)
            print("  one-sided Fisher: OR = %.2f, p = %.3g" % (orr, pv))
        out.append(rec)

    dest = os.path.join(SC_CAUSAL, "results", "string_background.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved: %s" % dest)


if __name__ == "__main__":
    main()
