"""Fig 3: Type-I error of the NB-LR test under zero-inflation.

Panel (a): asymptotic chi-square Type-I error against the zero-inflation rate
           pi0, for n=2700 and theta=2 over R=200 replicate datasets.
Panel (b): asymptotic versus permutation-calibrated Type-I error at the three
           configurations where calibration was run (R=60 replicate datasets,
           each calibrated against B=200 permutations).

Both panels are read from the experiment outputs rather than hardcoded:
    zinb_type1_v2_results.json    -> panel (a)   (scripts/zinb_type1_v2.py)
    zinb_calib_v2_results.json    -> panel (b)   (scripts/zinb_calib_v2.py)

Design notes
------------
* The figure is drawn at its printed size, so every font size below is the size
  the reader actually sees. The manuscript places it with
  ``includegraphics[width=0.82\textwidth]`` inside the 466.6 pt ``cas-sc``
  text column, i.e. 382.61 pt wide; ``FIG_W_IN`` is calibrated so that the
  emitted PDF page is exactly that wide (scale 1.0, no downscaling).
* ``bbox_inches='tight'`` is deliberately NOT used: it silently changes the
  canvas size and destroys the 1:1 mapping. ``savefig`` without it emits a page
  of exactly ``figsize * 72`` points.
* Panel (a) plots pi0 on its true scale. Bars at equally spaced positions would
  visually equate the 0.5-0.7 gap with the 0-0.5 gap, a factor-2.5 distortion;
  type-I error is in fact a curve in pi0, so it is drawn as one.
* Replicate counts are finite (R=200 in panel a, R=60 in panel b), so every
  estimate carries a 95 % Wilson score interval. Reading the intervals: all
  three calibrated rates have intervals that cover the nominal 0.05, while no
  asymptotic interval does. The intervals also show that the 0.5 -> 0.7 step
  of panel (a) is within Monte-Carlo error, whereas the overall inflation is
  not.
* Every numeric label is printed to three decimals so that the figure and
  Table 2 of the manuscript carry literally the same numbers.

Usage:
    python gen_fig3.py                 # write into ../../figures/
    python gen_fig3.py --outdir DIR    # write somewhere else (safe preview)
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(ROOT, "figures")


def _find(name):
    """Locate a result file in results/, falling back to the scripts/ copy."""
    for d in (os.path.join(ROOT, "results"), os.path.join(ROOT, "scripts")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit("cannot find %s in results/ or scripts/" % name)


TYPE1_JSON = _find("zinb_type1_v2_results.json")
CALIB_JSON = _find("zinb_calib_v2_results.json")

ALPHA = 0.05
PANEL_A_N = 2700
PANEL_A_THETA = 2.0
Z95 = 1.959964

# printed size: 0.82 x 466.6 pt (cas-sc text column) -> scale exactly 1.0
FIG_W_IN = 382.61 / 72.0
FIG_H_IN = 2.6200

C = {
    "asym":  "#C62828",   # asymptotic chi-square null (inflated)
    "cal":   "#0072B2",   # permutation-calibrated null (restored)
    "nom":   "#37474F",   # nominal level reference line
    "ink":   "#1F2933",   # all text
    "grid":  "#E9E9E9",
    "band":  "#C62828",
}

TITLE_FS = 8.0
AXLAB_FS = 7.5
TICK_FS = 7.5
LEGEND_FS = 7.5
DATA_FS = 6.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": TICK_FS,
    "axes.linewidth": 0.7,
    "axes.edgecolor": C["ink"],
    "axes.labelsize": AXLAB_FS,
    "axes.labelcolor": C["ink"],
    "axes.titlesize": TITLE_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "legend.fontsize": LEGEND_FS,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
})


def wilson(k, n, z=Z95):
    """95 % Wilson score interval for k successes out of n (bounded in [0,1])."""
    if n <= 0:
        return 0.0, 0.0
    p = k / float(n)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load_panel_a():
    """(pi0, type-I, lo, hi, R) for the n=2700, theta=2 sweep."""
    with open(TYPE1_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = [r for r in payload["results"]
            if r["n"] == PANEL_A_N and abs(r["theta"] - PANEL_A_THETA) < 1e-9]
    rows.sort(key=lambda r: r["pi_zero"])
    pi0, err, lo, hi, reps = [], [], [], [], []
    for r in rows:
        R = int(r["R"])
        k = int(round(r["empirical_type1"] * R))
        a, b = wilson(k, R)
        pi0.append(r["pi_zero"])
        err.append(r["empirical_type1"])
        lo.append(a)
        hi.append(b)
        reps.append(R)
    return pi0, err, lo, hi, reps


def load_panel_b():
    """Rows at the calibrated configurations: n, pi0, asym/cal with intervals."""
    with open(CALIB_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = sorted(payload["results"], key=lambda r: -r["n"])
    out = []
    for r in rows:
        R = int(r["R"])
        k_a = int(round(r["type1_asymptotic"] * R))
        k_c = int(round(r["type1_calibrated"] * R))
        a_lo, a_hi = wilson(k_a, R)
        c_lo, c_hi = wilson(k_c, R)
        out.append({
            "n": int(r["n"]), "pi0": float(r["pi0"]), "R": R,
            "asym": r["type1_asymptotic"], "cal": r["type1_calibrated"],
            "asym_lo": a_lo, "asym_hi": a_hi, "cal_lo": c_lo, "cal_hi": c_hi,
            "k_asym": k_a, "k_cal": k_c,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=FIG_DIR,
                    help="output directory (default: ../../figures)")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    pi0_a, err_a, lo_a, hi_a, reps_a = load_panel_a()
    rows_b = load_panel_b()

    # ---------- shared y-limit ----------
    tops = list(hi_a) + [r["asym_hi"] for r in rows_b] + [r["cal_hi"] for r in rows_b]
    ymax = math.ceil((max(tops) + 0.030) / 0.05) * 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_H_IN))
    # bottom leaves room for two-line x tick labels plus the axis title;
    # top leaves room for the shared legend above the two panel titles.
    fig.subplots_adjust(left=0.090, right=0.984, top=0.805, bottom=0.220,
                        wspace=0.28)

    # ================= Panel (a) =================
    ax1.axhspan(ALPHA, ymax, color=C["band"], alpha=0.045, lw=0, zorder=0)
    ax1.errorbar(pi0_a, err_a,
                 yerr=[[e - l for e, l in zip(err_a, lo_a)],
                       [h - e for e, h in zip(err_a, hi_a)]],
                 fmt="o-", color=C["asym"], ecolor=C["asym"], elinewidth=0.7,
                 capsize=1.8, capthick=0.7, markersize=3.2, linewidth=1.1,
                 zorder=3)
    ax1.axhline(ALPHA, color=C["nom"], linestyle=(0, (3.2, 1.8)), linewidth=1.0,
                zorder=2)
    for x, y, h in zip(pi0_a, err_a, hi_a):
        if h < ALPHA + 0.045:
            # the interval top sits on the nominal line: a label above it would
            # be crossed by the dashed line, so put it beside the marker instead
            ax1.annotate("%.3f" % y, (x, y), textcoords="offset points",
                         xytext=(3.2, 0.0), ha="left", va="center",
                         fontsize=DATA_FS, color=C["ink"])
        else:
            ax1.annotate("%.3f" % y, (x, h), textcoords="offset points",
                         xytext=(0, 2.2), ha="center", va="bottom",
                         fontsize=DATA_FS, color=C["ink"])
    ax1.set_xlim(-0.075, 0.99)
    ax1.set_xticks(pi0_a)
    ax1.set_xticklabels(["%.1f" % p for p in pi0_a])
    ax1.set_xlabel("Zero-inflation rate $\\pi_0$")
    ax1.set_ylabel("Empirical type-I error")
    ax1.set_ylim(0, ymax)
    ax1.yaxis.set_major_locator(MultipleLocator(0.10))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.set_title("(a) Asymptotic null ($n{=}2700$)", loc="left", pad=3.0,
                  fontweight="bold")
    ax1.grid(axis="y", color=C["grid"], linewidth=0.5, zorder=0, which="major")
    ax1.set_axisbelow(True)

    # ================= Panel (b) =================
    labels = ["$n{=}2700$\n$\\pi_0{=}0.9$",
              "$n{=}2700$\n$\\pi_0{=}0.7$",
              "$n{=}500$\n$\\pi_0{=}0.9$"]
    xb = list(range(len(rows_b)))
    w = 0.30
    off = w / 2 + 0.015
    asym = [r["asym"] for r in rows_b]
    cal = [r["cal"] for r in rows_b]
    err_a2 = [[r["asym"] - r["asym_lo"] for r in rows_b],
              [r["asym_hi"] - r["asym"] for r in rows_b]]
    err_c2 = [[r["cal"] - r["cal_lo"] for r in rows_b],
              [r["cal_hi"] - r["cal"] for r in rows_b]]

    ax2.axhspan(ALPHA, ymax, color=C["band"], alpha=0.045, lw=0, zorder=0)
    ax2.bar([i - off for i in xb], asym, width=w, color=C["asym"],
            edgecolor=C["ink"], linewidth=0.5, zorder=3,
            label="Asymptotic $\\chi^2_1$")
    ax2.errorbar([i - off for i in xb], asym, yerr=err_a2, fmt="none",
                 ecolor=C["ink"], elinewidth=0.7, capsize=1.8, capthick=0.7,
                 zorder=4)
    ax2.bar([i + off for i in xb], cal, width=w, color=C["cal"],
            edgecolor=C["ink"], linewidth=0.5, zorder=3,
            label="Permutation-calibrated")
    ax2.errorbar([i + off for i in xb], cal, yerr=err_c2, fmt="none",
                 ecolor=C["ink"], elinewidth=0.7, capsize=1.8, capthick=0.7,
                 zorder=4)
    for i, r in enumerate(rows_b):
        ax2.annotate("%.3f" % r["asym"], (i - off, r["asym_hi"]),
                     textcoords="offset points", xytext=(0, 2.2), ha="center",
                     va="bottom", fontsize=DATA_FS, color=C["ink"])
        ax2.annotate("%.3f" % r["cal"], (i + off, r["cal_hi"]),
                     textcoords="offset points", xytext=(0, 2.2), ha="center",
                     va="bottom", fontsize=DATA_FS, color=C["ink"])

    ax2.axhline(ALPHA, color=C["nom"], linestyle=(0, (3.2, 1.8)), linewidth=1.0,
                zorder=2, label="Nominal $\\alpha{=}0.05$")
    ax2.set_xlim(-0.62, len(rows_b) - 0.38)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels)
    for t in ax2.get_xticklabels():
        t.set_linespacing(1.45)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Empirical type-I error")
    ax2.set_ylim(0, ymax)
    ax2.yaxis.set_major_locator(MultipleLocator(0.10))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.set_title("(b) Permutation calibration", loc="left", pad=3.0,
                  fontweight="bold")
    ax2.grid(axis="y", color=C["grid"], linewidth=0.5, zorder=0, which="major")
    ax2.set_axisbelow(True)

    # ---------- shared legend ----------
    order = ["Asymptotic $\\chi^2_1$", "Permutation-calibrated",
             "Nominal $\\alpha{=}0.05$"]
    seen = {}
    for ax in (ax2, ax1):
        for h, nm in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(nm, h)
    names = [n for n in order if n in seen]
    fig.legend([seen[n] for n in names], names, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
               handlelength=1.5, handletextpad=0.5, columnspacing=1.6)

    out_pdf = os.path.join(outdir, "fig3_type1_zinb.pdf")
    out_png = os.path.join(outdir, "fig3_type1_zinb.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)

    # ---------- provenance ----------
    print("panel (a): n=%d theta=%.1f  R=%d" % (PANEL_A_N, PANEL_A_THETA,
                                                reps_a[0]))
    for p, e, l, h in zip(pi0_a, err_a, lo_a, hi_a):
        print("   pi0=%.1f  type-I=%.3f  95%% Wilson [%.3f, %.3f]"
              % (p, e, l, h))
    print("panel (b): R=%d" % rows_b[0]["R"])
    for r in rows_b:
        print("   n=%-5d pi0=%.1f  asym %.3f [%.3f,%.3f] (%d/%d)  "
              "cal %.3f [%.3f,%.3f] (%d/%d)"
              % (r["n"], r["pi0"], r["asym"], r["asym_lo"], r["asym_hi"],
                 r["k_asym"], r["R"], r["cal"], r["cal_lo"], r["cal_hi"],
                 r["k_cal"], r["R"]))
    print("ylim = (0, %.2f)" % ymax)
    print("SAVED", out_pdf, out_png)


if __name__ == "__main__":
    main()
