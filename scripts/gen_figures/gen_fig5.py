"""Fig 5: Type-I error of the NB-LR test under zero-inflation.

Panel (a): asymptotic chi-square Type-I error vs the zero-inflation rate pi0,
           for n=2700 and theta=2.0.
Panel (b): asymptotic vs permutation-calibrated Type-I error (Monte-Carlo null,
           B=200) at the three configurations where calibration was run.

Both panels are read from the experiment outputs rather than hardcoded:
    ../zinb_type1_v2_results.json    -> panel (a)
    ../zinb_calib_v2_results.json    -> panel (b)

Usage:
    python gen_fig5.py                 # write into ../../figures/
    python gen_fig5.py --outdir DIR    # write somewhere else (safe preview)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
})


def load_panel_a():
    """(pi0, empirical type-I) for the n=2700, theta=2 sweep."""
    with open(TYPE1_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = [r for r in payload["results"]
            if r["n"] == PANEL_A_N and abs(r["theta"] - PANEL_A_THETA) < 1e-9]
    rows.sort(key=lambda r: r["pi_zero"])
    return [r["pi_zero"] for r in rows], [r["empirical_type1"] for r in rows]


def load_panel_b():
    """(label, asymptotic, calibrated) at the calibrated configurations.

    The calibrated rates come from finite simulation counts (60 replicates,
    B=200), so they are stored as raw fractions (13/60 = 0.21666...). The
    published figure plots them at three decimals; the display rounding is
    applied here so that the regenerated figure matches it exactly.
    """
    with open(CALIB_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = sorted(payload["results"], key=lambda r: -r["n"])
    # 'n=%d,' padded to 8 characters keeps the pi0 fields aligned under the
    # x-axis rotation, as in the published figure.
    labels = [("n=%d," % r["n"]).ljust(8) + "pi0=%.1f" % r["pi0"] for r in rows]
    asym = [round(r["type1_asymptotic"], 3) for r in rows]
    cal = [round(r["type1_calibrated"], 3) for r in rows]
    return labels, asym, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=FIG_DIR,
                    help="output directory (default: ../../figures)")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    pi0_a, type1_a = load_panel_a()
    labels, asym, cal = load_panel_b()
    print("panel (a): %d points" % len(pi0_a))
    print("panel (b): %d configurations" % len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0))

    # ---------- Panel (a) ----------
    x = np.arange(len(pi0_a))
    ax1.bar(x, type1_a, width=0.62, color="#4c72b0", edgecolor="black",
            linewidth=0.6, label="Asymptotic $\\chi^2_1$")
    ax1.axhline(ALPHA, color="crimson", linestyle="--", linewidth=1.2,
                label="Nominal $\\alpha=0.05$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["$%.1f$" % p for p in pi0_a])
    ax1.set_xlabel("Zero-inflation rate $\\pi_0$")
    ax1.set_ylabel("Empirical type-I error")
    ax1.set_ylim(0, 0.35)
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax1.set_title("(a) Asymptotic $\\chi^2$ null", fontsize=9.5)
    ax1.legend(frameon=False, loc="upper left", handlelength=1.4)
    for xi, yi in zip(x, type1_a):
        ax1.text(xi, yi + 0.008, "%.2f" % yi, ha="center", fontsize=7.5)

    # ---------- Panel (b) ----------
    x2 = np.arange(len(labels))
    w = 0.36
    ax2.bar(x2 - w / 2, asym, width=w, color="#c44e52", edgecolor="black",
            linewidth=0.6, label="Asymptotic $\\chi^2_1$")
    ax2.bar(x2 + w / 2, cal, width=w, color="#55a868", edgecolor="black",
            linewidth=0.6, label="Permutation-calibrated")
    ax2.axhline(ALPHA, color="crimson", linestyle="--", linewidth=1.2,
                label="Nominal $\\alpha$")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=12, ha="right", fontsize=7.2)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Empirical type-I error")
    ax2.set_ylim(0, 0.35)
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.set_title("(b) Monte-Carlo null calibration", fontsize=9.5)
    ax2.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=7.2)
    for xi, (a, c) in enumerate(zip(asym, cal)):
        ax2.text(xi - w / 2, a + 0.008, "%.2f" % a, ha="center", fontsize=7)
        ax2.text(xi + w / 2, c + 0.008, "%.2f" % c, ha="center", fontsize=7)

    fig.tight_layout()
    out_pdf = os.path.join(outdir, "fig5_type1_zinb.pdf")
    out_png = os.path.join(outdir, "fig5_type1_zinb.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print("SAVED", out_pdf, out_png)


if __name__ == "__main__":
    main()
