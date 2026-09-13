"""
Type-I error of the NB-LR test: fixed alpha=1.0 vs the method-of-moments
dispersion that scCausal actually ships.

Motivation
----------
The manuscript's Type-I experiment (Table 3 / Figure 3) fits every NB GLM with
the statsmodels default, alpha = 1.0, while the method proposes a moment
estimator.  A reviewer is entitled to ask what the size of the *shipped* method
is under the same zero-inflation grid.

The moment estimator reacts to zero-inflation in a predictable way.  For a
ZINB(pi0) with NB2 mean mu and dispersion alpha, the marginal moments give

    alpha_hat = (Var - mean) / mean^2 = (alpha + pi0) / (1 - pi0),

so heavy dropout *inflates* the estimated dispersion (alpha=0.5, pi0=0.9 ->
alpha_hat ~ 14).  The fixed alpha=1.0 control therefore under-states the true
overdispersion precisely in the regime where it matters, and the question is
whether the moment estimator repairs the inflation of Section 3.4 or makes it
worse.

Design: identical generator, null, and grid to scripts/zinb_type1_v2.py, so the
numbers are directly comparable to Table 3 Panel (a).  Each replicate is scored
three ways on the SAME data:
    fixed  : alpha = 1.0                      (the existing control)
    moment : alpha = alpha_hat(y)             (what scCausal ships)
    oracle : alpha = the generating value     (upper bound, not deployable)

Writes results/zinb_type1_moment.json.

Usage
-----
    python scripts/zinb_type1_moment.py --reps 200
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats as st
from statsmodels.genmod.families import NegativeBinomial

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))
ALPHA_SIG = 0.05
FIXED_ALPHA = 1.0


def gen_nb(n, mu, theta, rng):
    p = theta / (theta + mu)
    return rng.negative_binomial(theta, p, size=n)


def gen_zinb(n, mu, theta, pi_zero, rng):
    z = (rng.random(n) < pi_zero)
    return gen_nb(n, mu, theta, rng) * (1 - z)


def moment_alpha(y):
    mu = y.mean()
    var = y.var()
    if mu <= 0:
        return 1e-4
    return float(max((var - mu) / (mu ** 2), 1e-4))


def lr_pval(y, Xn, Xa, alpha_hat):
    try:
        fam = NegativeBinomial(alpha=alpha_hat)
        m0 = sm.GLM(y, Xn, family=fam).fit(maxiter=100, disp=0)
        ll0 = m0.llf if m0.llf is not None else -1e12
        m1 = sm.GLM(y, Xa, family=fam).fit(maxiter=100, disp=0)
        ll1 = m1.llf if m1.llf is not None else -1e12
        lam = max(2.0 * (ll1 - ll0), 0.0)
        return float(st.chi2.sf(lam, df=1))
    except Exception:
        return float("nan")


def run(n, theta, pi_zero, reps, seed_base):
    rng_state = np.random.RandomState(seed_base)
    xS = rng_state.normal(size=n)
    xI = rng_state.normal(size=n)
    bS = 0.4
    Xn = np.column_stack([np.ones(n), xS])
    Xa = np.column_stack([np.ones(n), xS, xI])
    mu = np.exp(0.5 + bS * xS)

    oracle_alpha = 1.0 / theta
    rej = {"fixed": 0.0, "moment": 0.0, "oracle": 0.0}
    a_hats = []
    valid = 0
    for r in range(reps):
        rng_r = np.random.RandomState(seed_base + 1000 + r)
        y = gen_nb(n, mu, theta, rng_r) if pi_zero == 0 else gen_zinb(n, mu, theta, pi_zero, rng_r)
        a_hat = moment_alpha(y)
        a_hats.append(a_hat)
        ok = False
        for name, a in (("fixed", FIXED_ALPHA), ("moment", a_hat), ("oracle", oracle_alpha)):
            pv = lr_pval(y, Xn, Xa, a)
            if not np.isnan(pv):
                ok = True
                if pv < ALPHA_SIG:
                    rej[name] += 1.0
        if ok:
            valid += 1
    out = {
        "n": n, "theta": theta, "pi0": pi_zero, "reps": reps, "valid": valid,
        "alpha_true": round(oracle_alpha, 4),
        "alpha_hat_mean": round(float(np.mean(a_hats)), 4),
        "alpha_hat_median": round(float(np.median(a_hats)), 4),
        "type1_fixed": round(rej["fixed"] / max(valid, 1), 4),
        "type1_moment": round(rej["moment"] / max(valid, 1), 4),
        "type1_oracle": round(rej["oracle"] / max(valid, 1), 4),
    }
    print("  n=%-5d theta=%.0f pi0=%.1f | a_hat~%.2f (true %.2f) | TypeI  fixed=%.3f  moment=%.3f  oracle=%.3f"
          % (n, theta, pi_zero, out["alpha_hat_median"], out["alpha_true"],
             out["type1_fixed"], out["type1_moment"], out["type1_oracle"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    reps = 20 if args.quick else args.reps

    results = []
    t0 = time.time()
    print("=== Type-I: fixed alpha=1.0 vs moment alpha_hat vs oracle (nominal 0.05) ===")
    idx = 0
    for theta in (2.0,):
        for n in (2700, 500):
            for pi0 in (0.0, 0.5, 0.7, 0.9):
                results.append(run(n, theta, pi0, reps, seed_base=5000 + 97 * idx))
                idx += 1
    path = os.path.join(RESULT_DIR, "zinb_type1_moment.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"alpha_sig": ALPHA_SIG, "fixed_alpha": FIXED_ALPHA,
                   "reps": reps, "results": results}, fh, indent=2)
    print("\nwall: %.0fs   saved: %s" % (time.time() - t0, path))


if __name__ == "__main__":
    main()
