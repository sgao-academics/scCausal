# -*- coding: utf-8 -*-
"""
Monte-Carlo (permutation) null calibration for the NB-LR test under zero-inflation.

Question: Remark 1 of the paper proposes, for small n / higher-order conditioning /
zero-inflated regimes, to calibrate the null distribution by permutation rather than
relying on the asymptotic chi^2_1. Does that actually fix the inflated Type-I error?

Design:
  - Data generated from ZINB (pi0=0.9, theta=2.0) -- the severe dropout regime where
    the asymptotic chi^2_1 over-rejects (Type-I ~ 0.25).
  - For each replicate: compute observed LR Lambda_obs (null+alt NB fits).
  - Build permutation null: B times permute X_i against Y, refit, obtain Lambda^b.
    p_calib = (1 + #{Lambda^b >= Lambda_obs}) / (B+1).
  - Reject when p_calib < alpha. Measure empirical Type-I error of the calibrated test.
  - Compare with asymptotic p-value (chi^2_1) rejection on the SAME data.

Reproducible. torch-cuda128 env.
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from scipy import stats as st
import json, os

ALPHA = 0.05
R = 60          # outer replicates
B = 200         # permutations per replicate


def gen_zinb(n, mu, theta, pi_zero, rng):
    z = (rng.rand(n) < pi_zero).astype(float)
    p = theta / (theta + mu)
    cnt = rng.negative_binomial(theta, p, size=n)
    return cnt * (1 - z)


def nb_llf(y, X):
    try:
        m = sm.GLM(y, X, family=NegativeBinomial()).fit(maxiter=100, disp=0)
        return m.llf if m.llf is not None else -1e12
    except Exception:
        return -1e12


def lambda_stat(y, X_null, X_alt):
    ll0 = nb_llf(y, X_null)
    ll1 = nb_llf(y, X_alt)
    return max(2 * (ll1 - ll0), 0.0)


def run(n, pi0, theta, R, B, seed_base):
    rng_seed = np.random.RandomState(seed_base)
    xS = rng_seed.normal(size=n)
    xI = rng_seed.normal(size=n)
    X_null = np.column_stack([np.ones(n), xS])
    X_alt = np.column_stack([np.ones(n), xS, xI])
    mu = np.exp(0.5 + 0.4 * xS)

    reject_asym = 0.0
    reject_calib = 0.0
    valid = 0
    for r in range(R):
        y = gen_zinb(n, mu, theta, pi0, np.random.RandomState(seed_base + 1000 + r))
        lam_obs = lambda_stat(y, X_null, X_alt)
        # asymptotic p
        p_asym = st.chi2.sf(lam_obs, df=1)
        # permutation null: shuffle the X_i vector (breaks any X_i-y association),
        # keeping y and X_S fixed -- a valid Monte-Carlo null for the LR statistic.
        lam_b = []
        rngp = np.random.RandomState(seed_base + 500000)
        for b in range(B):
            xI_perm = rngp.permutation(xI)
            Xa = np.column_stack([np.ones(n), xS, xI_perm])
            lam_b.append(lambda_stat(y, X_null, Xa))
        lam_b = np.array(lam_b)
        p_calib = (1 + np.sum(lam_b >= lam_obs)) / (B + 1)
        if not np.isnan(p_asym):
            valid += 1
            if p_asym < ALPHA:
                reject_asym += 1.0
            if p_calib < ALPHA:
                reject_calib += 1.0
    return {"n": n, "pi0": pi0, "theta": theta, "alpha": ALPHA,
            "R": R, "B": B, "valid": valid,
            "type1_asymptotic": (reject_asym / valid if valid else float('nan')),
            "type1_calibrated": (reject_calib / valid if valid else float('nan'))}


def main():
    configs = [(2700, 0.9, 2.0), (2700, 0.7, 2.0), (500, 0.9, 2.0)]
    results = []
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(base, exist_ok=True)
    for idx, (n, pi0, th) in enumerate(configs):
        res = run(n, pi0, th, R, B, seed_base=3000 + idx)
        results.append(res)
        print("n=%d pi0=%.1f  asym=%.4f  calib=%.4f" %
              (n, pi0, res["type1_asymptotic"], res["type1_calibrated"]))
    out = os.path.join(base, "zinb_calib_v2_results.json")
    json.dump({"ALPHA": ALPHA, "results": results}, open(out, 'w', encoding='utf-8'), indent=2)
    print("Saved:", out)


if __name__ == "__main__":
    main()
