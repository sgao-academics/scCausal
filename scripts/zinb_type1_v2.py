# -*- coding: utf-8 -*-
"""
STRICT Type-I error simulation for NB-LR conditional independence test.

Design (methodologically clean):
  - A conditioning variable X_S drives the target Y via log(mu) = b0 + bS*X_S
    (so X_S is a genuine predictor), guaranteeing X_i is only *conditionally*
    independent of Y given X_S under H0 (beta_i = 0).
  - X_i is generated independent of X_S (hence of Y conditional on X_S).
  - We compare THREE generative mechanisms for Y:
       (M0) plain NB      -> model is correctly specified  (baseline, expect ~alpha)
       (M1) ZINB moderate (pi=0.5, 0.7)
       (M2) ZINB severe   (pi=0.9)  -> scRNA-seq dropout regime
  - If the model were correctly specified (M0), the empirical rejection rate
    should track alpha=0.05. Any upward drift in M1/M2 reveals how much the
    NB-LR test over-rejects under zero-inflation (the audit concern).

Reproducible. Runs on torch-cuda128 env (statsmodels 0.14.6).
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from scipy import stats as st
import itertools, json, os

ALPHA = 0.05
R = 200


def gen_nb(n, mu, theta, rng):
    p = theta / (theta + mu)
    return rng.negative_binomial(theta, p, size=n)


def gen_zinb(n, mu, theta, pi_zero, rng):
    z = (rng.rand(n) < pi_zero).astype(float)
    cnt = gen_nb(n, mu, theta, rng)
    return cnt * (1 - z)


def nb_lr_pval(y, X_null, X_alt):
    try:
        m0 = sm.GLM(y, X_null, family=NegativeBinomial()).fit(maxiter=100, disp=0)
        ll0 = m0.llf if m0.llf is not None else -1e12
        m1 = sm.GLM(y, X_alt, family=NegativeBinomial()).fit(maxiter=100, disp=0)
        ll1 = m1.llf if m1.llf is not None else -1e12
        lam = max(2.0 * (ll1 - ll0), 0.0)
        return float(st.chi2.sf(lam, df=1))
    except Exception:
        return float('nan')


def run(n, mechanism, theta, pi_zero, R, seed_base):
    """mechanism in {'nb','zinb'}."""
    xS = np.random.RandomState(seed_base).normal(size=n)
    xI = np.random.RandomState(seed_base + 100000).normal(size=n)
    bS = 0.4          # X_S -> Y effect (log-mu), nonzero => conditional test is meaningful
    X_null = np.column_stack([np.ones(n), xS])
    X_alt = np.column_stack([np.ones(n), xS, xI])
    rej = 0.0
    valid = 0
    for r in range(R):
        # mu depends on xS (conditional structure), so H0 means xI independent given xS
        mu = np.exp(0.5 + bS * xS)
        rng_r = np.random.RandomState(seed_base + 1000 + r)
        if mechanism == 'nb':
            y = gen_nb(n, mu, theta, rng_r)
        else:
            y = gen_zinb(n, mu, theta, pi_zero, rng_r)
        pv = nb_lr_pval(y, X_null, X_alt)
        if not np.isnan(pv):
            valid += 1
            if pv < ALPHA:
                rej += 1.0
    return {"n": n, "mechanism": mechanism, "theta": theta,
            "pi_zero": (pi_zero if mechanism == 'zinb' else 0.0),
            "alpha": ALPHA, "R": R, "valid": valid,
            "empirical_type1": (rej / valid if valid else float('nan'))}


def main():
    configs = []
    # baseline: correctly specified plain NB (should be ~alpha)
    for n, th in itertools.product([2700, 500], [2.0, 5.0]):
        configs.append(("nb", n, th, 0.0))
    # zero-inflated
    for n, th, pi0 in itertools.product([2700, 500], [2.0, 5.0], [0.5, 0.7, 0.9]):
        configs.append(("zinb", n, th, pi0))

    results = []
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "zinb_type1_v2_results.json")
    for idx, (mech, n, th, pi0) in enumerate(configs):
        res = run(n, mech, th, pi0, R, seed_base=2000 + idx)
        results.append(res)
        print("mech=%-5s n=%-4d th=%.1f pi0=%.1f -> TypeI=%.4f" %
              (res["mechanism"], n, th, pi0, res["empirical_type1"]))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"alpha": ALPHA, "R": R, "results": results}, f, indent=2)
    print("\nSaved:", out_path)
    print("=== SUMMARY ===")
    for r in results:
        dev = r["empirical_type1"] - r["alpha"]
        print("%-5s n=%-4d th=%.1f pi0=%.1f  type1=%.4f dev=%+.4f" %
              (r["mechanism"], r["n"], r["theta"], r["pi_zero"], r["empirical_type1"], dev))


if __name__ == "__main__":
    main()
