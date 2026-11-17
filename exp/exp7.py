"""exp_7: Robust pricing over adapted Wasserstein balls and small-dispersion
asymptotics for rating-dependent hazards (Section 4.4, Theorem 4.30, 4.35,
Corollary 4.31, 4.36, Proposition 4.29, 4.32).

Computes the worst-case call premium over the adapted ball U_eta(p) of
calibrated posteriors (Theorem 4.30), checks the extremizer mu_eta* (Eq.
4.56), verifies the sharp mean-absolute-deviation Lipschitz bound
(Proposition 4.32), and runs the small-dispersion Monte Carlo check of the
square-root separation between AW_2 and W_2 for calibrated hazard
perturbations lambda_R^eps = lambda_* + eps*zeta + c_eps (Theorem 4.35,
Corollary 4.36).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP7_DELTA, EXP7_ETAS, EXP7_EPSILONS, EXP7_N_MC_SMALL_DISP, EXP7_P,
    EXP7_STRIKES, EXP7_ZETA_PROBS, EXP7_ZETA_VALUES, RANDOM_SEED,
)
from src.robust_pricing import (
    extremizer_mu, mad_lipschitz_bound, predicted_asymptotics,
    robust_call_premium, robust_premium_above_coarse, small_dispersion_small_sample,
)


def run() -> dict:
    p = EXP7_P
    out: dict = {"p": p}

    # 1. Robust call premium over U_eta(p) and the extremizer (Theorem 4.30, Eq. 4.56)
    rows = []
    for eta in EXP7_ETAS:
        for k in EXP7_STRIKES:
            premium = robust_call_premium(p, eta, k)
            coarse = max(p - k, 0.0)
            above = robust_premium_above_coarse(eta, p, k)
            ext = extremizer_mu(p, eta)
            # the extremizer must itself be calibrated (mean p) and attain eta exactly
            assert abs(ext["mean"] - p) < 1e-10
            assert abs(ext["mad"] - eta) < 1e-10
            ext_price = float(np.sum(ext["weights"] * np.maximum(ext["support"] - k, 0.0)))
            rows.append({
                "eta": float(eta), "k": float(k), "robust_premium": premium,
                "coarse_premium": coarse, "premium_above_coarse": above,
                "extremizer_price_check": ext_price,
                "extremizer_matches_formula": bool(abs(ext_price - premium) < 1e-9),
            })
    out["robust_call_premium"] = rows

    # 2. Sharp mean-absolute-deviation Lipschitz bound (Proposition 4.32), g = |q-p| (L=1)
    g = lambda q: np.abs(q - p)
    mad_rows = []
    for eta in EXP7_ETAS:
        ext = extremizer_mu(p, eta)
        res = mad_lipschitz_bound(g, p, ext["support"], ext["weights"], L=1.0)
        res["eta"] = float(eta)
        mad_rows.append(res)
    out["mad_lipschitz_bound"] = mad_rows

    # 3. Small-dispersion asymptotics: AW_2 ~ eps^{1/2}, W_2 ~ eps (Theorem 4.35, Corollary 4.36)
    e_abs_zeta = float(np.sum(EXP7_ZETA_PROBS * np.abs(EXP7_ZETA_VALUES)))
    e_zeta_sq = float(np.sum(EXP7_ZETA_PROBS * EXP7_ZETA_VALUES ** 2))
    rng = np.random.default_rng(RANDOM_SEED)
    disp_rows = []
    for eps in EXP7_EPSILONS:
        exact = small_dispersion_small_sample(p, EXP7_DELTA, EXP7_ZETA_VALUES, EXP7_ZETA_PROBS, float(eps))
        pred = predicted_asymptotics(p, EXP7_DELTA, e_abs_zeta, e_zeta_sq, float(eps))
        # Monte Carlo cross-check of the exact finite-law computation
        zeta_mc = rng.choice(EXP7_ZETA_VALUES, size=EXP7_N_MC_SMALL_DISP, p=EXP7_ZETA_PROBS)
        mgf = np.mean(np.exp(-EXP7_DELTA * eps * zeta_mc))
        c_eps_mc = np.log(mgf) / EXP7_DELTA
        Q_mc = np.exp(-EXP7_DELTA * (-np.log(p) / EXP7_DELTA + eps * zeta_mc + c_eps_mc))
        aw2_mc = np.sqrt(np.mean(np.abs(Q_mc - p)))
        disp_rows.append({
            "epsilon": float(eps), "aw2_exact": exact["aw2"], "w2_exact": exact["w2"],
            "aw2_predicted": pred["aw2_pred"], "w2_predicted": pred["w2_pred"],
            "ratio_exact": exact["aw2"] / exact["w2"] if exact["w2"] and not np.isnan(exact["w2"]) else float("nan"),
            "ratio_predicted": pred["ratio_pred"], "aw2_mc": float(aw2_mc),
        })
    out["small_dispersion"] = disp_rows

    # Log-log regression of AW2/W2 ratio vs epsilon to check the eps^{-1/2} power law
    eps_arr = np.array([r["epsilon"] for r in disp_rows])
    ratio_arr = np.array([r["ratio_exact"] for r in disp_rows])
    valid = np.isfinite(ratio_arr) & (ratio_arr > 0)
    slope, intercept = np.polyfit(np.log(eps_arr[valid]), np.log(ratio_arr[valid]), 1)
    out["loglog_slope_ratio_vs_epsilon"] = float(slope)  # expected approx -0.5 (Corollary 4.36)

    with open("results/exp7_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ks = np.array([r["k"] for r in rows if r["eta"] == float(EXP7_ETAS[-1])])
    prem = np.array([r["robust_premium"] for r in rows if r["eta"] == float(EXP7_ETAS[-1])])
    coarse = np.array([r["coarse_premium"] for r in rows if r["eta"] == float(EXP7_ETAS[-1])])
    axes[0].plot(ks, prem, "o-", label=f"robust, eta={EXP7_ETAS[-1]:.3f}")
    axes[0].plot(ks, coarse, "s--", label="coarse (p-k)+")
    axes[0].set_xlabel("strike k")
    axes[0].set_ylabel("call premium")
    axes[0].set_title("Worst-case call premium over the adapted ball")
    axes[0].legend()

    aw2s = np.array([r["aw2_exact"] for r in disp_rows])
    w2s = np.array([r["w2_exact"] for r in disp_rows])
    axes[1].loglog(eps_arr, aw2s, "o-", label="AW_2")
    axes[1].loglog(eps_arr, w2s, "s-", label="W_2")
    axes[1].loglog(eps_arr, eps_arr ** 0.5 * aw2s[-1] / eps_arr[-1] ** 0.5, "k--", label="slope 1/2")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("distance")
    axes[1].set_title("Small-dispersion asymptotics: AW_2 vs W_2")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("results/exp7_robust_pricing.png", dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = run()
    print(f"exp_7: p={result['p']}")
    print("  all extremizers match closed-form premium:",
          all(r["extremizer_matches_formula"] for r in result["robust_call_premium"]))
    print("  all MAD Lipschitz bounds hold:", all(r["bound_holds"] for r in result["mad_lipschitz_bound"]))
    print(f"  log-log slope of AW2/W2 vs epsilon: {result['loglog_slope_ratio_vs_epsilon']:.3f} (expect ~ -0.5)")
