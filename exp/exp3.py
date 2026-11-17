"""exp_3: discrete-time hedging -- momentum payoff replication and the
stability bound for model-transfer hedging of defaultable bond prices
(Sec. 3.4, Theorem 3.19, Corollary 3.20, Proposition 3.22).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import bernoulli_given_posterior, two_point_posterior
from src import config as cfg
from src.momentum_hedge import (
    hedge_size_identities,
    l2_hedging_error,
    replicate_momentum_coarse,
    replicate_momentum_rating,
    stability_bound,
)
from src.single_review import aw2_squared_single_review, w2_squared_exact
from src.transition_utils import adapted_distance_single_review

RESULTS_DIR = "results"


def run_momentum_replication(rng: np.random.Generator) -> dict:
    p = cfg.EXP3_P
    n = cfg.EXP3_N_MC

    # Coarse model P^0: S0=S1=p, S2=D~Bernoulli(p).
    D = bernoulli_given_posterior(np.full(n, p), rng)
    gamma_coarse = replicate_momentum_coarse(p, D)
    max_abs_residual_coarse = float(np.max(np.abs(gamma_coarse)))

    # One-rating model P^mu: Q = p +/- delta (Corollary 3.9's symmetric law).
    Q = two_point_posterior(p, cfg.EXP3_MU_DELTA, n, rng)
    D_mu = bernoulli_given_posterior(Q, rng)
    rep = replicate_momentum_rating(p, Q, D_mu)
    max_abs_replication_error = float(np.max(np.abs(rep["value"] - rep["gamma"])))

    identities = hedge_size_identities(Q, p)
    aw2_sq_formula = aw2_squared_single_review(Q, p)
    w2_exact = w2_squared_exact(Q, p)

    return {
        "p": p, "n_mc": n, "delta": cfg.EXP3_MU_DELTA,
        "max_abs_residual_coarse_model": max_abs_residual_coarse,
        "max_abs_replication_error_rating_model": max_abs_replication_error,
        "E_abs_H2": identities["E|H2|"],
        "sqrt_E_H2_sq": identities["sqrt(E|H2|^2)"],
        "AW2_sq_from_H2": identities["AW2_sq"],
        "AW2_sq_closed_form": aw2_sq_formula,
        "W2_from_identities": identities["W2"],
        "W2_closed_form": np.sqrt(w2_exact),
    }


def run_stability_bound() -> dict:
    p = cfg.EXP3_P
    mu_support, mu_probs = cfg.EXP3_STABILITY_MU_SUPPORT, cfg.EXP3_STABILITY_MU_PROBS
    nu_support, nu_probs = cfg.EXP3_STABILITY_NU_SUPPORT, cfg.EXP3_STABILITY_NU_PROBS
    assert abs(float(np.sum(mu_probs * mu_support)) - p) < 1e-10
    assert abs(float(np.sum(nu_probs * nu_support)) - p) < 1e-10

    aw2_sq = adapted_distance_single_review(mu_support, mu_probs, nu_support, nu_probs)
    aw2 = float(np.sqrt(aw2_sq))

    # Claim: the rating-date straddle G = |S1 - S0| = |Q - p| (page 23), fully
    # revealed at the review date. Variance-optimal hedge H1 computed under
    # the second (recalibrated) model nu, H2 = 0 since the claim carries no
    # further second-period randomness (Sec. 3.4).
    G_nu = np.abs(nu_support - p)
    dS1_nu = nu_support - p
    cov = float(np.sum(nu_probs * (G_nu - np.sum(nu_probs * G_nu)) * dS1_nu))
    var = float(np.sum(nu_probs * dS1_nu ** 2))
    H1 = cov / var
    m = float(np.sum(nu_probs * G_nu)) - H1 * float(np.sum(nu_probs * dS1_nu))

    def l2_error(support: np.ndarray, probs: np.ndarray) -> float:
        G = np.abs(support - p)
        dS1 = support - p
        residual = G - m - H1 * dS1
        return float(np.sqrt(np.sum(probs * residual ** 2)))

    R_P = l2_error(mu_support, mu_probs)
    R_Q = l2_error(nu_support, nu_probs)
    realised_change = abs(R_P - R_Q)

    L_G = 1.0  # G=|S1-S0| is 1-Lipschitz in the first increment, 0-Lipschitz in the second
    K = abs(H1)  # sup|H_t|; H2 = 0
    L_H = 0.0  # H1, H2 are constants (not functions of path/rating), so the strategy is trivially 1-Lipschitz-transferable
    c_N = float(np.sqrt(2 * 1 / 2))  # c_N = sqrt(N(N-1)/2), N=2
    bound = stability_bound(L_G, K, c_N, L_H, aw2)

    return {
        "AW2_sq": aw2_sq, "AW2": aw2, "H1_under_nu": H1, "m": m,
        "R_P_mu": R_P, "R_Q_nu": R_Q, "realised_change": realised_change,
        "L_G": L_G, "K": K, "c_N": c_N, "L_H": L_H, "stability_bound": bound,
        "bound_holds": bool(realised_change <= bound + 1e-9),
    }


def run() -> dict:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    momentum = run_momentum_replication(rng)
    stability = run_stability_bound()
    return {"momentum": momentum, "stability": stability}


def save_results(result: dict) -> None:
    with open(f"{RESULTS_DIR}/exp3_results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    stability = result["stability"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["realised |R_P - R_Q|", "Prop. 3.22 bound"],
           [stability["realised_change"], stability["stability_bound"]],
           color=["tab:blue", "tab:orange"])
    ax.set_ylabel("hedging-error change")
    ax.set_title("Proposition 3.22: stability of the transferred hedge")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_stability_bound.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("exp_3: momentum payoff replication and hedge-stability bound")
    result = run()
    m = result["momentum"]
    print(f"  coarse model: max|Gamma residual| = {m['max_abs_residual_coarse_model']:.3e} (should be 0)")
    print(f"  rating model: max|replication error| = {m['max_abs_replication_error_rating_model']:.3e} (should be 0)")
    print(f"  E|H2| = {m['E_abs_H2']:.6f} vs AW2_sq closed form = {m['AW2_sq_closed_form']:.6f}")
    s = result["stability"]
    print(f"  AW2(mu,nu) = {s['AW2']:.6f}; realised |R_P-R_Q| = {s['realised_change']:.6f}; "
          f"bound = {s['stability_bound']:.6f}; holds = {s['bound_holds']}")
    save_results(result)
    print("  saved results/exp3_results.json, exp3_stability_bound.png")


if __name__ == "__main__":
    main()
