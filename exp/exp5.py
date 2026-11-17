"""exp_5: model-transfer hedging in a single calibrated rating-review bucket
(Sec. 4.5-4.6): bootstrap roll error, the exact identities of Theorem 4.16 /
4.32, the survival-filtered projection of Proposition 4.37, and the
claim-weighted bound of Theorem 4.23, illustrated on Example 4.22.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src.bootstrap_hedge import (
    calibrate_bootstrap,
    claim_weighted_bound,
    claim_weighted_residual,
    expected_psi,
    roll_residual,
    survival_filtered_projection,
)
from src.cox_hazard import psi_from_paths

RESULTS_DIR = "results"


def run_bootstrap_roll() -> dict:
    t1, T, xi = cfg.EXP5_T1, cfg.EXP5_T, cfg.EXP5_XI
    hazards_L, probs = cfg.EXP5_HAZARDS_L, cfg.EXP5_PROBS
    roll = roll_residual(hazards_L, probs, t1, T, xi)
    e_psi_flat = expected_psi(hazards_L, probs, t1, T, xi, roll["B_T"])
    return {**roll, **{f"flat_{k}": v for k, v in e_psi_flat.items()}}


def run_filtered_projection(roll: dict) -> dict:
    """E[Psi_T(a, b^#)] under the survival-filtered hazard b^# (Prop. 4.37),
    compared with the flat bootstrap: the filtered projection reproduces the
    same aggregate survival curve exactly but, because its flat hazards cross
    the time-varying filtered hazard, generates a strictly positive
    adaptation slack (Example 4.22)."""
    t1, T, xi = cfg.EXP5_T1, cfg.EXP5_T, cfg.EXP5_XI
    hazards_L, probs = cfg.EXP5_HAZARDS_L, cfg.EXP5_PROBS
    proj = survival_filtered_projection(hazards_L, probs, t1, T, xi, n_grid=4000)
    u, h_u = proj["u"], proj["h_u"]
    # crossing time with the middle hazard
    mid = hazards_L[1]
    sign = np.sign(h_u - mid)
    crossings = u[1:][np.diff(sign) != 0]
    crossing_time = float(crossings[0]) if len(crossings) else None

    e_psi_filtered = 0.0
    for L, prob in zip(hazards_L, probs):
        a_grid = np.full_like(u, L)
        e_psi_filtered += prob * psi_from_paths(a_grid, h_u, u)
    slack = e_psi_filtered - roll["flat_E_Psi"]
    return {
        "crossing_time": crossing_time,
        "h_t1": float(h_u[0]),
        "h_T": float(h_u[-1]),
        "S_T_check": float(proj["S_u"][-1]),
        "S_from_h_check": float(proj["S_from_h"][-1]),
        "E_Psi_filtered": e_psi_filtered,
        "adaptation_slack": slack,
    }, proj


def run_claim_weighted_bound(roll: dict) -> dict:
    """A simple two-date linear claim with cash flows at the mid-bucket date
    and at maturity T, comparing the bootstrap and rating-dependent survival
    curves (Theorem 4.23)."""
    t1, T, xi = cfg.EXP5_T1, cfg.EXP5_T, cfg.EXP5_XI
    hazards_L, probs = cfg.EXP5_HAZARDS_L, cfg.EXP5_PROBS
    maturities = np.array([T / 2, T])
    nu_weights = np.array([0.5, 0.5])
    beta = roll["beta"]
    P_t1_A = np.array([float(np.sum(probs * np.exp(-(xi + hazards_L * (u - t1))))) for u in maturities])
    P_t1_B = np.array([float(np.exp(-(xi + beta * (u - t1)))) for u in maturities])
    residual = claim_weighted_residual(P_t1_A, P_t1_B, nu_weights)
    bound = claim_weighted_bound(hazards_L, probs, t1, maturities, xi, beta, np.abs(nu_weights))
    return {
        "maturities": maturities, "nu_weights": nu_weights,
        "P_t1_A": P_t1_A, "P_t1_B": P_t1_B,
        "abs_residual": abs(residual), "bound": bound,
        "bound_holds": bool(abs(residual) <= bound + 1e-9),
    }


def run() -> dict:
    roll = run_bootstrap_roll()
    filtered, proj = run_filtered_projection(roll)
    claim = run_claim_weighted_bound(roll)
    return {"roll": roll, "filtered": filtered, "claim": claim}, proj


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def save_results(result: dict, proj: dict) -> None:
    with open(f"{RESULTS_DIR}/exp5_results.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(proj["u"], proj["h_u"], label=r"filtered hazard $h_u$ (Prop. 4.37)")
    for L in cfg.EXP5_HAZARDS_L:
        ax.axhline(L, linestyle="--", alpha=0.4)
    ax.set_xlabel("u (years)")
    ax.set_ylabel("hazard rate")
    ax.set_title("Example 4.22: survival-filtered hazard vs flat post-review hazards")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp5_filtered_hazard.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("exp_5: single-bucket model-transfer hedging (Example 4.22)")
    result, proj = run()
    roll = result["roll"]
    print(f"  F = {roll['F']:.6f}, beta = {roll['beta']:.6f}")
    print(f"  E[1_tau>t1 |epsilon|] = {roll['E_abs_epsilon']:.6f} = phi_T = {roll['phi_T']:.6f}")
    print(f"  flat bootstrap E[Psi_T] = {roll['flat_E_Psi']:.6f} (should equal phi_T: {roll['phi_T']:.6f})")
    filt = result["filtered"]
    print(f"  crossing time = {filt['crossing_time']}, filtered E[Psi_T] = {filt['E_Psi_filtered']:.6f}, "
          f"slack = {filt['adaptation_slack']:.6f}")
    claim = result["claim"]
    print(f"  claim-weighted: |residual| = {claim['abs_residual']:.6f} <= bound = {claim['bound']:.6f} "
          f"(holds={claim['bound_holds']})")
    save_results(result, proj)
    print("  saved results/exp5_results.json, exp5_filtered_hazard.png")


if __name__ == "__main__":
    main()
