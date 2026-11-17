"""exp_4: adapted Wasserstein distance between Cox bond prices with
deterministic hazards -- closed-form mismatch, common-shock coupling, sharp
hazard envelope, L^1 stability and the adaptation-slack decomposition
(Sec. 4.1-4.3, Theorems 4.3, 4.7, 4.9, 4.11, Propositions 4.6, Corollary 4.8).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src.cox_hazard import (
    adaptation_slack_decomposition,
    common_shock_coupling_mismatch,
    cumulative_hazard,
    default_layer_reduction,
    equal_calibration_range,
    l1_stability_check,
    psi_from_paths,
    sharp_range,
    simulate_cir,
)

RESULTS_DIR = "results"


def run_closed_form_and_coupling(rng: np.random.Generator) -> dict:
    T = cfg.EXP4_T
    n_grid = 2001
    t = np.linspace(0.0, T, n_grid)
    a = np.full(n_grid, cfg.EXP4_HA / T)  # flat hazard with integrated hazard h_a
    b = np.full(n_grid, cfg.EXP4_HB / T)
    psi_closed_form = psi_from_paths(a, b, t)
    psi_coupling = common_shock_coupling_mismatch(a, b, t, cfg.EXP4_COMMON_SHOCK_PATHS, rng)

    A_t = cumulative_hazard(a, t)
    B_t = cumulative_hazard(b, t)
    M_t = cumulative_hazard(np.maximum(a, b), t)
    slack = adaptation_slack_decomposition(A_t, B_t, M_t)

    return {
        "T": T, "h_a": cfg.EXP4_HA, "h_b": cfg.EXP4_HB,
        "psi_closed_form": psi_closed_form,
        "psi_common_shock_coupling_mc": psi_coupling,
        "abs_diff": abs(psi_closed_form - psi_coupling),
        "slack_T": float(slack["slack"][-1]),
        "phi_T": float(slack["phi_t"][-1]),
    }


def run_sharp_range() -> dict:
    T, L = cfg.EXP4_T, cfg.EXP4_HAZARD_CAP_L
    h_a, h_b = cfg.EXP4_HA, cfg.EXP4_HB
    lower, upper = sharp_range(h_a, h_b, L, T)
    eq_lower, eq_upper = equal_calibration_range(h_a, L, T)
    return {
        "T": T, "L": L, "h_a": h_a, "h_b": h_b,
        "range_lower": lower, "range_upper": upper,
        "equal_calibration_h": h_a, "equal_calibration_lower": eq_lower,
        "equal_calibration_upper": eq_upper,
    }


def run_stability_sharpness() -> list[dict]:
    T = cfg.EXP4_T
    n_grid = 2001
    t = np.linspace(0.0, T, n_grid)
    b = np.zeros(n_grid)
    out = []
    for eps in cfg.EXP4_STABILITY_EPS:
        a = np.zeros(n_grid)
        a_tilde = np.full(n_grid, eps / T)
        check = l1_stability_check(a, b, a_tilde, t)
        out.append({"epsilon": float(eps), **check, "ratio_lhs_over_eps": check["lhs"] / eps})
    return out


def run_random_intensity(rng: np.random.Generator) -> dict:
    T, dt = cfg.EXP4_T, cfg.EXP4_CIR_DT
    t, lam_a = simulate_cir(cfg.EXP4_CIR_KAPPA, cfg.EXP4_CIR_THETA, cfg.EXP4_CIR_LAMBDA0,
                             cfg.EXP4_CIR_SIGMA, T, dt, cfg.EXP4_CIR_PATHS, rng)
    _, lam_b = simulate_cir(cfg.EXP4_CIR_KAPPA, cfg.EXP4_CIR_THETA, cfg.EXP4_CIR_LAMBDA0,
                             cfg.EXP4_CIR_SIGMA * 1.5, T, dt, cfg.EXP4_CIR_PATHS, rng)
    e_psi = default_layer_reduction(lam_a, lam_b, t)
    return {"T": T, "sigma_a": cfg.EXP4_CIR_SIGMA, "sigma_b": cfg.EXP4_CIR_SIGMA * 1.5,
            "n_paths": cfg.EXP4_CIR_PATHS, "E_Psi_T": e_psi}


def run() -> dict:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    return {
        "closed_form_and_coupling": run_closed_form_and_coupling(rng),
        "sharp_range": run_sharp_range(),
        "l1_stability": run_stability_sharpness(),
        "random_intensity": run_random_intensity(rng),
    }


def save_results(result: dict) -> None:
    with open(f"{RESULTS_DIR}/exp4_results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    stab = result["l1_stability"]
    eps = [r["epsilon"] for r in stab]
    lhs = [r["lhs"] for r in stab]
    rhs = [r["rhs"] for r in stab]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(eps, lhs, "o-", label=r"$|\Psi_T(a,b)-\Psi_T(\tilde a,b)|$")
    ax.loglog(eps, rhs, "s--", label=r"$\int|a-\tilde a|$ (bound, constant 1)")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("value")
    ax.set_title("Theorem 4.9: sharp L$^1$ stability constant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_l1_stability.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("exp_4: Cox bond prices with deterministic hazards")
    result = run()
    c = result["closed_form_and_coupling"]
    print(f"  Psi_T closed form = {c['psi_closed_form']:.6f}, common-shock MC = {c['psi_common_shock_coupling_mc']:.6f}, "
          f"abs diff = {c['abs_diff']:.4e}")
    r = result["sharp_range"]
    print(f"  sharp range for (h_a={r['h_a']}, h_b={r['h_b']}): [{r['range_lower']:.6f}, {r['range_upper']:.6f}]")
    ri = result["random_intensity"]
    print(f"  E[Psi_T] under CIR factors (sigma={ri['sigma_a']},{ri['sigma_b']}) = {ri['E_Psi_T']:.6f}")
    save_results(result)
    print("  saved results/exp4_results.json, exp4_l1_stability.png")


if __name__ == "__main__":
    main()
