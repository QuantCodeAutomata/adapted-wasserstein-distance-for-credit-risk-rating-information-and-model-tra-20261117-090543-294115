"""exp_2: multi-period rating migration -- backward dynamic programming and
synchronous (Knothe-Rosenblatt) coupling for the adapted Wasserstein distance
between two Markov rating-chain bond-price processes (Sec. 3.3, Theorems
3.16 and 3.18).

Uses the stylized rating-transition systems of ``src.rating_data`` (see that
module's docstring and results/RESULTS.md for why they replace the S&P
Global Ratings 2024 study tables). For a chosen pair of systems (U.S. vs.
Europe), the second system is first calibrated, rating by rating, so that
its initial bond price matches the first system's (page 45), exactly as
``src.transition_utils.calibrate_scale_to_price`` implements it via
``regional_variant``.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.rating_data import RATINGS, regional_variant, sp_style_systems
from src.transition_utils import (
    backward_dp_adapted_distance,
    calibrate_scale_to_price,
    jafry_schuermann_mobility,
    price_kernels,
    stochastic_monotonicity_violation,
    survival_probabilities,
    synchronous_coupling_default_mismatch,
)

RESULTS_DIR = "results"
RATING_INDEX = {name: i + 1 for i, name in enumerate(RATINGS)}  # +1: index 0 is Default


def run() -> dict:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    systems = sp_style_systems()
    P_us = systems["us"]
    P_eu_raw = systems["europe"]

    mono_us = stochastic_monotonicity_violation(P_us)
    mono_eu_raw = stochastic_monotonicity_violation(P_eu_raw)
    mobility_us = jafry_schuermann_mobility(P_us)
    mobility_eu = jafry_schuermann_mobility(P_eu_raw)

    rows = []
    for rating_name in cfg.EXP2_INITIAL_RATINGS:
        idx = RATING_INDEX[rating_name]
        for N in cfg.EXP2_MATURITIES:
            N = int(N)
            q_us = survival_probabilities(P_us, N)
            target_price = q_us[0, idx]
            scale, P_eu = calibrate_scale_to_price(
                lambda s: regional_variant(P_eu_raw, s), target_price, N, idx
            )
            q_eu = survival_probabilities(P_eu, N)
            assert abs(q_eu[0, idx] - target_price) < 1e-8, "calibration must match the initial price exactly"

            sp_us, kp_us = price_kernels(P_us, q_us, N, idx)
            sp_eu, kp_eu = price_kernels(P_eu, q_eu, N, idx)
            aw2_sq = backward_dp_adapted_distance(sp_us, kp_us, sp_eu, kp_eu, N)
            aw2 = np.sqrt(max(aw2_sq, 0.0))

            sync_mismatch = synchronous_coupling_default_mismatch(
                P_us, P_eu, N, cfg.EXP2_SYNC_PATHS, rng, initial_rating=idx
            )
            mono_eu_cal = stochastic_monotonicity_violation(P_eu)

            rows.append({
                "initial_rating": rating_name, "N": N, "calibration_scale": scale,
                "target_price": target_price, "AW2_sq_exact_LP": aw2_sq, "AW2_exact_LP": aw2,
                "synchronous_coupling_mismatch": sync_mismatch,
                "sync_minus_exact": sync_mismatch - aw2_sq,
                "mono_violation_us": mono_us, "mono_violation_eu_calibrated": mono_eu_cal,
            })
            print(f"  rating={rating_name:6s} N={N:3d} AW2={aw2:.6f} AW2_sq={aw2_sq:.6f} "
                  f"sync_mismatch={sync_mismatch:.6f} (diff={sync_mismatch - aw2_sq:+.6f})")

    table = pd.DataFrame(rows)
    max_sync_error = float((table["synchronous_coupling_mismatch"] - table["AW2_sq_exact_LP"]).abs().max())
    # Theorem 3.18: equality for stochastically monotone kernels; here both
    # matrices have a small monotonicity violation (page 46 reports the
    # S&P data has one too), so the synchronous coupling only approximates
    # the exact LP value, with an error of the same small order as the
    # violation itself.
    summary = {
        "mono_violation_us": mono_us,
        "mono_violation_europe_raw": mono_eu_raw,
        "mobility_index_us": mobility_us,
        "mobility_index_europe_raw": mobility_eu,
        "mobility_index_relative_gap_pct": abs(mobility_us - mobility_eu) / mobility_us * 100,
        "max_abs_sync_minus_exact": max_sync_error,
    }
    return {"table": table, "summary": summary}


def save_results(result: dict) -> None:
    table: pd.DataFrame = result["table"]
    table.to_csv(f"{RESULTS_DIR}/exp2_adapted_distances.csv", index=False)
    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as f:
        json.dump(result["summary"], f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rating_name in cfg.EXP2_INITIAL_RATINGS:
        sub = table[table["initial_rating"] == rating_name].sort_values("N")
        ax.plot(sub["N"], sub["AW2_exact_LP"], "o-", label=rating_name)
    ax.set_xlabel("maturity N (years)")
    ax.set_ylabel(r"$AW_2(P^{US}, P^{Europe})$")
    ax.set_title("Theorem 3.16: exact adapted distance between calibrated rating systems")
    ax.legend(title="initial rating")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_adapted_distance_by_maturity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["AW2_sq_exact_LP"], table["synchronous_coupling_mismatch"], "o")
    lims = [0, max(table["AW2_sq_exact_LP"].max(), table["synchronous_coupling_mismatch"].max()) * 1.05]
    ax.plot(lims, lims, "k--", label="y = x (exact equality)")
    ax.set_xlabel(r"exact LP value $AW_2^2$")
    ax.set_ylabel("synchronous-coupling Monte Carlo mismatch")
    ax.set_title("Theorem 3.18: synchronous coupling vs. exact LP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_synchronous_vs_exact.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("exp_2: dynamic programming and synchronous coupling for adapted distance")
    result = run()
    print("  monotonicity violation (US):", result["summary"]["mono_violation_us"])
    print("  max |synchronous - exact LP|:", result["summary"]["max_abs_sync_minus_exact"])
    save_results(result)
    print("  saved results/exp2_adapted_distances.csv, exp2_summary.json, "
          "exp2_adapted_distance_by_maturity.png, exp2_synchronous_vs_exact.png")


if __name__ == "__main__":
    main()
