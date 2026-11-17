"""exp_8: Empirical comparison of calibrated rating systems (Section 5.1,
Table 1, page 45-47): adapted Wasserstein distances between regional,
vintage and cycle variants of the same base rating system, the
stochastic-monotonicity test and the synchronous-coupling gap, the distance
to the no-information benchmark (Theorem 3.15), the Jafry-Schuermann
mobility index, straddle prices, and the hedging-error stability bound of
Proposition 3.22.

The S&P Global Ratings 2024 annual transition study [69] used in the paper
is a proprietary PDF report; it is not served by the QuantX Dataloader or by
the Massive REST API (see src/rating_data.py). The three calibrated-system
comparisons are built instead from a single stylized base transition matrix
(data/generate_synthetic.py) using the paper's own calibration transforms
(src/rating_data.py), as recorded in results/RESULTS.md.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EXP8_INITIAL_RATINGS, EXP8_MATURITIES, EXP8_SYNC_PATHS, RANDOM_SEED
from src.rating_data import RATINGS, sp_style_systems
from src.transition_utils import (
    backward_dp_adapted_distance, calibrate_scale_to_price, jafry_schuermann_mobility,
    price_kernels, stochastic_monotonicity_violation, survival_probabilities,
    synchronous_coupling_default_mismatch,
)
from src.rating_data import regional_variant
from src.momentum_hedge import stability_bound


def _rating_index(name: str) -> int:
    return RATINGS.index(name)


def no_info_distance(P: np.ndarray, N: int, initial_rating: int) -> float:
    """Theorem 3.15: AW_2^2(P^{0,N}, Q) = E_Q|Y_{N-1} - p|, the distance from
    the completely coarse benchmark that reveals nothing before maturity."""
    q = survival_probabilities(P, N)
    p = q[0, initial_rating]
    # law of Y_{N-1} starting from initial_rating, propagated N-1 steps
    dist = np.zeros(P.shape[0])
    dist[initial_rating] = 1.0
    Pk = np.linalg.matrix_power(P, N - 1)
    law = Pk[initial_rating]
    y_values = q[N - 1]
    return float(np.sum(law * np.abs(y_values - p)))


def run() -> dict:
    systems = sp_style_systems()
    rng = np.random.default_rng(RANDOM_SEED)
    pairs = {
        "regional_us_vs_europe": (systems["us"], systems["europe"]),
        "vintage_long_run_vs_2024": (systems["long_run"], systems["vintage_2024"]),
        "cycle_recession_vs_ttc": (systems["recession"], systems["long_run"]),
    }

    out: dict = {"pairs": {}}
    mono = {name: stochastic_monotonicity_violation(P) for name, P in systems.items() if name != "base"}
    out["monotonicity_violation"] = mono
    out["mobility_index"] = {name: jafry_schuermann_mobility(P) for name, P in systems.items() if name != "base"}

    for pair_name, (base_p, base_q) in pairs.items():
        rows = []
        for rating_name in EXP8_INITIAL_RATINGS:
            r = _rating_index(rating_name)
            for N in EXP8_MATURITIES:
                N = int(N)
                q_p = survival_probabilities(base_p, N)
                p0 = q_p[0, r]
                # calibrate system 2's default column scale so its initial price matches system 1
                # (page 45: "scaling its default column by a constant ... so the initial bond
                # price matches, then renormalizing the remaining transitions")
                scale, calibrated_q = calibrate_scale_to_price(
                    lambda s: regional_variant(base_q, s), p0, N, r, bracket=(0.05, 20.0)
                )
                sp_p, kp_p = price_kernels(base_p, q_p, N, r)
                q_q = survival_probabilities(calibrated_q, N)
                sp_q, kp_q = price_kernels(calibrated_q, q_q, N, r)
                aw2_sq = backward_dp_adapted_distance(sp_p, kp_p, sp_q, kp_q, N)
                aw2 = float(np.sqrt(max(aw2_sq, 0.0)))
                sync_mismatch = synchronous_coupling_default_mismatch(
                    base_p, calibrated_q, N, EXP8_SYNC_PATHS, rng, initial_rating=r
                )
                no_info = no_info_distance(base_p, N, r)

                # rating-date straddle G_t = |X_t - X_0| at t=1: |E_1 G - E_2 G| vs sqrt(t)*AW_2
                straddle_p = float(np.sum(base_p[r] * np.abs(q_p[1] - p0)))
                straddle_q = float(np.sum(calibrated_q[r] * np.abs(q_q[1] - p0)))
                straddle_gap = abs(straddle_p - straddle_q)
                straddle_bound = aw2  # sqrt(t)*AW_2 with t=1

                rows.append({
                    "rating": rating_name, "N": N, "p0": float(p0), "scale": float(scale),
                    "aw2": aw2, "aw2_sq": float(aw2_sq),
                    "synchronous_default_mismatch": sync_mismatch,
                    "sync_vs_lp_rel_diff": float(abs(sync_mismatch - aw2_sq) / aw2_sq) if aw2_sq > 1e-12 else 0.0,
                    "no_info_distance": no_info,
                    "straddle_price_p": straddle_p, "straddle_price_q": straddle_q,
                    "straddle_gap": straddle_gap, "straddle_bound_sqrt_t_aw2": straddle_bound,
                    "straddle_gap_over_bound": float(straddle_gap / straddle_bound) if straddle_bound > 1e-12 else 0.0,
                })
        out["pairs"][pair_name] = rows

    # hedging stability bound (Proposition 3.22) for the one-period straddle, cycle pair
    cyc_rows = out["pairs"]["cycle_recession_vs_ttc"]
    bb_n2 = next(r for r in cyc_rows if r["rating"] == "BB" and r["N"] == int(EXP8_MATURITIES[0]))
    L_G, K, c_N, L_H = 1.0, 1.0, 1.0, 1.0  # one-period straddle: see src/momentum_hedge.stability_bound docstring
    bound = stability_bound(L_G, K, c_N, L_H, bb_n2["aw2"])
    out["hedging_stability_example"] = {
        "rating": "BB", "N": int(EXP8_MATURITIES[0]), "aw2": bb_n2["aw2"],
        "straddle_gap": bb_n2["straddle_gap"], "bound": bound,
        "bound_holds": bool(bb_n2["straddle_gap"] <= bound + 1e-9),
    }

    with open("results/exp8_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pair_name, rows in out["pairs"].items():
        for rating_name in EXP8_INITIAL_RATINGS:
            sub = [r for r in rows if r["rating"] == rating_name]
            ax.plot([r["N"] for r in sub], [r["aw2"] for r in sub], "o-",
                    label=f"{pair_name} / {rating_name}")
    ax.set_xlabel("maturity N (years)")
    ax.set_ylabel("AW_2")
    ax.set_title("Adapted Wasserstein distance between calibrated rating systems")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/exp8_distances.png", dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = run()
    print("exp_8: monotonicity violation:", result["monotonicity_violation"])
    for pair_name, rows in result["pairs"].items():
        max_sync_diff = max(r["sync_vs_lp_rel_diff"] for r in rows)
        print(f"  {pair_name}: max sync-vs-LP relative diff = {max_sync_diff:.2e}, "
              f"AW_2 range = [{min(r['aw2'] for r in rows):.4f}, {max(r['aw2'] for r in rows):.4f}]")
    print("  hedging stability bound holds:", result["hedging_stability_example"]["bound_holds"])
