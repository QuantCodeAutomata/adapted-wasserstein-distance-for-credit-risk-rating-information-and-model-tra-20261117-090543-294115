"""exp_6: Rolling bootstrap along a quoted term structure and hedging with a
window of liquid bonds (Section 4.7-4.8, Theorem 4.26, Proposition 4.28).

Builds the review-model market model for a seven-year bond with annual pins
on the stylized S&P-style transition matrix, calibrates the rolling bootstrap
hedge to its survival curve, and verifies the bucket-by-bucket identity of
Theorem 4.26: the realised residual, the Frechet term and the adapted bound
coincide exactly in the review model, where the post-review hazard on each
bucket is revealed at its left pin. It also reports the Krylov-space
dimension of the survival-weighted kernel that underlies the liquidity-window
hedge of Section 4.8 (used quantitatively in exp_10).
"""
from __future__ import annotations

import json

import numpy as np

from src.config import EXP6_K_YEARS, EXP6_PATHS, EXP6_SVD_THRESHOLD, RANDOM_SEED
from src.rating_data import sp_style_systems
from src.rolling_hedge import (
    bootstrap_cumulative_hazard,
    bootstrap_rates,
    hazard_from_default_prob,
    krylov_matrix,
    renormalized_non_default_transitions,
    review_model_survival_curve,
    rolling_hedge_review,
    rolling_positions,
    survival_weighted_kernel,
)


def run() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    systems = sp_style_systems()
    P = systems["base"]
    Q = renormalized_non_default_transitions(P)
    hazards = hazard_from_default_prob(P[1:, 0])
    n_nd = Q.shape[0]
    K = EXP6_K_YEARS
    pins = np.arange(K + 1, dtype=float)

    rows = []
    rating_names = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"][:n_nd]
    for start in range(n_nd):
        surv = review_model_survival_curve(Q, hazards, start, K)
        betas = bootstrap_rates(surv, pins)
        B = bootstrap_cumulative_hazard(betas, pins)
        N = rolling_positions(B)
        res = rolling_hedge_review(hazards, Q, start, pins, betas, N, EXP6_PATHS, rng)
        rows.append({
            "start_rating": rating_names[start],
            "survival_curve": surv.tolist(),
            "sum_realized_abs": res["sum_bucket_realized_abs"],
            "sum_frechet": res["sum_frechet"],
            "sum_adapted": res["sum_adapted"],
            "mean_signed_residual": res["mean_signed_residual"],
            # Theorem 4.26: under the review model the three quantities coincide exactly.
            "max_pairwise_gap": float(max(
                abs(res["sum_bucket_realized_abs"] - res["sum_frechet"]),
                abs(res["sum_frechet"] - res["sum_adapted"]),
                abs(res["sum_bucket_realized_abs"] - res["sum_adapted"]),
            )),
        })

    # Krylov space of the survival-weighted kernel K_hat (Section 4.8), used
    # by exp_10 to determine how many liquid bonds span the hedge exactly.
    K_hat = survival_weighted_kernel(Q, hazards)
    kry = krylov_matrix(K_hat, n_nd)
    svals = np.linalg.svd(kry, compute_uv=False)
    rel = svals / svals[0]
    eff_dim = int(np.sum(rel > EXP6_SVD_THRESHOLD))

    out = {
        "K_years": K,
        "n_paths": EXP6_PATHS,
        "buckets": rows,
        "krylov_singular_values": svals.tolist(),
        "krylov_relative_singular_values": rel.tolist(),
        "krylov_effective_dimension": eff_dim,
    }
    with open("results/exp6_results.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    result = run()
    print(f"exp_6: K={result['K_years']} years, {result['n_paths']} MC paths")
    for row in result["buckets"]:
        print(f"  {row['start_rating']:6s} realized={row['sum_realized_abs']:.6f} "
              f"frechet={row['sum_frechet']:.6f} adapted={row['sum_adapted']:.6f} "
              f"max_gap={row['max_pairwise_gap']:.2e} mean_signed={row['mean_signed_residual']:.2e}")
    print(f"  Krylov effective dimension d = {result['krylov_effective_dimension']}")
    print(f"  Krylov singular values: {np.array(result['krylov_singular_values'])}")
