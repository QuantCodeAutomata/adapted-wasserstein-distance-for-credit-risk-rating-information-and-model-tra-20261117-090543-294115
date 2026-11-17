"""exp_10: Hedging with multiple instruments -- bootstrap desk vs market desk
with a liquidity window of m bonds (Section 4.8, Theorem 4.27, Proposition 4.28).

Both desks trade only bonds maturing within the next m years. The bootstrap
desk rolls a constant window-m ladder calibrated to the quoted survival
curve (generalizing the single-bond roll of Section 4.7, Eq. 4.49). The
market desk instead solves, at every pin, a least-squares match of the
liability's conditional price on the span of the first m Krylov vectors
of the survival-weighted kernel K_hat (Eq. 4.51). The Krylov singular values
reveal the effective dimension d of the rating-driven term structure
(Proposition 4.28): the market desk should hedge (numerically) exactly once
m >= d, while the bootstrap desk only improves gradually with m.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EXP10_K_YEARS, EXP10_PATHS_RUN, EXP10_SVD_THRESHOLD, EXP10_WINDOW_RANGE, RANDOM_SEED
from src.rating_data import sp_style_systems
from src.rolling_hedge import (
    bootstrap_cumulative_hazard,
    bootstrap_rates,
    hazard_from_default_prob,
    krylov_matrix,
    market_desk_weights,
    renormalized_non_default_transitions,
    rolling_positions,
    simulate_default_times_given_ratings,
    simulate_review_ratings,
    survival_weighted_kernel,
)

START_RATINGS = ["BB", "B"]


def _conditional_price_powers(K_hat: np.ndarray, K: int) -> list[np.ndarray]:
    """powers[h] = K_hat^h @ 1, the h-year-ahead conditional survival price
    from every starting rating under the current-rating-hazard review model
    (Eq. 4.51: K_hat(i,i') = exp(-h_i) Q(i,i'), so (K_hat^h 1)(i) is the
    h-step survival price starting from rating i)."""
    n = K_hat.shape[0]
    vec = np.ones(n)
    powers = [vec.copy()]
    for _ in range(K):
        vec = K_hat @ vec
        powers.append(vec.copy())
    return powers


def run() -> dict:
    """Run the window-hedge comparison for every start rating and window m,
    and the Krylov spanning diagnostic, saving results/exp10_results.json
    and results/exp10_window_hedge.png."""
    rng = np.random.default_rng(RANDOM_SEED)
    systems = sp_style_systems()
    P = systems["base"]
    Q = renormalized_non_default_transitions(P)
    hazards = hazard_from_default_prob(P[1:, 0])
    n_nd = Q.shape[0]
    K = EXP10_K_YEARS
    pins = np.arange(K + 1, dtype=float)
    rating_names = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"][:n_nd]

    # Survival-weighted kernel (Sec. 4.8) and its Krylov space [1, K_hat.1, ...].
    K_hat = survival_weighted_kernel(Q, hazards)
    powers = _conditional_price_powers(K_hat, K)
    kry = krylov_matrix(K_hat, n_nd)
    svals = np.linalg.svd(kry, compute_uv=False)
    rel = svals / svals[0]
    eff_dim = int(np.sum(rel > EXP10_SVD_THRESHOLD))

    out = {
        "K_years": K, "n_paths": EXP10_PATHS_RUN,
        "krylov_singular_values": svals.tolist(),
        "krylov_relative_singular_values": rel.tolist(),
        "krylov_effective_dimension": eff_dim,
        "by_rating": {},
    }

    for name in START_RATINGS:
        start = rating_names.index(name)
        curve = np.array([powers[k][start] for k in range(K + 1)])
        betas = bootstrap_rates(curve, pins)
        B = bootstrap_cumulative_hazard(betas, pins)
        N = rolling_positions(B)

        # Review model under the current-rating-hazard convention: the hazard
        # applying over bucket (t_b, t_{b+1}] is that of the rating already
        # known at the left pin t_b, consistent with K_hat's convention and
        # with the equality case "hazard revealed at its left pin" (Eq. 4.47).
        ratings = simulate_review_ratings(Q, start, K, EXP10_PATHS_RUN, rng)
        ratings_shift = np.concatenate([ratings[:, :1], ratings[:, :-1]], axis=1)
        default_time = simulate_default_times_given_ratings(ratings_shift, hazards, pins, rng, intra_year=False)

        rows = []
        for m in EXP10_WINDOW_RANGE:
            m = int(min(m, K))
            # ---- Bootstrap desk: constant window-m ladder (generalizes Eq. 4.49) ----
            boot_total = np.zeros(EXP10_PATHS_RUN)
            for b in range(K):
                alive = default_time > pins[b]
                idx_prev = min(b - 1 + m, K) if b >= 1 else 0
                idx_new = min(b + m, K)
                h_prev, h_new = idx_prev - b, idx_new - b
                price_prev = powers[h_prev][ratings[:, b]]
                price_new = powers[h_new][ratings[:, b]]
                eps_b = N[idx_prev] * price_prev - N[idx_new] * price_new
                boot_total += np.where(alive, eps_b, 0.0)
            boot_abs = float(np.mean(np.abs(boot_total)))

            # ---- Market desk: least-squares match on the m window bonds (Eq. 4.51) ----
            basis_dim = min(m, n_nd)
            mkt_total = np.zeros(EXP10_PATHS_RUN)
            for b in range(K):
                alive = default_time > pins[b]
                theta = market_desk_weights(K_hat, b, K, basis_dim)
                basis_prices = np.array([powers[h][ratings[:, b]] for h in range(1, basis_dim + 1)])
                port_value = theta @ basis_prices
                target_price = powers[max(K - b, 0)][ratings[:, b]]
                mkt_total += np.where(alive, port_value - target_price, 0.0)
            mkt_abs = float(np.mean(np.abs(mkt_total)))

            rows.append({"m": m, "bootstrap_abs_residual": boot_abs, "market_abs_residual": mkt_abs})
        out["by_rating"][name] = rows

    with open("results/exp10_results.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in out["by_rating"].items():
        ms = [r["m"] for r in rows]
        ax.plot(ms, [r["bootstrap_abs_residual"] for r in rows], "o--", label=f"bootstrap {name}")
        ax.plot(ms, [r["market_abs_residual"] for r in rows], "s-", label=f"market {name}")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("liquidity window m (years)")
    ax.set_ylabel("E|total cash residual|")
    ax.set_title("Window hedge: bootstrap desk vs market desk (Sec. 4.8)")
    ax.legend(fontsize=8)
    ax.axvline(eff_dim, color="gray", linestyle=":", label=f"Krylov dim d={eff_dim}")
    fig.tight_layout()
    fig.savefig("results/exp10_window_hedge.png", dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = run()
    print(f"exp_10: K={result['K_years']} years, {result['n_paths']} MC paths, "
          f"Krylov effective dimension d={result['krylov_effective_dimension']}")
    print(f"  Krylov singular values: {np.array(result['krylov_singular_values'])}")
    for name, rows in result["by_rating"].items():
        print(f"  {name}:")
        for r in rows:
            print(f"    m={r['m']}: bootstrap={r['bootstrap_abs_residual']:.6f} "
                  f"market={r['market_abs_residual']:.6f}")
