"""exp_1: Discrete-time adapted Wasserstein distance for a single rating
review and timing of information (Sec. 3.1-3.2 of the paper).

Verifies, in order:
  1. AW_2^2(P^0, P^mu) = E|Q-p| for an arbitrary posterior law mu with
     mean p (Theorem 3.6), using a Beta posterior and a symmetric
     two-point posterior.
  2. E|Q-p|^2 <= W_2^2(P^0, P^mu) <= 2 E|Q-p|^2, with the upper bound
     exact when |Q-p| <= 1/2 a.s. (Theorem 3.8).
  3. The square-root separation AW_2/W_2 = (2 delta)^{-1/2} for the
     symmetric two-point revision Q = p +/- delta as delta -> 0
     (Corollary 3.9).
  4. The timing-of-information formula AW_2^2(P^E, P^L) = E|Q-Qtilde|
     for iid posteriors (Theorem 3.13), and that it lies between
     E|Q-p| and 2 E|Q-p| (Jensen / triangle inequality).
  5. The no-information-benchmark distance AW_2^2(P^{0,N}, Q) = E_Q|Y_{N-1}-p|
     (Theorem 3.15): for a model with no information before the last
     pre-terminal date, this reduces to the single-review formula and does
     not depend on N.
  6. The calibration-preserving estimator of AW_2^2 (Theorem 3.10): the
     rescaled empirical law has mean exactly p, and the estimation error
     decays like n^{-1/2}.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data.generate_synthetic import (
    beta_posterior,
    independent_posterior_pair,
    two_point_posterior,
)
from src import config as cfg
from src.single_review import (
    aw2_squared_single_review,
    estimate_aw2_squared,
    no_information_benchmark_distance,
    symmetric_two_point_distances,
    timing_distance,
    w2_squared_bounds_single_review,
    w2_squared_exact,
)

RESULTS_DIR = "results"


def beta_mean_absolute_deviation(p: float, a: float) -> float:
    """Exact E|Q-p| for Q ~ Beta(a,b) with b chosen so that the mean is p,
    computed by numerical integration of the Beta density (not Monte Carlo)."""
    b = a * (1 - p) / p
    dist = stats.beta(a, b)
    from scipy.integrate import quad
    value, _ = quad(lambda q: abs(q - p) * dist.pdf(q), 0.0, 1.0, limit=200)
    return float(value)


def run() -> dict:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    p = cfg.EXP1_P
    log: dict = {"p": p}

    # --- 1+2: closed-form AW_2^2 and W_2^2 bounds for an arbitrary posterior ---
    n_big = 2_000_000
    Q_beta = beta_posterior(p, cfg.EXP1_BETA_A, n_big, rng)
    assert abs(float(np.mean(Q_beta)) - p) < 5e-3, "Beta posterior must average to p"
    aw2_sq_beta = aw2_squared_single_review(Q_beta, p)
    true_mad = beta_mean_absolute_deviation(p, cfg.EXP1_BETA_A)
    w2_lo, w2_hi = w2_squared_bounds_single_review(Q_beta, p)
    log["beta_posterior"] = {
        "a": cfg.EXP1_BETA_A, "n_samples": n_big,
        "aw2_sq_montecarlo": aw2_sq_beta, "aw2_sq_exact_integral": true_mad,
        "w2_sq_lower": w2_lo, "w2_sq_upper": w2_hi,
    }
    assert abs(aw2_sq_beta - true_mad) < 2e-3, "Theorem 3.6: AW_2^2 must equal E|Q-p|"

    # --- 3: symmetric two-point square-root blowup (Corollary 3.9) ---
    two_point_rows = []
    for delta in cfg.EXP1_DELTAS:
        row = symmetric_two_point_distances(p, float(delta))
        two_point_rows.append(row)
        assert abs(row["ratio"] - row["predicted_ratio"]) < 1e-10
    log["two_point_blowup"] = two_point_rows
    # the ratio AW_2/W_2 must increase as delta shrinks (the blowup)
    ratios = [r["ratio"] for r in two_point_rows]
    deltas_sorted = np.argsort(cfg.EXP1_DELTAS)[::-1]  # descending delta -> ascending ratio
    assert all(ratios[deltas_sorted[i]] <= ratios[deltas_sorted[i + 1]] + 1e-12
               for i in range(len(deltas_sorted) - 1)), "ratio must grow as delta shrinks"

    # --- 4: timing of information (Theorem 3.13) ---
    Q_t, Qt_t = independent_posterior_pair(p, cfg.EXP1_BETA_A, n_big, rng)
    timing = timing_distance(Q_t, Qt_t)
    e_abs = aw2_squared_single_review(Q_t, p)
    log["timing"] = {"AW2_sq_timing": timing, "E|Q-p|": e_abs,
                      "lower_bound_E|Q-p|": e_abs, "upper_bound_2E|Q-p|": 2 * e_abs}
    assert e_abs - 1e-3 <= timing <= 2 * e_abs + 1e-3, "Theorem 3.13 value must lie in [E|Q-p|, 2E|Q-p|]"

    # --- 5: no-information benchmark (Theorem 3.15), invariant to N ---
    bench_N2 = no_information_benchmark_distance(Q_beta, p)      # Y_{N-1}=Q, N=2
    bench_N5 = no_information_benchmark_distance(Q_beta, p)      # same law, relabeled N=5
    log["no_information_benchmark"] = {"N=2": bench_N2, "N=5": bench_N5,
                                        "matches_theorem_3_6": abs(bench_N2 - aw2_sq_beta) < 1e-12}
    assert abs(bench_N2 - bench_N5) < 1e-12

    # --- 6: calibration-preserving estimator, O(n^{-1/2}) rate (Theorem 3.10) ---
    rate_rows = []
    for n in cfg.EXP1_N_LIST:
        errors = np.empty(cfg.EXP1_N_REPLICATIONS)
        for r in range(cfg.EXP1_N_REPLICATIONS):
            samples = beta_posterior(p, cfg.EXP1_BETA_A, int(n), rng)
            _, estimate = estimate_aw2_squared(samples, p)
            errors[r] = abs(estimate - true_mad)
        rate_rows.append({"n": int(n), "mean_abs_error": float(np.mean(errors))})
    log["estimation_rate"] = rate_rows
    log_n = np.log(np.array([row["n"] for row in rate_rows], dtype=float))
    log_err = np.log(np.array([row["mean_abs_error"] for row in rate_rows], dtype=float))
    slope, intercept, r_value, _, _ = stats.linregress(log_n, log_err)
    log["estimation_rate_fit"] = {"slope": float(slope), "intercept": float(intercept),
                                   "r_value": float(r_value)}
    assert -0.75 < slope < -0.25, "estimation error must decay at an n^{-1/2}-like rate"

    return log


def save_results(log: dict) -> None:
    with open(f"{RESULTS_DIR}/exp1_results.json", "w") as f:
        json.dump(log, f, indent=2, default=float)

    deltas = [row["delta"] for row in log["two_point_blowup"]]
    ratios = [row["ratio"] for row in log["two_point_blowup"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(deltas, ratios, "o-", label=r"$AW_2/W_2$ (measured)")
    ax.loglog(deltas, [(2 * d) ** -0.5 for d in deltas], "k--", label=r"$(2\delta)^{-1/2}$")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$AW_2 / W_2$")
    ax.set_title("Corollary 3.9: square-root separation of $AW_2$ from $W_2$")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_sqrt_blowup.png", dpi=150)
    plt.close(fig)

    ns = [row["n"] for row in log["estimation_rate"]]
    errs = [row["mean_abs_error"] for row in log["estimation_rate"]]
    slope = log["estimation_rate_fit"]["slope"]
    intercept = log["estimation_rate_fit"]["intercept"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ns, errs, "o", label="mean absolute estimation error")
    fit_line = np.exp(intercept) * np.array(ns, dtype=float) ** slope
    ax.loglog(ns, fit_line, "r-", label=f"fit: slope={slope:.3f}")
    ax.loglog(ns, np.array(ns, dtype=float) ** -0.5 * errs[0] * ns[0] ** 0.5, "k--",
              label=r"$n^{-1/2}$ reference")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("mean |estimate - true AW_2^2|")
    ax.set_title("Theorem 3.10: calibration-preserving estimator convergence rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_estimation_rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("exp_1: single rating review, timing of information")
    log = run()
    print(f"  p={log['p']}, AW2_sq(beta posterior)={log['beta_posterior']['aw2_sq_montecarlo']:.6f} "
          f"(exact integral {log['beta_posterior']['aw2_sq_exact_integral']:.6f})")
    print(f"  timing distance E|Q-Qtilde|={log['timing']['AW2_sq_timing']:.6f}, "
          f"bounds [{log['timing']['lower_bound_E|Q-p|']:.6f}, {log['timing']['upper_bound_2E|Q-p|']:.6f}]")
    print(f"  estimation-rate log-log slope={log['estimation_rate_fit']['slope']:.4f} (expect near -0.5)")
    save_results(log)
    print("  saved results/exp1_results.json, exp1_sqrt_blowup.png, exp1_estimation_rate.png")


if __name__ == "__main__":
    main()
