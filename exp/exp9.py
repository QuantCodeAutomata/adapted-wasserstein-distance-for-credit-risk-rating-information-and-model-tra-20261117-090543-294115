"""exp_9: Monte Carlo study of the rolling bootstrap hedge under the review,
intra-year migration and CIR intensity market models (Section 5.2, Figure 2).

For a seven-year zero-recovery bond with annual pins, starting from the BB
rating of the stylized S&P-style transition matrix, this experiment compares
the realised expected absolute cash residual of the rolling bootstrap hedge
with the Frechet bound E|exp(-A)-exp(-B)| and the adapted bound E[Psi] (Eq.
4.45-4.48), and decomposes the slack between realised and adapted into a
within-bucket (conditional Jensen) component and a hazard-crossing component
(Eq. 4.47), under three market models:
  1. review: the rating for the coming year is revealed at each pin;
  2. intra-year: the rating migrates once at a uniform time inside the year;
  3. CIR: the hazard follows a CIR process, Euler-discretized at dt=1/250.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP9_CIR_DT, EXP9_CIR_KAPPA, EXP9_CIR_LAMBDA0, EXP9_CIR_PATHS_RUN, EXP9_CIR_SIGMAS,
    EXP9_CIR_THETA, EXP9_K_YEARS, EXP9_PATHS_RUN, RANDOM_SEED,
)
from src.rating_data import sp_style_systems
from src.rolling_hedge import (
    bootstrap_cumulative_hazard, bootstrap_rates, cir_zero_coupon_price,
    hazard_from_default_prob, intra_year_model_survival_curve,
    renormalized_non_default_transitions, review_model_survival_curve,
    rolling_hedge_cir, rolling_hedge_intra_year, rolling_hedge_review, rolling_positions,
)

START_RATING_NAME = "BB"


def decompose(res: dict) -> dict:
    """Split the slack E[Psi]-E|epsilon| into a within-bucket (Jensen) part
    and a hazard-crossing part, by bucket then summed (Eq. 4.47, step 9 of
    the methodology). realised <= frechet (=E|phi|) <= adapted (=E[Psi]):
    the first inequality is Jensen's (|E[X|F]| <= E[|X| | F]) applied to the
    bucket residual, so frechet - realised is the within-bucket conditional
    Jensen slack; the second is Eq. 4.33 (Psi = phi + a nonnegative
    adaptation-slack term), so adapted - frechet is the hazard-crossing
    slack coming from unordered post-review hazards."""
    adapted = res["sum_adapted"]
    frechet = res["sum_frechet"]
    realized = res["sum_bucket_realized_abs"]
    within_bucket = max(frechet - realized, 0.0)
    hazard_crossing = max(adapted - frechet, 0.0)
    return {
        "adapted": adapted, "frechet": frechet, "realized": realized,
        "within_bucket_slack": within_bucket, "hazard_crossing_slack": hazard_crossing,
        "ratio_realized_to_adapted": realized / adapted if adapted > 0 else float("nan"),
        "within_bucket_share": within_bucket / (adapted - realized) if adapted > realized else float("nan"),
        "hazard_crossing_share": hazard_crossing / (adapted - realized) if adapted > realized else float("nan"),
    }


def run() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    systems = sp_style_systems()
    P = systems["base"]
    Q = renormalized_non_default_transitions(P)
    hazards = hazard_from_default_prob(P[1:, 0])
    rating_names = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]
    start = rating_names.index(START_RATING_NAME)
    K = EXP9_K_YEARS
    pins = np.arange(K + 1, dtype=float)

    results: dict = {"start_rating": START_RATING_NAME, "K_years": K, "n_paths": EXP9_PATHS_RUN}

    # 1. Review model
    surv_rev = review_model_survival_curve(Q, hazards, start, K)
    betas_r = bootstrap_rates(surv_rev, pins)
    N_r = rolling_positions(bootstrap_cumulative_hazard(betas_r, pins))
    res_r = rolling_hedge_review(hazards, Q, start, pins, betas_r, N_r, EXP9_PATHS_RUN, rng)
    results["review"] = decompose(res_r)
    results["review"]["mean_signed_residual"] = res_r["mean_signed_residual"]
    results["review"]["total_cash_sample"] = res_r["total_cash"][:5000].tolist()

    # 2. Intra-year migration model
    surv_intra = intra_year_model_survival_curve(Q, hazards, start, K)
    betas_i = bootstrap_rates(surv_intra, pins)
    N_i = rolling_positions(bootstrap_cumulative_hazard(betas_i, pins))
    res_i = rolling_hedge_intra_year(hazards, Q, start, pins, betas_i, N_i, EXP9_PATHS_RUN, rng)
    results["intra_year"] = decompose(res_i)
    results["intra_year"]["mean_signed_residual"] = res_i["mean_signed_residual"]
    results["intra_year"]["total_cash_sample"] = res_i["total_cash"][:5000].tolist()

    # 3. CIR intensity model, for three volatility levels
    results["cir"] = {}
    for sigma in EXP9_CIR_SIGMAS:
        lam0 = EXP9_CIR_LAMBDA0
        surv_cir = cir_zero_coupon_price(pins[1:], lam0, EXP9_CIR_KAPPA, EXP9_CIR_THETA, sigma)
        surv_cir = np.concatenate([[1.0], surv_cir])
        betas_c = bootstrap_rates(surv_cir, pins)
        N_c = rolling_positions(bootstrap_cumulative_hazard(betas_c, pins))
        res_c = rolling_hedge_cir(EXP9_CIR_KAPPA, EXP9_CIR_THETA, lam0, sigma, pins, betas_c, N_c,
                                   EXP9_CIR_PATHS_RUN, EXP9_CIR_DT, rng)
        dec = decompose(res_c)
        dec["mean_signed_residual"] = res_c["mean_signed_residual"]
        results["cir"][f"sigma_{sigma}"] = dec

    with open("results/exp9_results.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2, default=float)

    # Figure 2 analogue: stacked-bar slack decomposition + residual density
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["review", "intra_year"] + [f"CIR sigma={s}" for s in EXP9_CIR_SIGMAS]
    decs = [results["review"], results["intra_year"]] + [results["cir"][f"sigma_{s}"] for s in EXP9_CIR_SIGMAS]
    realized = [d["realized"] for d in decs]
    within = [d["within_bucket_slack"] for d in decs]
    crossing = [d["hazard_crossing_slack"] for d in decs]
    x = np.arange(len(labels))
    axes[0].bar(x, realized, label="Realised E|epsilon|")
    axes[0].bar(x, within, bottom=realized, label="Within-bucket slack")
    axes[0].bar(x, crossing, bottom=np.array(realized) + np.array(within), label="Hazard-crossing slack")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Expected absolute residual")
    axes[0].set_title("Rolling-hedge residual vs adapted bound decomposition")
    axes[0].legend()

    axes[1].hist(res_r["total_cash"], bins=80, alpha=0.5, label="review", density=True)
    axes[1].hist(res_i["total_cash"], bins=80, alpha=0.5, label="intra-year", density=True)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Total cash residual")
    axes[1].set_title("Distribution of the total cash residual")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("results/exp9_decomposition.png", dpi=120)
    plt.close(fig)
    return results


if __name__ == "__main__":
    out = run()
    print(f"exp_9: start={out['start_rating']}, K={out['K_years']}, paths={out['n_paths']}")
    print(f"  review:     ratio={out['review']['ratio_realized_to_adapted']:.4f} "
          f"mean_signed={out['review']['mean_signed_residual']:.2e}")
    print(f"  intra-year: ratio={out['intra_year']['ratio_realized_to_adapted']:.4f} "
          f"within_share={out['intra_year']['within_bucket_share']:.3f} "
          f"crossing_share={out['intra_year']['hazard_crossing_share']:.3f} "
          f"mean_signed={out['intra_year']['mean_signed_residual']:.2e}")
    for sigma in EXP9_CIR_SIGMAS:
        d = out["cir"][f"sigma_{sigma}"]
        print(f"  CIR sigma={sigma}: ratio={d['ratio_realized_to_adapted']:.4f} "
              f"mean_signed={d['mean_signed_residual']:.2e}")
