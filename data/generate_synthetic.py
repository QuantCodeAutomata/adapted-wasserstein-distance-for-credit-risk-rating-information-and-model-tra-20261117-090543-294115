"""Synthetic sample generation for the experiments' Monte Carlo components.

No market data is used anywhere in this replication: every experiment
(exp_1 through exp_10) is a numerical verification of closed-form theorems
or a Monte Carlo simulation of models fully specified by the paper (rating
chains, Cox hazards, CIR intensities). This module generates the random
draws those simulations need: posterior-law samples for the calibration
estimator of Theorem 3.10, symmetric two-point and Beta posterior laws for
Sections 3-4, and bounded centered shocks for the small-dispersion
asymptotics of Theorem 4.35.
"""
from __future__ import annotations

import numpy as np


def two_point_posterior(p: float, delta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from mu_delta = 0.5*delta_{p-delta} + 0.5*delta_{p+delta}."""
    assert 0 < delta < min(p, 1 - p)
    signs = rng.choice([-1.0, 1.0], size=n)
    return p + signs * delta


def beta_posterior(p: float, a: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from a Beta(a,b) rescaled to [0,1] with mean exactly p,
    where b = a*(1-p)/p so that a/(a+b) = p."""
    b = a * (1 - p) / p
    return rng.beta(a, b, size=n)


def independent_posterior_pair(p: float, a: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two independent samples Q, Q_tilde ~ mu (Beta posterior) for the
    timing-of-information comparison (Theorem 3.13)."""
    b = a * (1 - p) / p
    Q = rng.beta(a, b, size=n)
    Q_tilde = rng.beta(a, b, size=n)
    return Q, Q_tilde


def bernoulli_given_posterior(Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """D ~ Bernoulli(Q) elementwise, the terminal default indicator given
    the revealed posterior survival probability."""
    u = rng.uniform(size=Q.shape)
    return (u < Q).astype(float)


# ---------------------------------------------------------------------------
# Stylized rating-transition data for exp_2, exp_6, exp_8, exp_9 and exp_10.
#
# Section 5 of the paper computes its empirical illustrations from the S&P
# Global Ratings 2024 annual corporate default and transition study: global
# and regional one-year average transition matrices, the 2024 one-year
# matrix, and annual default/upgrade/downgrade rates by rating category.
# That study is a proprietary PDF report; it is not served by the QuantX
# Dataloader (OHLCV bars) or by the Massive REST API (reference, trades,
# quotes, corporate actions), so its exact published figures cannot be
# retrieved through either data path used by this replication (see
# results/RESULTS.md, "Deviations from the paper").
#
# ``stylized_sp_transition_matrix`` generates a seven-category rating system
# (AAA, AA, A, BBB, BB, B, CCC/C, plus an absorbing Default state) with the
# qualitative shape the paper describes for agency data: monotone decay of
# credit quality, a one-year default probability that increases sharply in
# the lower categories, and small off-diagonal upward jumps into CCC/C from
# the top categories that break strict stochastic monotonicity (page 46:
# "largest violation ... caused by small upward-jump frequencies of the top
# categories into CCC/C"). The paper's own calibration transforms (scaling
# the default column, replacing it with regime-specific mean default rates)
# are implemented in src/rating_data.py and applied to this base matrix.
# ---------------------------------------------------------------------------

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]
N_RATINGS = len(RATINGS)  # non-default categories
DEFAULT_INDEX = 0  # index 0 of the full (m+1)-state chain is Default
# Full state ordering used everywhere: [Default, AAA, ..., CCC/C]


def stylized_sp_transition_matrix() -> np.ndarray:
    """Stylized one-year migration matrix (rows/cols: Default, AAA..CCC/C).

    Off-diagonal migration intensities decay geometrically with rating
    distance, upper categories are more stable than lower ones, and default
    probabilities increase sharply towards CCC/C -- the qualitative pattern
    described in Sec. 5.1. Rows sum to one by construction.
    """
    n = N_RATINGS + 1
    # one-year default probability by rating (illustrative, increasing down the scale)
    default_prob = np.array([0.0000, 0.0002, 0.0006, 0.0018, 0.0070, 0.0400, 0.2600])
    # stability (probability of staying in the same rating) by rating
    stay_prob = np.array([0.92, 0.90, 0.88, 0.85, 0.80, 0.72, 0.55])
    mat = np.zeros((n, n))
    mat[DEFAULT_INDEX, DEFAULT_INDEX] = 1.0  # default is absorbing
    for i in range(N_RATINGS):
        row = np.zeros(n)
        row[DEFAULT_INDEX] = default_prob[i]
        row[i + 1] = stay_prob[i]
        remaining = 1.0 - default_prob[i] - stay_prob[i]
        # distribute the remainder geometrically over the other non-default ratings
        others = [j for j in range(N_RATINGS) if j != i]
        weights = np.array([0.5 ** abs(i - j) for j in others], dtype=float)
        # small upward-jump component into CCC/C from the top ratings, breaking
        # strict stochastic monotonicity as described on page 46
        if i <= 2:
            jump_target = N_RATINGS - 1
            if jump_target in others:
                weights[others.index(jump_target)] += 0.6
        weights = weights / weights.sum()
        for w, j in zip(weights, others):
            row[j + 1] += remaining * w
        mat[i + 1] = row
    assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-10)
    return mat
