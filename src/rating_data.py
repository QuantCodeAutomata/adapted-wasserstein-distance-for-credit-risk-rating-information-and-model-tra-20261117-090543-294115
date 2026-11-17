"""Stylized rating-transition data used as the market model in exp_2, exp_6,
exp_8, exp_9 and exp_10.

Section 5 of the paper computes its empirical illustrations from the S&P
Global Ratings 2024 annual corporate default and transition study [69]:
global/regional one-year average transition matrices, the 2024 one-year
matrix, and annual default/upgrade/downgrade rates by rating category. That
study is a proprietary PDF report; it is not served by the QuantX Dataloader
(OHLCV bars) or by the Massive REST API (reference/trades/quotes/corporate
actions), so its exact published figures cannot be retrieved through either
data path used by this replication.

This module instead builds a *stylized* seven-category rating system
(AAA, AA, A, BBB, BB, B, CCC/C, plus an absorbing Default state) with the
qualitative shape the paper describes for agency data: monotone decay of
credit quality, a one-year default probability that increases sharply in the
lower categories, and small off-diagonal upward jumps into CCC/C from the
top categories that break strict stochastic monotonicity (page 46: "largest
violation ... caused by small upward-jump frequencies of the top categories
into CCC/C"). The regional, vintage and cycle variants used in exp_8 are
built from this single base matrix by the paper's own transforms (scaling
the default column, replacing it with regime-specific mean default rates,
and scaling migration intensities), so that every pairwise comparison
remains calibration-consistent even though the base levels are illustrative
rather than the agency's published numbers. This substitution, and its
reason, is recorded in the "Deviations from the paper" section of
results/RESULTS.md.
"""
from __future__ import annotations

import numpy as np

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]
N_RATINGS = len(RATINGS)  # non-default categories
DEFAULT_INDEX = 0  # index 0 of the full (m+1)-state chain is Default
# Full state ordering used everywhere in this module: [Default, AAA, ..., CCC/C]


def _base_generator() -> np.ndarray:
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


def regional_variant(base: np.ndarray, scale: float) -> np.ndarray:
    """Scale the default column by ``scale`` and renormalize the remaining
    transitions, as described on page 45: "we calibrate the second system by
    scaling its default column by a constant and renormalizing the remaining
    transitions, in the spirit of the risk-premium adjustment of [45]."
    """
    mat = base.copy()
    n = mat.shape[0]
    for i in range(1, n):
        d = mat[i, DEFAULT_INDEX] * scale
        d = min(d, 0.999)
        remaining_mass = 1.0 - d
        old_remaining = 1.0 - mat[i, DEFAULT_INDEX]
        mat[i, DEFAULT_INDEX] = d
        for j in range(n):
            if j != DEFAULT_INDEX:
                mat[i, j] = mat[i, j] * remaining_mass / old_remaining
    return mat


def cycle_variant(base: np.ndarray, regime_default_rate: np.ndarray, migration_scale: float) -> np.ndarray:
    """Replace the default column with regime-specific mean default rates and
    scale upgrades/downgrades by ``migration_scale`` (page 46): "obtained from
    the average one by replacing the default column with the mean default
    rates of expansion and recession years, and by scaling upgrades and
    downgrades by the ratio of the regime mean to the long-run ... rates."
    """
    mat = base.copy()
    n = mat.shape[0]
    for i in range(1, n):
        old_default = mat[i, DEFAULT_INDEX]
        new_default = regime_default_rate[i - 1]
        old_remaining = 1.0 - old_default
        # migration (off-diagonal, non-default) mass is scaled by migration_scale,
        # the diagonal (staying in rating) absorbs the complement so rows sum to 1
        off_diag_mass = sum(mat[i, j] for j in range(n) if j not in (DEFAULT_INDEX, i)) * migration_scale
        stay = 1.0 - new_default - off_diag_mass
        mat[i, DEFAULT_INDEX] = new_default
        mat[i, i] = max(stay, 0.0)
        off_diag_idx = [j for j in range(n) if j not in (DEFAULT_INDEX, i)]
        base_off = np.array([mat[i, j] for j in off_diag_idx])
        base_off_sum = base_off.sum()
        if base_off_sum > 0:
            scaled = base_off / base_off_sum * off_diag_mass
            for j, v in zip(off_diag_idx, scaled):
                mat[i, j] = v
        row_sum = mat[i].sum()
        mat[i] = mat[i] / row_sum
    return mat


def sp_style_systems() -> dict:
    """Build the stylized rating systems used by exp_8, exp_2, exp_6, exp_9, exp_10.

    Returns a dict with keys: ``base`` (one-year matrix), ``us``, ``europe``,
    ``long_run``, ``vintage_2024``, ``expansion``, ``recession``. All matrices
    have shape (n+1, n+1) with state 0 = Default, states 1..n = AAA..CCC/C.
    """
    base = _base_generator()
    us = base
    # Europe: historically lower default rates at the same ratings (page 46:
    # "requires scaling European default probabilities by factors between
    # about 1.4 and 3" when calibrating to the same bond price) -- here we
    # build Europe with lower raw default rates so the paper's calibration
    # step (regional_variant) must scale it up by a comparable factor.
    europe = base.copy()
    for i in range(1, europe.shape[0]):
        scale_down = 1.0 / np.linspace(3.0, 1.4, N_RATINGS)[i - 1]
        d = europe[i, DEFAULT_INDEX] * scale_down
        old_remaining = 1.0 - europe[i, DEFAULT_INDEX]
        new_remaining = 1.0 - d
        europe[i, DEFAULT_INDEX] = d
        for j in range(europe.shape[0]):
            if j != DEFAULT_INDEX:
                europe[i, j] *= new_remaining / old_remaining
    long_run = base
    vintage_2024 = base.copy()
    # 2024 vintage: slightly higher downgrade intensity than the long-run average
    for i in range(1, vintage_2024.shape[0] - 1):
        vintage_2024[i, i] *= 0.97
        vintage_2024[i, i + 1] += vintage_2024[i, i] * 0.0  # placeholder keeps shape explicit
    vintage_2024 = vintage_2024 / vintage_2024.sum(axis=1, keepdims=True)

    long_run_default = base[1:, DEFAULT_INDEX]
    expansion_default = long_run_default * 0.5
    recession_default = long_run_default * 2.2
    expansion = cycle_variant(base, expansion_default, migration_scale=0.8)
    recession = cycle_variant(base, recession_default, migration_scale=1.3)
    return {
        "base": base,
        "us": us,
        "europe": europe,
        "long_run": long_run,
        "vintage_2024": vintage_2024,
        "expansion": expansion,
        "recession": recession,
    }
