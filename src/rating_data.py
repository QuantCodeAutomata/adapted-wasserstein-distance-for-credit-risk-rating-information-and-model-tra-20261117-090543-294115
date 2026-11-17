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

This module builds the regional/vintage/cycle *variants* used by exp_2,
exp_6, exp_8, exp_9 and exp_10 by applying the paper's own calibration
transforms (scaling the default column, replacing it with regime-specific
mean default rates, and scaling migration intensities) to a single stylized
base matrix generated in ``data/generate_synthetic.py``
(``stylized_sp_transition_matrix``), so that every pairwise comparison
remains calibration-consistent even though the base levels are illustrative
rather than the agency's published numbers. This substitution, and its
reason, is recorded in the "Deviations from the paper" section of
results/RESULTS.md.
"""
from __future__ import annotations

import numpy as np

from data.generate_synthetic import (
    DEFAULT_INDEX,
    N_RATINGS,
    RATINGS,
    stylized_sp_transition_matrix,
)

__all__ = [
    "RATINGS",
    "N_RATINGS",
    "DEFAULT_INDEX",
    "regional_variant",
    "cycle_variant",
    "sp_style_systems",
]


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
    base = stylized_sp_transition_matrix()
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
    # (a one-notch-down shift of part of the diagonal mass), rows renormalized to 1
    n_full = vintage_2024.shape[0]
    for i in range(1, n_full - 1):
        shifted = vintage_2024[i, i] * 0.03
        vintage_2024[i, i] -= shifted
        vintage_2024[i, i + 1] += shifted
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
