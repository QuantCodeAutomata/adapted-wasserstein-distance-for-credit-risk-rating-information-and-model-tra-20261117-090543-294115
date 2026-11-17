"""Unit tests for src/transition_utils.py (Sec. 3.3, Theorems 3.16, 3.18)."""
import numpy as np
import pytest

from src.transition_utils import (
    survival_probabilities,
    price_kernels,
    backward_dp_adapted_distance,
    synchronous_coupling_default_mismatch,
    stochastic_monotonicity_violation,
    jafry_schuermann_mobility,
)
from src.rating_data import sp_style_systems, N_RATINGS


def test_survival_probabilities_terminal_condition():
    systems = sp_style_systems()
    P = systems["base"]
    q = survival_probabilities(P, N=5)
    assert np.array_equal(q[5], (np.arange(P.shape[0]) != 0).astype(float))
    assert q[0, 0] == 0.0  # default state never survives


def test_dp_distance_zero_for_identical_models():
    systems = sp_style_systems()
    P = systems["base"]
    N = 3
    r = 3  # some initial rating index
    q = survival_probabilities(P, N)
    state_p, kern_p = price_kernels(P, q, N, r)
    aw2_sq = backward_dp_adapted_distance(state_p, kern_p, state_p, kern_p, N)
    assert aw2_sq == pytest.approx(0.0, abs=1e-10)


def test_dp_distance_matches_synchronous_coupling_when_monotone():
    rng = np.random.default_rng(0)
    # Build matrices that are stochastically monotone under the raw state
    # index convention [Default, AAA(best), ..., CCC/C(worst)] used by
    # stochastic_monotonicity_violation: start from matrices monotone under
    # the reverse convention (index 1 worst ... index n-1 best) and permute
    # indices 1 and 3 to match the function's expected ordering.
    P_rev = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.10, 0.70, 0.15, 0.05],
        [0.03, 0.10, 0.75, 0.12],
        [0.01, 0.02, 0.07, 0.90],
    ])
    Q_rev = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.12, 0.65, 0.18, 0.05],
        [0.04, 0.12, 0.70, 0.14],
        [0.02, 0.03, 0.10, 0.85],
    ])
    perm = [0, 3, 2, 1]

    def permute(mat: np.ndarray) -> np.ndarray:
        return mat[np.ix_(perm, perm)]

    P = permute(P_rev)
    Qmat = permute(Q_rev)
    assert stochastic_monotonicity_violation(P) == pytest.approx(0.0, abs=1e-9)
    assert stochastic_monotonicity_violation(Qmat) == pytest.approx(0.0, abs=1e-9)
    N = 2
    r = 2
    q_p = survival_probabilities(P, N)
    q_q = survival_probabilities(Qmat, N)
    state_p, kern_p = price_kernels(P, q_p, N, r)
    state_q, kern_q = price_kernels(Qmat, q_q, N, r)
    aw2_sq = backward_dp_adapted_distance(state_p, kern_p, state_q, kern_q, N)
    mc = synchronous_coupling_default_mismatch(P, Qmat, N, 400_000, rng, initial_rating=r)
    assert mc == pytest.approx(aw2_sq, abs=1e-2)


def test_monotonicity_violation_nonnegative_for_stylized_system():
    systems = sp_style_systems()
    v = stochastic_monotonicity_violation(systems["base"])
    assert v >= 0.0
    assert v < 0.05  # the paper reports violations of at most 0.0016 for S&P data


def test_jafry_schuermann_mobility_nonnegative():
    systems = sp_style_systems()
    m = jafry_schuermann_mobility(systems["base"])
    assert m >= 0
