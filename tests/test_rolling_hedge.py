"""Unit tests for src/rolling_hedge.py (Sec. 4.7-4.8)."""
import numpy as np
import pytest

from src.rolling_hedge import (
    hazard_from_default_prob,
    renormalized_non_default_transitions,
    bootstrap_rates,
    bootstrap_cumulative_hazard,
    rolling_positions,
    simulate_review_ratings,
    simulate_default_times_given_ratings,
    survival_weighted_kernel,
    krylov_matrix,
    market_desk_weights,
)
from src.rating_data import sp_style_systems


def test_hazard_from_default_prob_matches_log_formula():
    d = np.array([0.01, 0.05])
    h = hazard_from_default_prob(d)
    assert np.allclose(h, -np.log(1 - d))


def test_renormalized_transitions_rows_sum_to_one():
    P = sp_style_systems()["base"]
    Q = renormalized_non_default_transitions(P)
    assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(Q >= 0)


def test_bootstrap_rates_reproduce_survival_curve():
    pins = np.array([0.0, 1.0, 2.0, 3.0])
    survival = np.array([1.0, 0.97, 0.93, 0.88])
    betas = bootstrap_rates(survival, pins)
    B = bootstrap_cumulative_hazard(betas, pins)
    reconstructed = np.exp(-B)
    assert np.allclose(reconstructed, survival, atol=1e-10)


def test_rolling_positions_terminal_value_is_one():
    B = np.array([0.0, 0.05, 0.12, 0.25])
    N = rolling_positions(B)
    assert N[-1] == pytest.approx(1.0)
    # N_k = exp(-(B_T - B_{t_k})) <= 1 and increases towards 1 as t_k -> T
    assert np.all(N <= 1.0 + 1e-12)
    assert np.all(np.diff(N) >= -1e-12)


def test_simulate_review_ratings_absorbing_default():
    P = sp_style_systems()["base"]
    Q = renormalized_non_default_transitions(P)
    rng = np.random.default_rng(0)
    h = hazard_from_default_prob(P[1:, 0])
    ratings = simulate_review_ratings(Q, start_rating=5, K=7, n_paths=2000, rng=rng)
    assert ratings.shape == (2000, 8)
    defaulted = ratings == -1
    # once defaulted (-1), a path must remain -1 (absorbing)
    for k in range(1, 8):
        assert np.all(defaulted[defaulted[:, k - 1], k])


def test_default_times_consistent_with_hazard_level():
    P = sp_style_systems()["base"]
    h = hazard_from_default_prob(P[1:, 0])
    rng = np.random.default_rng(1)
    pins = np.arange(8, dtype=float)
    ratings = np.zeros((5000, 8), dtype=int)  # stay in best rating throughout
    default_times = simulate_default_times_given_ratings(ratings, h, pins, rng)
    survival_7y = np.mean(default_times > 7.0)
    expected = np.exp(-h[0] * 7.0)
    assert survival_7y == pytest.approx(expected, abs=0.02)


def test_krylov_matrix_columns_match_powers():
    K_hat = np.array([[0.8, 0.1], [0.2, 0.7]])
    km = krylov_matrix(K_hat, 3)
    assert np.allclose(km[:, 0], np.ones(2))
    assert np.allclose(km[:, 1], K_hat @ np.ones(2))
    assert np.allclose(km[:, 2], K_hat @ (K_hat @ np.ones(2)))


def test_market_desk_weights_exact_match_when_m_equals_dimension():
    K_hat = np.array([[0.8, 0.1], [0.2, 0.7]])
    theta = market_desk_weights(K_hat, k=0, K_total=2, m=2)
    basis = krylov_matrix(K_hat, 2)
    target = K_hat @ np.ones(2)
    assert np.allclose(basis @ theta, target, atol=1e-8)
