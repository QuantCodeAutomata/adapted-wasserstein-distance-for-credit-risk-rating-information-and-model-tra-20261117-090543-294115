"""Unit tests for src/single_review.py (Sec. 3.1-3.2)."""
import numpy as np
import pytest

from src.single_review import (
    aw2_squared_single_review,
    w2_squared_bounds_single_review,
    w2_squared_exact,
    symmetric_two_point_distances,
    timing_distance,
    no_information_benchmark_distance,
    calibration_preserving_map,
    estimate_aw2_squared,
)
from data.generate_synthetic import (
    two_point_posterior,
    beta_posterior,
    independent_posterior_pair,
)


def test_aw2_equals_mean_abs_deviation():
    rng = np.random.default_rng(0)
    p = 0.9
    Q = beta_posterior(p, 5.0, 200_000, rng)
    assert abs(float(np.mean(Q)) - p) < 1e-2
    aw2_sq = aw2_squared_single_review(Q, p)
    assert aw2_sq == pytest.approx(float(np.mean(np.abs(Q - p))), abs=1e-12)
    assert aw2_sq > 0


def test_w2_bounds_bracket_and_equal_small_deviation():
    rng = np.random.default_rng(1)
    p = 0.9
    delta = 0.05
    Q = two_point_posterior(p, delta, 100_000, rng)
    lower, upper = w2_squared_bounds_single_review(Q, p)
    assert lower <= upper
    assert lower == pytest.approx(delta ** 2, rel=1e-6)
    assert upper == pytest.approx(2 * delta ** 2, rel=1e-6)
    exact = w2_squared_exact(Q, p)
    assert exact == pytest.approx(upper, rel=1e-6)


def test_symmetric_two_point_sqrt_blowup():
    p = 0.9
    small = symmetric_two_point_distances(p, 0.001)
    large = symmetric_two_point_distances(p, 0.05)
    assert small["ratio"] == pytest.approx(small["predicted_ratio"])
    assert large["ratio"] == pytest.approx(large["predicted_ratio"])
    # the ratio AW2/W2 diverges as delta -> 0 (Corollary 3.9)
    assert small["ratio"] > large["ratio"]


def test_timing_distance_between_jensen_and_triangle_bounds():
    rng = np.random.default_rng(2)
    p = 0.85
    Q, Q_tilde = independent_posterior_pair(p, 5.0, 300_000, rng)
    e_abs_qp = float(np.mean(np.abs(Q - p)))
    dist = timing_distance(Q, Q_tilde)
    assert dist >= e_abs_qp - 5e-3  # Jensen: E|Q-Qtilde| >= E|Q-p| (approx, MC noise)
    assert dist <= 2 * e_abs_qp + 5e-3  # triangle inequality


def test_no_information_benchmark_nonnegative():
    rng = np.random.default_rng(3)
    p = 0.8
    Y = beta_posterior(p, 4.0, 50_000, rng)
    dist = no_information_benchmark_distance(Y, p)
    assert dist >= 0


def test_calibration_preserving_map_reproduces_mean():
    rng = np.random.default_rng(4)
    p = 0.7
    samples = beta_posterior(0.6, 3.0, 20_000, rng)  # sample mean != p
    m, estimate = estimate_aw2_squared(samples, p)
    calibrated = calibration_preserving_map(samples, m, p)
    assert float(np.mean(calibrated)) == pytest.approx(p, abs=1e-9)
    assert estimate > 0


def test_calibration_map_contracts_when_m_above_p():
    q = np.array([0.0, 0.5, 1.0])
    out = calibration_preserving_map(q, m=0.8, p=0.4)
    assert np.all(out <= q + 1e-12)  # contraction towards 0 when m >= p


def test_calibration_map_stretches_when_m_below_p():
    q = np.array([0.0, 0.5, 1.0])
    out = calibration_preserving_map(q, m=0.3, p=0.6)
    assert np.all(out >= q - 1e-12)  # stretching towards 1 when m < p


def test_estimation_error_decays_like_inverse_sqrt_n():
    rng = np.random.default_rng(5)
    p = 0.9
    true_mu = beta_posterior(p, 5.0, 2_000_000, rng)
    true_value = float(np.mean(np.abs(true_mu - p)))
    errors = {}
    for n in (200, 2000, 20000):
        reps = [
            abs(estimate_aw2_squared(beta_posterior(p, 5.0, n, rng), p)[1] - true_value)
            for _ in range(80)
        ]
        errors[n] = np.mean(reps)
    # error should shrink as n grows (O(n^{-1/2}) rate, Theorem 3.10)
    assert errors[20000] < errors[200]
    assert errors[2000] < errors[200] * 2
