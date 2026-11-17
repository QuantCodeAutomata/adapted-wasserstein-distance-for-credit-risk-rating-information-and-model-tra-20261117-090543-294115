"""Unit tests for src/cox_hazard.py (Sec. 4.1-4.3)."""
import numpy as np
import pytest

from src.cox_hazard import (
    cumulative_hazard,
    psi_mismatch,
    psi_from_paths,
    common_shock_coupling_mismatch,
    sharp_range,
    equal_calibration_range,
    adaptation_slack_decomposition,
    l1_stability_check,
    simulate_cir,
    default_layer_reduction,
)


def test_psi_mismatch_zero_for_identical_hazards():
    t = np.linspace(0, 5, 500)
    a = np.full_like(t, 0.05)
    psi = psi_from_paths(a, a, t)
    assert psi == pytest.approx(0.0, abs=1e-10)


def test_psi_mismatch_matches_frechet_bound_for_ordered_hazards():
    t = np.linspace(0, 5, 500)
    a = np.full_like(t, 0.03)
    b = np.full_like(t, 0.08)  # b >= a everywhere, so max(a,b)=b and M=B
    psi = psi_from_paths(a, b, t)
    A_T, B_T = 0.03 * 5, 0.08 * 5
    expected = np.exp(-A_T) + np.exp(-B_T) - 2 * np.exp(-B_T)
    assert psi == pytest.approx(expected, rel=1e-6)
    assert psi == pytest.approx(abs(np.exp(-A_T) - np.exp(-B_T)), rel=1e-6)


def test_common_shock_coupling_matches_closed_form():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 5, 500)
    a = np.full_like(t, 0.04)
    b = np.full_like(t, 0.07)
    closed_form = psi_from_paths(a, b, t)
    mc = common_shock_coupling_mismatch(a, b, t, 300_000, rng)
    assert mc == pytest.approx(closed_form, abs=5e-3)


def test_sharp_range_lower_le_upper_and_matches_frechet():
    h_a, h_b, L, T = 0.15, 0.3, 0.25, 5.0
    lower, upper = sharp_range(h_a, h_b, L, T)
    assert lower <= upper
    assert lower == pytest.approx(abs(np.exp(-h_a) - np.exp(-h_b)))


def test_equal_calibration_range_zero_lower_bound():
    lower, upper = equal_calibration_range(0.2, 0.25, 5.0)
    assert lower == 0.0
    assert upper > 0


def test_adaptation_slack_decomposition_consistency():
    t = np.linspace(0, 5, 200)
    a = 0.02 + 0.01 * np.sin(t)
    b = 0.05 + 0.02 * np.cos(t)
    A = cumulative_hazard(a, t)
    B = cumulative_hazard(b, t)
    M = cumulative_hazard(np.maximum(a, b), t)
    out = adaptation_slack_decomposition(A, B, M)
    assert np.all(out["slack"] >= -1e-9)
    assert np.all(out["kappa_t"] >= -1e-9)


def test_l1_stability_bound_holds_and_is_sharp_for_small_epsilon():
    t = np.linspace(0, 5, 2000)
    a = np.zeros_like(t)
    b = np.zeros_like(t)
    for eps in (1e-1, 1e-2, 1e-3):
        a_tilde = np.full_like(t, eps / 5.0)
        out = l1_stability_check(a, b, a_tilde, t)
        assert out["bound_holds"]
        # sharpness: lhs/rhs -> 1 as eps -> 0 (Theorem 4.9)
        assert out["lhs"] / out["rhs"] > 0.9


def test_cir_simulation_nonnegative_and_mean_reverts():
    rng = np.random.default_rng(1)
    t, lam = simulate_cir(kappa=0.3, theta=0.03, lambda0=0.02, sigma=0.08,
                           T=5.0, dt=1 / 250, n_paths=2000, rng=rng)
    assert np.all(lam >= 0)
    assert abs(float(np.mean(lam[:, -1])) - 0.03) < 0.02


def test_default_layer_reduction_matches_deterministic_case():
    t = np.linspace(0, 5, 200)
    a_paths = np.tile(np.full_like(t, 0.03), (10, 1))
    b_paths = np.tile(np.full_like(t, 0.07), (10, 1))
    e_psi = default_layer_reduction(a_paths, b_paths, t)
    closed_form = psi_from_paths(a_paths[0], b_paths[0], t)
    assert e_psi == pytest.approx(closed_form, rel=1e-6)
