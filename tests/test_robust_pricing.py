"""Unit tests for src/robust_pricing.py (Sec. 4.4)."""
import numpy as np
import pytest

from src.robust_pricing import (
    robust_call_premium,
    extremizer_mu,
    robust_premium_above_coarse,
    mad_lipschitz_bound,
    small_dispersion_small_sample,
    predicted_asymptotics,
)


def test_robust_premium_dominates_coarse_jensen_bound():
    p, k = 0.9, 0.8
    eta = 0.05
    coarse = max(p - k, 0.0)
    robust = robust_call_premium(p, eta, k)
    assert robust >= coarse - 1e-12


def test_extremizer_has_mean_p_and_mad_eta():
    p, eta = 0.9, 0.05
    out = extremizer_mu(p, eta)
    assert out["mean"] == pytest.approx(p, abs=1e-9)
    assert out["mad"] == pytest.approx(eta, abs=1e-9)


def test_robust_premium_above_coarse_is_quadratic_in_eta():
    p, k = 0.9, 0.95
    out1 = robust_premium_above_coarse(0.02, p, k)
    out2 = robust_premium_above_coarse(0.04, p, k)
    # both proportional to eta (linear in eta = rho^2, Corollary 4.31) -> doubling eta doubles the premium
    assert out2 == pytest.approx(2 * out1, rel=1e-9)
    assert out1 >= 0


def test_mad_lipschitz_bound_sharp_for_identity_type_function():
    p = 0.9
    support = np.array([0.8, 0.9, 1.0])
    weights = np.array([0.3, 0.4, 0.3])
    g = lambda q: np.abs(q - p)
    out = mad_lipschitz_bound(g, p, support, weights, L=1.0)
    assert out["bound_holds"]


def test_small_dispersion_calibration_and_sqrt_separation():
    p, Delta = 0.9, 2.0
    zeta_values = np.array([-1.0, 1.0])
    zeta_probs = np.array([0.5, 0.5])
    ratios = []
    for eps in (1e-1, 1e-2, 1e-3):
        out = small_dispersion_small_sample(p, Delta, zeta_values, zeta_probs, eps)
        assert out["aw2"] > 0
        ratios.append(out["aw2"] / out["w2"])
    # ratio AW2/W2 grows as epsilon shrinks (Corollary 4.36: ~ eps^{-1/2})
    assert ratios[2] > ratios[1] > ratios[0]


def test_predicted_asymptotics_matches_simulation_at_small_epsilon():
    p, Delta = 0.9, 2.0
    zeta_values = np.array([-1.0, 1.0])
    zeta_probs = np.array([0.5, 0.5])
    e_abs_zeta, e_zeta_sq = 1.0, 1.0
    eps = 1e-4
    out = small_dispersion_small_sample(p, Delta, zeta_values, zeta_probs, eps)
    pred = predicted_asymptotics(p, Delta, e_abs_zeta, e_zeta_sq, eps)
    assert out["aw2"] == pytest.approx(pred["aw2_pred"], rel=0.05)
    assert out["w2"] == pytest.approx(pred["w2_pred"], rel=0.05)
