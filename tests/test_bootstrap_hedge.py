"""Unit tests for src/bootstrap_hedge.py (Sec. 4.5-4.6, Example 4.22)."""
import numpy as np
import pytest

from src.bootstrap_hedge import (
    calibrate_bootstrap,
    roll_residual,
    expected_psi,
    survival_filtered_projection,
    claim_weighted_residual,
    claim_weighted_bound,
)

HAZARDS_L = np.array([0.02, 0.06, 0.20])
PROBS = np.array([0.5, 0.3, 0.2])
T1, T, XI = 0.0, 5.0, 0.0


def test_example_4_22_calibration_matches_paper_numbers():
    out = calibrate_bootstrap(HAZARDS_L, PROBS, T1, T, XI)
    assert out["F"] == pytest.approx(0.748240, abs=1e-5)
    assert out["beta"] == pytest.approx(0.058006, abs=1e-5)


def test_roll_residual_identity_e_abs_epsilon_equals_phi_T():
    out = roll_residual(HAZARDS_L, PROBS, T1, T, XI)
    assert out["E_abs_epsilon"] == pytest.approx(0.156597, abs=1e-5)
    assert out["E_abs_epsilon"] == pytest.approx(out["phi_T"], abs=1e-9)


def test_expected_psi_equals_roll_residual_for_flat_bootstrap():
    roll = roll_residual(HAZARDS_L, PROBS, T1, T, XI)
    psi = expected_psi(HAZARDS_L, PROBS, T1, T, XI, roll["B_T"])
    # Theorem 4.16 equality: flat rating-dependent hazards -> ordered post-review
    # hazards vs. a single flat bootstrap hazard, so E[phi]=E[Psi]
    assert psi["E_Psi"] == pytest.approx(roll["E_abs_epsilon"], abs=1e-5)
    assert psi["slack"] == pytest.approx(0.0, abs=1e-6)


def test_survival_filtered_projection_reproduces_survival_curve():
    out = survival_filtered_projection(HAZARDS_L, PROBS, T1, T, XI)
    assert np.allclose(out["S_u"], out["S_from_h"], atol=2e-3)
    assert out["h_u"][0] == pytest.approx(float(np.sum(PROBS * HAZARDS_L)), abs=1e-6)


def test_filtered_projection_generates_positive_adaptation_slack():
    proj = survival_filtered_projection(HAZARDS_L, PROBS, T1, T, XI)
    B_T_filtered = -np.log(proj["S_u"][-1])
    psi_filtered = expected_psi(HAZARDS_L, PROBS, T1, T, XI, B_T_filtered)
    roll = roll_residual(HAZARDS_L, PROBS, T1, T, XI)
    assert psi_filtered["E_Psi"] >= roll["E_abs_epsilon"] - 1e-6
    assert psi_filtered["slack"] >= -1e-9


def test_claim_weighted_residual_matches_direct_difference():
    P_A = np.array([0.9, 0.8])
    P_B = np.array([0.88, 0.81])
    nu = np.array([100.0, 50.0])
    out = claim_weighted_residual(P_A, P_B, nu)
    expected = (P_B[0] - P_A[0]) * 100.0 + (P_B[1] - P_A[1]) * 50.0
    assert out == pytest.approx(expected)


def test_claim_weighted_bound_nonnegative():
    bound = claim_weighted_bound(
        HAZARDS_L, PROBS, T1, np.array([1.0, 3.0, 5.0]), XI, 0.058006,
        np.array([1.0, 1.0, 1.0]),
    )
    assert bound >= 0
