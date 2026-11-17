"""Model-transfer hedging in a single calibrated rating-review bucket
(Sec. 4.5-4.6): bootstrap roll error, exact identities, survival-filtered
projection and the claim-weighted bound.
"""
from __future__ import annotations

import numpy as np

from src.cox_hazard import psi_mismatch


def calibrate_bootstrap(hazards_L: np.ndarray, probs: np.ndarray, t1: float, T: float, xi: float) -> dict:
    """Eq. 4.23-4.24: F = E_hat[exp(-Lambda_T)], beta = -log(F)/(T-t1),
    B_u = Xi + (u-t1)*beta for u in [t1,T]. ``hazards_L`` are the flat
    post-review hazards and ``probs`` their probabilities (Example 4.22)."""
    assert abs(probs.sum() - 1.0) < 1e-10
    Lambda_T = xi + hazards_L * (T - t1)
    F = float(np.sum(probs * np.exp(-Lambda_T)))
    beta = -np.log(F) / (T - t1)
    return {"F": F, "beta": beta, "Lambda_T": Lambda_T}


def roll_residual(hazards_L: np.ndarray, probs: np.ndarray, t1: float, T: float, xi: float) -> dict:
    """epsilon = F - exp(-Lambda_T) per factor realization (Eq. 4.30);
    E[1_{tau>t1}|epsilon|] = E|exp(-B_T)-exp(-A_T)| (Theorem 4.32)."""
    calib = calibrate_bootstrap(hazards_L, probs, t1, T, xi)
    F, beta, Lambda_T = calib["F"], calib["beta"], calib["Lambda_T"]
    B_T = xi + (T - t1) * beta
    epsilon = F - np.exp(-Lambda_T)
    e_abs_epsilon = float(np.sum(probs * np.abs(epsilon)))
    phi_T = float(np.sum(probs * np.abs(np.exp(-B_T) - np.exp(-Lambda_T))))
    assert abs(e_abs_epsilon - phi_T) < 1e-9, "Theorem 4.32 identity failed"
    return {**calib, "epsilon": epsilon, "E_abs_epsilon": e_abs_epsilon, "phi_T": phi_T, "B_T": B_T}


def expected_psi(hazards_L: np.ndarray, probs: np.ndarray, t1: float, T: float, xi: float, B_T: float) -> dict:
    """E[Psi_T(a,b)] averaged over the factor (rating) distribution, and the
    decomposition E[phi_T] = E[Psi_T] - 2 E[exp(-max(A,B))(1-exp(-kappa))]
    (Eq. 4.33)."""
    A_T = xi + hazards_L * (T - t1)
    psis = np.array([psi_mismatch(a, B_T, max(a, B_T)) for a in A_T])
    e_psi = float(np.sum(probs * psis))
    phis = np.abs(np.exp(-A_T) - np.exp(-B_T))
    e_phi = float(np.sum(probs * phis))
    slack_terms = e_psi - e_phi
    return {"E_Psi": e_psi, "E_phi": e_phi, "slack": slack_terms}


def survival_filtered_projection(hazards_L: np.ndarray, probs: np.ndarray, t1: float, T: float, xi: float, n_grid: int = 2000) -> dict:
    """h_u = E_hat[lambda_u * exp(-Lambda_u)] / S(u), u in [t1,T]
    (Proposition 4.37). Since hazards are flat per rating, lambda_u = L_i and
    exp(-Lambda_u(i)) = exp(-xi - (u-t1)*L_i) for each branch i.
    Returns the grid, the filtered hazard curve, the survival curve S(u),
    and the crossing time where the filtered hazard equals the middle
    hazard (if it lies in (t1,T))."""
    u = np.linspace(t1, T, n_grid)
    exp_neg_Lambda = np.exp(-xi - np.outer(u - t1, hazards_L))  # shape (n_grid, n_ratings)
    S_u = exp_neg_Lambda @ probs
    numerator = (hazards_L[None, :] * exp_neg_Lambda) @ probs
    h_u = numerator / S_u
    assert np.all(h_u > 0)
    # verify S(u) = exp(-integral_0^u h) up to numerical integration error
    cum_h = np.concatenate(([0.0], np.cumsum((h_u[1:] + h_u[:-1]) / 2 * np.diff(u))))
    S_from_h = np.exp(-xi) * np.exp(-cum_h)
    return {"u": u, "h_u": h_u, "S_u": S_u, "S_from_h": S_from_h}


def claim_weighted_residual(P_t1_A: np.ndarray, P_t1_B: np.ndarray, nu_weights: np.ndarray) -> float:
    """epsilon^nu = sum_u [P_t1^B(u) - P_t1^A(u)] * nu(u) for a linear claim
    with cash-flow measure nu (Theorem 4.23)."""
    return float(np.sum((P_t1_B - P_t1_A) * nu_weights))


def claim_weighted_bound(hazards_L: np.ndarray, probs: np.ndarray, t1: float, maturities: np.ndarray,
                          xi: float, B_rate: float, abs_nu_weights: np.ndarray) -> float:
    """E[Phi_{|nu|}(a,b)] = sum_u |nu|(u) * E[Psi_u(a,b)] for the total
    variation measure |nu| (Theorem 4.23)."""
    total = 0.0
    for u, w in zip(maturities, abs_nu_weights):
        A_u = xi + hazards_L * (u - t1)
        B_u = xi + (u - t1) * B_rate
        psis = np.array([psi_mismatch(a, B_u, max(a, B_u)) for a in A_u])
        total += w * float(np.sum(probs * psis))
    return total
