"""Robust pricing over adapted Wasserstein balls and small-dispersion
asymptotics for rating-dependent hazards (Sec. 4.4, Theorem 4.30, 4.35,
Corollary 4.31, 4.36, Proposition 4.29, 4.32).
"""
from __future__ import annotations

import numpy as np


def robust_call_premium(p: float, eta: float, k: float) -> float:
    """sup_{mu in U_eta(p)} E_mu[(Q-k)+] (Theorem 4.30):
    k<=p: p-k+eta*k/(2p); k>=p: eta*(1-k)/(2(1-p))."""
    assert 0 <= eta <= 2 * p * (1 - p) + 1e-12, "eta must be attainable: eta <= 2p(1-p)"
    if k <= p:
        return p - k + eta * k / (2 * p)
    return eta * (1 - k) / (2 * (1 - p))


def extremizer_mu(p: float, eta: float) -> dict:
    """mu_eta* = eta/(2p) * delta_0 + [1 - eta/(2p) - eta/(2(1-p))] * delta_p
    + eta/(2(1-p)) * delta_1 (Eq. 4.56)."""
    w0 = eta / (2 * p)
    w1 = eta / (2 * (1 - p))
    wp = 1.0 - w0 - w1
    assert wp >= -1e-12, "eta too large for a valid three-point extremizer"
    weights = np.array([max(w0, 0.0), max(wp, 0.0), max(w1, 0.0)])
    support = np.array([0.0, p, 1.0])
    mean = float(np.sum(weights * support))
    mad = float(np.sum(weights * np.abs(support - p)))
    return {"support": support, "weights": weights, "mean": mean, "mad": mad}


def robust_premium_above_coarse(eta: float, p: float, k: float) -> float:
    """Corollary 4.31: robust premium above the coarse price (p-k)+ is
    rho^2*k/(2p) for k<=p and rho^2*(1-k)/(2(1-p)) for k>=p, with rho^2=eta."""
    coarse = max(p - k, 0.0)
    return robust_call_premium(p, eta, k) - coarse


def mad_lipschitz_bound(g, p: float, mu_support: np.ndarray, mu_weights: np.ndarray, L: float) -> dict:
    """Proposition 4.32: |E_mu[g] - g(p)| <= L * eta where eta = E_mu|Q-p|."""
    eta = float(np.sum(mu_weights * np.abs(mu_support - p)))
    lhs = abs(float(np.sum(mu_weights * g(mu_support))) - g(p))
    rhs = L * eta
    return {"eta": eta, "lhs": lhs, "rhs": rhs, "bound_holds": lhs <= rhs + 1e-9}


def small_dispersion_small_sample(
    p: float, Delta: float, zeta_values: np.ndarray, zeta_probs: np.ndarray, epsilon: float,
) -> dict:
    """Theorem 4.35: calibrated perturbation lambda_R^eps = lambda_* + eps*zeta + c_eps,
    where lambda_* = -log(p)/Delta and c_eps solves the exact calibration
    E[exp(-Delta*lambda_R^eps)] = p (Eq. 4.61, solved exactly here rather
    than only to second order, since zeta is a finite discrete law).
    Returns Q_eps = exp(-Delta*lambda_R^eps) on the support of zeta with its
    probabilities, together with AW_2 and W_2 computed from this exact law.
    """
    lambda_star = -np.log(p) / Delta
    # exact calibration: find c such that sum_z probs(z) * p * exp(-Delta*eps*z - Delta*c) = p
    # i.e. exp(-Delta*c) = 1 / E[exp(-Delta*eps*zeta)]
    mgf = float(np.sum(zeta_probs * np.exp(-Delta * epsilon * zeta_values)))
    c_eps = np.log(mgf) / Delta
    lambda_R = lambda_star + epsilon * zeta_values + c_eps
    Q_eps = np.exp(-Delta * lambda_R)
    mean_Q = float(np.sum(zeta_probs * Q_eps))
    assert abs(mean_Q - p) < 1e-9, "calibration must reproduce the initial price p exactly"
    aw2_sq = float(np.sum(zeta_probs * np.abs(Q_eps - p)))
    aw2 = np.sqrt(aw2_sq)
    w2_sq = 2.0 * float(np.sum(zeta_probs * (Q_eps - p) ** 2)) if np.all(np.abs(Q_eps - p) <= 0.5) else np.nan
    w2 = np.sqrt(w2_sq) if not np.isnan(w2_sq) else np.nan
    return {"c_eps": c_eps, "Q_eps": Q_eps, "aw2": aw2, "w2": w2, "aw2_sq": aw2_sq, "w2_sq": w2_sq}


def predicted_asymptotics(p: float, Delta: float, e_abs_zeta: float, e_zeta_sq: float, epsilon: float) -> dict:
    """AW_2 ~ sqrt(p*Delta*E|zeta|)*|eps|^{1/2}; W_2 ~ sqrt(2)*p*Delta*sqrt(E[zeta^2])*|eps| (Theorem 4.35)."""
    aw2_pred = np.sqrt(p * Delta * e_abs_zeta) * np.sqrt(abs(epsilon))
    w2_pred = np.sqrt(2.0) * p * Delta * np.sqrt(e_zeta_sq) * abs(epsilon)
    ratio_pred = np.sqrt(e_abs_zeta) / (np.sqrt(2 * p * Delta) * np.sqrt(e_zeta_sq)) * abs(epsilon) ** (-0.5)
    return {"aw2_pred": float(aw2_pred), "w2_pred": float(w2_pred), "ratio_pred": float(ratio_pred)}
