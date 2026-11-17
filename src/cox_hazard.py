"""Adapted Wasserstein distance between Cox bond prices with deterministic
hazards (Sec. 4.1-4.3): closed-form mismatch, common-shock coupling, sharp
hazard envelope, L^1 stability, adaptation-slack decomposition and the
default-layer reduction for random intensities.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def cumulative_hazard(hazard: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative integral A_t = integral_0^t a(s) ds via the trapezoidal rule."""
    cum = cumulative_trapezoid(hazard, t, initial=0.0)
    return cum


def psi_mismatch(A_T: float, B_T: float, M_T: float) -> float:
    """Psi_T(a,b) = e^{-A_T} + e^{-B_T} - 2 e^{-M_T} (Theorem 4.3)."""
    return float(np.exp(-A_T) + np.exp(-B_T) - 2.0 * np.exp(-M_T))


def psi_from_paths(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> float:
    """Psi_T(a,b) from deterministic hazard paths a(t), b(t) on a common grid."""
    A = cumulative_hazard(a, t)
    B = cumulative_hazard(b, t)
    M = cumulative_hazard(np.maximum(a, b), t)
    return psi_mismatch(A[-1], B[-1], M[-1])


def common_shock_coupling_mismatch(a: np.ndarray, b: np.ndarray, t: np.ndarray, n_paths: int, rng: np.random.Generator) -> float:
    """Common-shock optimal coupling (Eq. 4.10-4.11): three independent
    inhomogeneous Poisson processes with intensities c=min(a,b),
    a_idio=(a-b)+, b_idio=(b-a)+. sigma = first jump of N^c+N^a,
    tau = first jump of N^c+N^b. Returns the Monte Carlo estimate of
    P[1_{sigma>T} != 1_{tau>T}], which should match psi_from_paths(a,b,t).
    """
    T = t[-1]
    c = np.minimum(a, b)
    a_idio = np.maximum(a - b, 0.0)
    b_idio = np.maximum(b - a, 0.0)
    C = cumulative_hazard(c, t)
    A_idio = cumulative_hazard(a_idio, t)
    B_idio = cumulative_hazard(b_idio, t)

    def _first_jump_time(cum_hazard: np.ndarray, grid: np.ndarray, exp_draw: np.ndarray) -> np.ndarray:
        # first time cum_hazard(t) >= exp_draw, else +inf (no jump by T)
        idx = np.searchsorted(cum_hazard, exp_draw, side="left")
        out = np.full(exp_draw.shape, np.inf)
        valid = idx < len(grid)
        out[valid] = grid[idx[valid]]
        return out

    e_c = rng.exponential(size=n_paths)
    e_a = rng.exponential(size=n_paths)
    e_b = rng.exponential(size=n_paths)
    jump_c = _first_jump_time(C, t, e_c)
    jump_a = _first_jump_time(A_idio, t, e_a)
    jump_b = _first_jump_time(B_idio, t, e_b)
    sigma = np.minimum(jump_c, jump_a)
    tau = np.minimum(jump_c, jump_b)
    default_sigma = sigma <= T
    default_tau = tau <= T
    return float(np.mean(default_sigma != default_tau))


def sharp_range(h_a: float, h_b: float, L: float, T: float) -> tuple[float, float]:
    """Sharp range of Psi_T over bounded hazard schedules with fixed
    integrated hazards h_a, h_b, cap L, bucket length T (Theorem 4.7):
    inf = |e^{-h_a}-e^{-h_b}|, sup = e^{-h_a}+e^{-h_b}-2e^{-min(h_a+h_b,LT)}.
    """
    assert 0 <= h_a <= L * T and 0 <= h_b <= L * T, "integrated hazards must be attainable with cap L on [0,T]"
    lower = abs(np.exp(-h_a) - np.exp(-h_b))
    upper = np.exp(-h_a) + np.exp(-h_b) - 2.0 * np.exp(-min(h_a + h_b, L * T))
    return float(lower), float(upper)


def equal_calibration_range(h: float, L: float, T: float) -> tuple[float, float]:
    """Corollary 4.8: for h_a=h_b=h, Psi_T ranges over [0, 2(e^{-h}-e^{-min(2h,LT)})]."""
    upper = 2.0 * (np.exp(-h) - np.exp(-min(2 * h, L * T)))
    return 0.0, float(upper)


def adaptation_slack_decomposition(A_t: np.ndarray, B_t: np.ndarray, M_t: np.ndarray) -> dict:
    """Psi_t = phi_t + 2*e^{-max(A_t,B_t)}*(1-e^{-kappa_t}),
    phi_t = |e^{-A_t}-e^{-B_t}|, kappa_t = M_t - max(A_t,B_t) (Proposition 4.6)."""
    max_AB = np.maximum(A_t, B_t)
    kappa_t = M_t - max_AB
    phi_t = np.abs(np.exp(-A_t) - np.exp(-B_t))
    slack = 2.0 * np.exp(-max_AB) * (1.0 - np.exp(-kappa_t))
    psi_t = phi_t + slack
    psi_direct = np.exp(-A_t) + np.exp(-B_t) - 2.0 * np.exp(-M_t)
    assert np.allclose(psi_t, psi_direct, atol=1e-10)
    return {"phi_t": phi_t, "kappa_t": kappa_t, "slack": slack, "psi_t": psi_t}


def l1_stability_check(a: np.ndarray, b: np.ndarray, a_tilde: np.ndarray, t: np.ndarray) -> dict:
    """|Psi_T(a,b) - Psi_T(a_tilde,b)| <= integral |a-a_tilde| (Theorem 4.9)."""
    psi_1 = psi_from_paths(a, b, t)
    psi_2 = psi_from_paths(a_tilde, b, t)
    lhs = abs(psi_1 - psi_2)
    rhs = float(np.trapz(np.abs(a - a_tilde), t))
    return {"lhs": lhs, "rhs": rhs, "bound_holds": lhs <= rhs + 1e-9}


def simulate_cir(kappa: float, theta: float, lambda0: float, sigma: float,
                  T: float, dt: float, n_paths: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Euler-Maruyama simulation of d lambda = kappa(theta-lambda)dt + sigma*sqrt(lambda)*dW.
    Returns (time grid, lambda paths of shape (n_paths, n_steps+1))."""
    n_steps = int(round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    lam = np.zeros((n_paths, n_steps + 1))
    lam[:, 0] = lambda0
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        dW = rng.normal(size=n_paths) * sqrt_dt
        drift = kappa * (theta - lam[:, k]) * dt
        diffusion = sigma * np.sqrt(np.maximum(lam[:, k], 0.0)) * dW
        lam[:, k + 1] = np.maximum(lam[:, k] + drift + diffusion, 0.0)  # full truncation
    return t, lam


def default_layer_reduction(lam_paths_a: np.ndarray, lam_paths_b: np.ndarray, t: np.ndarray) -> float:
    """E[Psi_T(a,b)] for random intensity paths a,b sharing the factor grid t
    (Theorem 4.11): average the pathwise closed-form mismatch over factor
    realizations, which is the value of the adapted transport problem between
    the factor laws when the coupling is doubly stochastic."""
    A_T = cumulative_hazard(lam_paths_a, t) if lam_paths_a.ndim == 1 else np.trapz(lam_paths_a, t, axis=1)
    B_T = np.trapz(lam_paths_b, t, axis=1)
    M_T = np.trapz(np.maximum(lam_paths_a, lam_paths_b), t, axis=1)
    psi = np.exp(-A_T) + np.exp(-B_T) - 2.0 * np.exp(-M_T)
    return float(np.mean(psi))
