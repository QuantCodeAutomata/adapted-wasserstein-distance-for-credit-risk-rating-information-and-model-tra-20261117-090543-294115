"""Single rating review, timing of information and calibration-preserving
estimation (Sec. 3.1-3.2).

Custom implementation of the closed-form formulas; no optimal-transport
library is needed because Theorems 3.6, 3.8 and 3.13 reduce the adapted and
ordinary Wasserstein distances to explicit expectations. General-purpose
portfolio and optimal-transport libraries (pyportfolioopt, skfolio, POT) do
not implement adapted or causal Wasserstein distances, so this module follows
the paper's closed-form formulas directly.
"""
from __future__ import annotations

import numpy as np


def aw2_squared_single_review(Q: np.ndarray, p: float) -> float:
    """AW_2^2(P^0, P^mu) = E|Q - p| (Theorem 3.6, Eq. 3.3)."""
    return float(np.mean(np.abs(Q - p)))


def w2_squared_bounds_single_review(Q: np.ndarray, p: float) -> tuple[float, float]:
    """(lower, upper) = (E|Q-p|^2, 2 E|Q-p|^2) for W_2^2(P^0, P^mu) (Theorem 3.8)."""
    e2 = float(np.mean((Q - p) ** 2))
    return e2, 2.0 * e2


def w2_squared_exact(Q: np.ndarray, p: float) -> float:
    """Exact W_2^2 when |Q-p| <= 1/2 almost surely, where the upper bound of
    Theorem 3.8 is attained: W_2^2 = 2 E|Q-p|^2."""
    assert np.all(np.abs(Q - p) <= 0.5 + 1e-12), "Theorem 3.8 equality requires |Q-p|<=1/2 a.s."
    return 2.0 * float(np.mean((Q - p) ** 2))


def symmetric_two_point_distances(p: float, delta: float) -> dict:
    """Corollary 3.9: Q = p +/- delta with probability 1/2 each.
    AW_2 = sqrt(delta), W_2 = sqrt(2)*delta, ratio = (2*delta)^{-1/2}.
    """
    assert 0 < delta < min(p, 1 - p), "delta must keep Q in (0,1)"
    aw2_sq = delta  # E|Q-p| = delta
    aw2 = np.sqrt(aw2_sq)
    w2_sq = 2.0 * delta ** 2  # 2*E|Q-p|^2, |Q-p|=delta<=1/2 assumed by the caller
    w2 = np.sqrt(w2_sq)
    ratio = aw2 / w2
    predicted_ratio = (2.0 * delta) ** (-0.5)
    return {
        "delta": delta, "aw2": aw2, "w2": w2, "ratio": ratio,
        "predicted_ratio": predicted_ratio, "aw2_sq": aw2_sq, "w2_sq": w2_sq,
    }


def timing_distance(Q: np.ndarray, Q_tilde: np.ndarray) -> float:
    """AW_2^2(P^E, P^L) = E|Q - Q_tilde| for Q, Q_tilde iid ~ mu (Theorem 3.13)."""
    return float(np.mean(np.abs(Q - Q_tilde)))


def no_information_benchmark_distance(Y_last: np.ndarray, p: float) -> float:
    """AW_2^2(P^{0,N}, Q) = E_Q|Y_{N-1} - p| (Theorem 3.15)."""
    return float(np.mean(np.abs(Y_last - p)))


def calibration_preserving_map(q: np.ndarray, m: float, p: float) -> np.ndarray:
    """Piecewise-linear calibration-preserving map T_{m,p} (Theorem 3.10):
    T(q) = (p/m) q if m >= p, else 1 - (1-p)/(1-m) * (1-q).
    """
    q = np.asarray(q, dtype=float)
    if m >= p:
        if m == 0:
            return np.full_like(q, p)
        return (p / m) * q
    if m == 1:
        return np.full_like(q, p)
    return 1.0 - (1.0 - p) / (1.0 - m) * (1.0 - q)


def estimate_aw2_squared(samples: np.ndarray, p: float) -> tuple[float, float]:
    """Calibration-preserving estimator of AW_2^2 (Theorem 3.10).

    Returns (true-law-style empirical mean before calibration, calibrated
    estimate). ``samples`` are n iid draws Q_1,...,Q_n ~ mu.
    """
    m = float(np.mean(samples))
    calibrated = calibration_preserving_map(samples, m, p)
    assert abs(float(np.mean(calibrated)) - p) < 1e-9, "calibrated law must have mean exactly p"
    estimate = float(np.mean(np.abs(calibrated - p)))
    return m, estimate
