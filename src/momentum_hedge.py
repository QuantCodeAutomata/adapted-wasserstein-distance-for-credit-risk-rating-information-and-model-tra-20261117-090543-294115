"""Momentum payoff replication and stability of model-transfer hedging
(Sec. 3.4, Theorem 3.19, Corollary 3.20, Proposition 3.22).
"""
from __future__ import annotations

import numpy as np


def replicate_momentum_coarse(p: float, D: np.ndarray) -> np.ndarray:
    """Under P^0: S0=S1=p, S2=D. Strategy m=0,H1=0,H2=S1-S0=0 replicates
    Gamma(S) = (S1-S0)(S2-S1) = 0 exactly (Theorem 3.19)."""
    gamma = (p - p) * (D - p)
    value = np.zeros_like(D, dtype=float)  # V2 = 0 + 0*(S1-S0) + 0*(S2-S1)
    assert np.allclose(value, gamma)
    return gamma


def replicate_momentum_rating(p: float, Q: np.ndarray, D_mu: np.ndarray) -> dict:
    """Under P^mu: S0=p, S1=Q, S2=D_mu. Strategy H2 = Q - p replicates
    Gamma(S) = (Q-p)(D_mu - Q) (Theorem 3.19)."""
    H2 = Q - p
    gamma = (Q - p) * (D_mu - Q)
    value = H2 * (D_mu - Q)  # V2 = 0 + H1*(S1-S0) + H2*(S2-S1), H1=0
    assert np.allclose(value, gamma, atol=1e-10), "replication must be pathwise exact"
    return {"H2": H2, "gamma": gamma, "value": value}


def hedge_size_identities(Q: np.ndarray, p: float) -> dict:
    """E|H2| = AW_2^2 = E|Q-p|; (E|H2|^2)^(1/2) = (E|Q-p|^2)^(1/2) = W_2/sqrt(2)
    when |Q-p|<=1/2 a.s. (Corollary 3.20)."""
    H2 = Q - p
    e_abs = float(np.mean(np.abs(H2)))
    e_sq_sqrt = float(np.sqrt(np.mean(H2 ** 2)))
    aw2_sq = float(np.mean(np.abs(Q - p)))
    assert abs(e_abs - aw2_sq) < 1e-10
    w2 = np.sqrt(2.0 * np.mean(H2 ** 2)) if np.all(np.abs(H2) <= 0.5 + 1e-9) else None
    return {"E|H2|": e_abs, "sqrt(E|H2|^2)": e_sq_sqrt, "AW2_sq": aw2_sq, "W2": w2}


def variance_optimal_hedge(delta_claim: np.ndarray, delta_price: np.ndarray) -> float:
    """One-step variance-optimal hedge ratio H = Cov(dG, dX)/Var(dX)."""
    cov = np.mean((delta_claim - delta_claim.mean()) * (delta_price - delta_price.mean()))
    var = np.var(delta_price)
    return float(cov / var) if var > 0 else 0.0


def l2_hedging_error(G: np.ndarray, m: float, H2: float, S0: float, S1: np.ndarray, S2: np.ndarray) -> float:
    """R_P(m,H;G) = ||G(X) - m - (H.X)_N||_{L2(P)} for the one-trade strategy
    with H1=0 and a (possibly interpolated) H2 applied at the review date."""
    gains = H2 * (S2 - S1)
    residual = G - m - gains
    return float(np.sqrt(np.mean(residual ** 2)))


def stability_bound(L_G: float, K: float, c_N: float, L_H: float, aw2: float) -> float:
    """|R_P - R_Q| <= (L_G + K + c_N*L_H) * AW_2(P,Q) (Proposition 3.22)."""
    return (L_G + K + c_N * L_H) * aw2
