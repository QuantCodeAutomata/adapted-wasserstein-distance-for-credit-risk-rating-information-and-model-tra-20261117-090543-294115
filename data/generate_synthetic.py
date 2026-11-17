"""Synthetic sample generation for the experiments' Monte Carlo components.

No market data is used anywhere in this replication: every experiment
(exp_1 through exp_10) is a numerical verification of closed-form theorems
or a Monte Carlo simulation of models fully specified by the paper (rating
chains, Cox hazards, CIR intensities). This module generates the random
draws those simulations need: posterior-law samples for the calibration
estimator of Theorem 3.10, symmetric two-point and Beta posterior laws for
Sections 3-4, and bounded centered shocks for the small-dispersion
asymptotics of Theorem 4.35.
"""
from __future__ import annotations

import numpy as np


def two_point_posterior(p: float, delta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from mu_delta = 0.5*delta_{p-delta} + 0.5*delta_{p+delta}."""
    assert 0 < delta < min(p, 1 - p)
    signs = rng.choice([-1.0, 1.0], size=n)
    return p + signs * delta


def beta_posterior(p: float, a: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from a Beta(a,b) rescaled to [0,1] with mean exactly p,
    where b = a*(1-p)/p so that a/(a+b) = p."""
    b = a * (1 - p) / p
    return rng.beta(a, b, size=n)


def independent_posterior_pair(p: float, a: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two independent samples Q, Q_tilde ~ mu (Beta posterior) for the
    timing-of-information comparison (Theorem 3.13)."""
    b = a * (1 - p) / p
    Q = rng.beta(a, b, size=n)
    Q_tilde = rng.beta(a, b, size=n)
    return Q, Q_tilde


def bernoulli_given_posterior(Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """D ~ Bernoulli(Q) elementwise, the terminal default indicator given
    the revealed posterior survival probability."""
    u = rng.uniform(size=Q.shape)
    return (u < Q).astype(float)
