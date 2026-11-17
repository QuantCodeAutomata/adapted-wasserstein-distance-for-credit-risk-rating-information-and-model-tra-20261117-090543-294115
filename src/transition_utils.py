"""Shared utilities for finite-state rating-chain bond prices (Sec. 3.3).

Implements the forward recursion for survival probabilities, the induced
price kernels, the backward dynamic-programming recursion of Theorem 3.16,
the synchronous (Knothe-Rosenblatt) coupling of Theorem 3.18, the
stochastic-monotonicity test, and the Jafry-Schuermann mobility index used
in exp_2 and exp_8.
"""
from __future__ import annotations

import numpy as np

try:
    import ot  # POT: Python Optimal Transport, used for the exact transport subproblems
    _HAVE_POT = True
except ImportError:  # pragma: no cover - POT is listed as available in this image
    _HAVE_POT = False
from scipy.optimize import linprog


def survival_probabilities(P: np.ndarray, N: int) -> np.ndarray:
    """Backward recursion q_t(i) = P[R_N != 0 | R_t = i], t = 0..N.

    ``P`` is the one-period transition matrix (state 0 = default, absorbing).
    Returns an array of shape (N+1, n_states) with q_N = 1_{i != 0}.
    """
    n = P.shape[0]
    q = np.zeros((N + 1, n))
    q[N] = (np.arange(n) != 0).astype(float)
    for t in range(N - 1, -1, -1):
        q[t] = P @ q[t + 1]
    return q


def price_kernels(P: np.ndarray, q: np.ndarray, N: int, initial_rating: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build the price-state spaces E_t and the transition kernels kappa_t
    between them, for t = 0, ..., N-2, plus the state space E_{N-1} used as
    the base case of the recursion (Sec. 3.3, Theorem 3.16).

    States are represented by rating index (0 = default, 1..n-1 = ratings);
    the associated price label is q[t, index]. E_0 is the single state
    ``initial_rating`` (the bond's rating today), so that X_0 = q[0,
    initial_rating] = p after calibration. For t = 1, ..., N-1, E_t is the
    full index set {0, ..., n-1} (generic migration matrices make every
    rating reachable after at least one period; unreachable states simply
    carry zero probability mass, which optimal transport handles natively).
    kappa_t(i, j) = P(i, j) for t = 0, ..., N-2 (index i ranges over E_t,
    j over E_{t+1}); kappa_0 uses the single row ``initial_rating``.

    Returns (state_spaces, kernels) where state_spaces[t] is the array of
    *prices* q[t, i] for i in E_t (t = 0, ..., N-1), and kernels[t] is the
    matrix of transition probabilities from E_t to E_{t+1} (t = 0, ..., N-2).
    """
    n = P.shape[0]
    state_spaces: list[np.ndarray] = [np.array([q[0, initial_rating]])]
    for t in range(1, N):
        state_spaces.append(q[t].copy())
    kernels: list[np.ndarray] = []
    kernels.append(P[initial_rating:initial_rating + 1, :])  # kappa_0: E_0={r} -> E_1 (n states)
    for t in range(1, N - 1):
        kernels.append(P.copy())  # kappa_t: E_t (n states) -> E_{t+1} (n states)
    return state_spaces, kernels


def optimal_transport_cost(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> float:
    """Exact optimal transport value for marginals a, b and cost matrix ``cost``."""
    a = a / a.sum()
    b = b / b.sum()
    if _HAVE_POT:
        plan = ot.emd(a, b, cost)
        return float(np.sum(plan * cost))
    n_a, n_b = cost.shape
    A_eq = []
    b_eq = []
    for i in range(n_a):
        row = np.zeros(n_a * n_b)
        row[i * n_b:(i + 1) * n_b] = 1.0
        A_eq.append(row)
        b_eq.append(a[i])
    for j in range(n_b):
        row = np.zeros(n_a * n_b)
        row[j::n_b] = 1.0
        A_eq.append(row)
        b_eq.append(b[j])
    res = linprog(cost.flatten(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"LP transport failed: {res.message}")
    return float(res.fun)


def backward_dp_adapted_distance(
    state_spaces_p: list[np.ndarray], kernels_p: list[np.ndarray],
    state_spaces_q: list[np.ndarray], kernels_q: list[np.ndarray],
    N: int,
) -> float:
    """Backward recursion of Theorem 3.16: V_{N-1}(x,y) = |x-y|,
    V_t(x,y) = min over couplings of kappa_t(x,.), lambda_t(y,.) of the
    expected continuation value. Returns AW_2^2 = V_0(p, p).
    """
    e_p_N1, e_q_N1 = state_spaces_p[N - 1], state_spaces_q[N - 1]
    V_next = np.abs(e_p_N1[:, None] - e_q_N1[None, :])  # V_{N-1}(x,y) = |x-y|
    for t in range(N - 2, -1, -1):
        e_p_t, e_q_t = state_spaces_p[t], state_spaces_q[t]
        kappa_t, lambda_t = kernels_p[t], kernels_q[t]
        V_t = np.zeros((len(e_p_t), len(e_q_t)))
        for xi in range(len(e_p_t)):
            for yi in range(len(e_q_t)):
                V_t[xi, yi] = optimal_transport_cost(kappa_t[xi], lambda_t[yi], V_next)
        V_next = V_t
    assert V_next.shape == (1, 1), "state space at t=0 must be a single point {p}"
    return float(V_next[0, 0])


def synchronous_coupling_default_mismatch(
    P: np.ndarray, Qmat: np.ndarray, N: int, n_paths: int, rng: np.random.Generator,
    initial_rating: int = 0,
) -> float:
    """Monte Carlo estimate of pi^s[X_N != Y_N] under the synchronous
    (Knothe-Rosenblatt) coupling: both chains are driven by the same
    uniform innovations via their quantile functions (Eq. 3.11), started
    from the common initial rating ``initial_rating`` at t=0.
    """
    n = P.shape[0]
    x = np.full(n_paths, initial_rating, dtype=int)
    y = np.full(n_paths, initial_rating, dtype=int)
    cdf_p = np.cumsum(P, axis=1)
    cdf_q = np.cumsum(Qmat, axis=1)
    for _ in range(N):
        u = rng.uniform(size=n_paths)
        new_x = np.zeros(n_paths, dtype=int)
        new_y = np.zeros(n_paths, dtype=int)
        for i in range(n):
            mask_x = x == i
            if mask_x.any():
                new_x[mask_x] = np.searchsorted(cdf_p[i], u[mask_x], side="right")
            mask_y = y == i
            if mask_y.any():
                new_y[mask_y] = np.searchsorted(cdf_q[i], u[mask_y], side="right")
        x, y = new_x, new_y
    default_x = (x == 0)
    default_y = (y == 0)
    return float(np.mean(default_x != default_y))


def stochastic_monotonicity_violation(P: np.ndarray) -> float:
    """max_{i<i',k} (F_{i'}(k) - F_i(k)), the largest violation of stochastic
    monotonicity of the rows of P ordered by increasing credit quality
    (page 46): a stochastically monotone chain has, for every pair of
    starting ratings i (worse) < i' (better) and every quality threshold k,
    F_i(k) >= F_i'(k), i.e. the worse-rated obligor is at least as likely to
    end up in a bucket of quality <= k.

    ``P`` is stored with raw state indices [Default, AAA, AA, ..., CCC/C],
    which is NOT the quality order (index 1 = AAA is the best, not the
    second-worst). The cumulative distribution F_i(k) = P[R_1 in quality
    bucket <= k | R_0 = i] is only meaningful once columns are reordered into
    true quality order [Default, CCC/C, B, ..., AAA] (worst to best); rows
    compared are the non-default ratings, also reordered worst (CCC/C) to
    best (AAA). Returns 0 if P is stochastically monotone.
    """
    n = P.shape[0]
    col_order = [0] + list(range(n - 1, 0, -1))  # [Default, CCC/C, ..., AAA]: worst to best column
    row_order = list(range(n - 1, 0, -1))  # [CCC/C, ..., AAA]: worst to best non-default row
    cdf = np.cumsum(P[np.ix_(row_order, col_order)], axis=1)  # cdf[a, k] = F_{row_order[a]}(k)
    m = len(row_order)
    max_violation = 0.0
    for a in range(m):  # a indexes worse ratings as a increases... row_order[0]=CCC/C is worst
        for b in range(a + 1, m):  # row_order[b] is strictly better quality than row_order[a]
            diff = cdf[b] - cdf[a]  # should be <= 0 for monotonicity; positive values are violations
            max_violation = max(max_violation, float(np.max(diff)))
    return max_violation


def jafry_schuermann_mobility(P: np.ndarray) -> float:
    """Average singular value of P - I (page 47)."""
    n = P.shape[0]
    svals = np.linalg.svd(P - np.eye(n), compute_uv=False)
    return float(np.mean(svals))
