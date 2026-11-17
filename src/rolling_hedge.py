"""Rolling bootstrap hedge along a term structure and hedging with a window
of liquid bonds (Sec. 4.7-4.8), shared by exp_6, exp_9 and exp_10.
"""
from __future__ import annotations

import numpy as np


def hazard_from_default_prob(default_prob: np.ndarray) -> np.ndarray:
    """h_i = -log(1 - P(i,0)), the one-year hazard implied by rating i's
    one-year default probability (Sec. 5.2)."""
    return -np.log(1.0 - default_prob)


def renormalized_non_default_transitions(P: np.ndarray) -> np.ndarray:
    """Q(i,j) = P(i,j) / (1 - P(i,0)) for j != 0: the migration matrix among
    non-default ratings conditional on surviving one year (Sec. 5.2)."""
    n = P.shape[0]
    Q = np.zeros((n - 1, n - 1))  # drop the default row/column; index 0 = AAA ... n-2 = CCC/C
    for i in range(1, n):
        denom = 1.0 - P[i, 0]
        for j in range(1, n):
            Q[i - 1, j - 1] = P[i, j] / denom
    return Q


def bootstrap_rates(survival_curve: np.ndarray, pins: np.ndarray) -> np.ndarray:
    """beta_k from exp(-beta_k*(t_k-t_{k-1})) = P_0(t_k)/P_0(t_{k-1}) (Eq. 4.44)."""
    betas = np.zeros(len(pins) - 1)
    for k in range(1, len(pins)):
        ratio = survival_curve[k] / survival_curve[k - 1]
        betas[k - 1] = -np.log(ratio) / (pins[k] - pins[k - 1])
    return betas


def bootstrap_cumulative_hazard(betas: np.ndarray, pins: np.ndarray) -> np.ndarray:
    """B_{t_k} = sum_{l<=k} beta_l*(t_l - t_{l-1}), B_{t_0}=0."""
    B = np.zeros(len(pins))
    for k in range(1, len(pins)):
        B[k] = B[k - 1] + betas[k - 1] * (pins[k] - pins[k - 1])
    return B


def rolling_positions(B: np.ndarray) -> np.ndarray:
    """N_k = exp(-(B_T - B_{t_k})), k=0,...,K (position held on [t_{k-1},t_k])."""
    return np.exp(-(B[-1] - B))


def simulate_review_ratings(Q: np.ndarray, start_rating: int, K: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate the non-default rating path at each annual pin under the
    review model (migration revealed exactly at the pins). Ratings are
    indices 0..n-2 among non-default categories (0=AAA,...). Once a path
    defaults it is flagged with -1 and stays there."""
    n = Q.shape[0]
    cdf = np.cumsum(Q, axis=1)
    ratings = np.full((n_paths, K + 1), -1, dtype=int)
    ratings[:, 0] = start_rating
    for k in range(1, K + 1):
        alive = ratings[:, k - 1] != -1
        u = rng.uniform(size=n_paths)
        new_rating = np.full(n_paths, -1)
        for i in range(n):
            mask = alive & (ratings[:, k - 1] == i)
            if mask.any():
                new_rating[mask] = np.searchsorted(cdf[i], u[mask], side="right")
        ratings[:, k] = new_rating
    return ratings


def simulate_default_times_given_ratings(ratings_path: np.ndarray, hazards: np.ndarray, pins: np.ndarray,
                                          rng: np.random.Generator, intra_year: bool = False) -> np.ndarray:
    """Cox default times given the (review or intra-year) rating path.

    ``ratings_path`` has shape (n_paths, K+1): ratings_path[:,k] is the
    rating that applies on the interval (t_{k-1}, t_k] under the review
    model. When ``intra_year`` is True, a second migration (to
    ratings_path[:, k]) occurs at a uniform time inside (t_{k-1}, t_k],
    with ratings_path[:, k-1] applying before it.
    """
    n_paths, K1 = ratings_path.shape
    K = K1 - 1
    E = rng.exponential(size=n_paths)
    cum_hazard = np.zeros(n_paths)
    default_time = np.full(n_paths, np.inf)
    alive = np.ones(n_paths, dtype=bool)
    if intra_year:
        mig_times = rng.uniform(size=(n_paths, K)) * np.diff(pins)[None, :] + pins[:-1][None, :]
    for k in range(K):
        if not alive.any():
            break
        t0, t1 = pins[k], pins[k + 1]
        if intra_year:
            h_before = hazards[ratings_path[:, k]]
            h_after = hazards[ratings_path[:, k + 1]]
            mig_t = mig_times[:, k]
            seg1 = h_before * (mig_t - t0)
            seg2 = h_after * (t1 - mig_t)
            add = np.where(alive, seg1, 0.0)
            newly_default = alive & (cum_hazard + add >= E)
            # resolve default time within the first segment if crossed there
            frac = np.zeros(n_paths)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(h_before > 0, (E - cum_hazard) / h_before, np.inf)
            dt_seg1 = t0 + frac
            cross_in_seg1 = newly_default & (dt_seg1 <= mig_t)
            default_time[cross_in_seg1] = dt_seg1[cross_in_seg1]
            cum_hazard = np.where(alive, cum_hazard + seg1, cum_hazard)
            alive = alive & ~newly_default
            add2 = np.where(alive, seg2, 0.0)
            newly_default2 = alive & (cum_hazard + add2 >= E)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac2 = np.where(h_after > 0, (E - cum_hazard) / h_after, np.inf)
            dt_seg2 = mig_t + frac2
            default_time[newly_default2] = dt_seg2[newly_default2]
            cum_hazard = np.where(alive, cum_hazard + seg2, cum_hazard)
            alive = alive & ~newly_default2
        else:
            h = hazards[ratings_path[:, k + 1]]
            add = np.where(alive, h * (t1 - t0), 0.0)
            newly_default = alive & (cum_hazard + add >= E)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(h > 0, (E - cum_hazard) / h, np.inf)
            dt = t0 + frac
            default_time[newly_default] = dt[newly_default]
            cum_hazard = np.where(alive, cum_hazard + add, cum_hazard)
            alive = alive & ~newly_default
    return default_time


def survival_weighted_kernel(Q: np.ndarray, one_year_hazard_by_rating: np.ndarray) -> np.ndarray:
    """K_hat(i,i') = exp(-h_i) * Q(i,i') (Sec. 4.8, Sec. 5.3: review model,
    one-year survival-weighted transition kernel among non-default ratings)."""
    return np.exp(-one_year_hazard_by_rating)[:, None] * Q


def krylov_matrix(K_hat: np.ndarray, n_powers: int) -> np.ndarray:
    """Columns [1, K_hat.1, K_hat^2.1, ..., K_hat^{n_powers-1}.1]."""
    n = K_hat.shape[0]
    vec = np.ones(n)
    cols = [vec.copy()]
    for _ in range(n_powers - 1):
        vec = K_hat @ vec
        cols.append(vec.copy())
    return np.column_stack(cols)


def market_desk_weights(K_hat: np.ndarray, k: int, K_total: int, m: int) -> np.ndarray:
    """Least-squares weights theta on the Krylov vectors
    [K_hat^0.1, ..., K_hat^{m-1}.1] matching the liability price vector
    K_hat^{K_total-k-1}.1 at t_{k+1} (Sec. 4.8, Eq. 4.51)."""
    n = K_hat.shape[0]
    vec = np.ones(n)
    basis = []
    for _ in range(m):
        basis.append(vec.copy())
        vec = K_hat @ vec
    basis = np.column_stack(basis)
    target_power = max(K_total - k - 1, 0)
    target_vec = np.ones(n)
    for _ in range(target_power):
        target_vec = K_hat @ target_vec
    theta, *_ = np.linalg.lstsq(basis, target_vec, rcond=None)
    return theta
