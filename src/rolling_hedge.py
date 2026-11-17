"""Rolling bootstrap hedge along a term structure and hedging with a window
of liquid bonds (Sec. 4.7-4.8), shared by exp_6, exp_9 and exp_10.
"""
from __future__ import annotations

import numpy as np


def cir_zero_coupon_price(tau: np.ndarray | float, lam: np.ndarray | float,
                           kappa: float, theta: float, sigma: float) -> np.ndarray:
    """Closed-form CIR affine survival price P(tau) = A(tau) exp(-B(tau) lam)
    (Cox-Ingersoll-Ross 1985), used as the exact conditional expectation
    E[exp(-integral_t^{t+tau} lambda_s ds) | lambda_t = lam] when the hazard
    follows a CIR process (Sec. 5.2-5.3): the Cox default time's survival
    probability is formally identical to a CIR zero-coupon bond price with
    the hazard playing the role of the short rate."""
    tau = np.asarray(tau, dtype=float)
    lam = np.asarray(lam, dtype=float)
    gamma = np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
    exp_g = np.exp(gamma * tau)
    denom = (gamma + kappa) * (exp_g - 1.0) + 2.0 * gamma
    B = 2.0 * (exp_g - 1.0) / denom
    A = (2.0 * gamma * np.exp((kappa + gamma) * tau / 2.0) / denom) ** (2.0 * kappa * theta / sigma ** 2)
    return A * np.exp(-B * lam)


def intra_year_conditional_price(h_before: np.ndarray, h_after: np.ndarray) -> np.ndarray:
    """E_{u~U(0,1)}[exp(-h_before*u - h_after*(1-u))], the exact pre-migration
    conditional price of a one-year survival claim when a single migration
    occurs at a uniform time u inside the year (Sec. 4.7, intra-year model).
    Closed form: (e^{-h_after} - e^{-h_before})/(h_before-h_after) when the
    two hazards differ, else e^{-h_before} (removable singularity)."""
    h_before = np.asarray(h_before, dtype=float)
    h_after = np.asarray(h_after, dtype=float)
    diff = h_before - h_after
    same = np.abs(diff) < 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (np.exp(-h_after) - np.exp(-h_before)) / np.where(same, 1.0, diff)
    out = np.where(same, np.exp(-h_before), ratio)
    return out


def price_curve_from_origin(P: np.ndarray, start_rating: int, K: int) -> np.ndarray:
    """P_0(t_k) = P[not in default after k one-year steps | start_rating]
    (k=0,...,K), the quoted zero-coupon survival curve observed at time 0
    (Eq. 4.44), obtained by forward matrix powers of the one-period
    transition matrix P (state 0 = Default): P_0(t_k) = sum_{j!=0} (P^k)[r,j].
    """
    n = P.shape[0]
    curve = np.zeros(K + 1)
    Pk = np.eye(n)
    curve[0] = 1.0
    for k in range(1, K + 1):
        Pk = Pk @ P
        curve[k] = 1.0 - Pk[start_rating, 0]
    return curve


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


def review_model_survival_curve(Q: np.ndarray, hazards: np.ndarray, start_rating: int, K: int) -> np.ndarray:
    """P_0(t_k) under the review model's own law (Sec. 4.7): the bucket
    (t_{k-1},t_k] hazard is that of the rating revealed at t_k, so survival
    to t_k is E[exp(-sum_{j<=k} h(R_j))] with R_j the Q-chain from
    start_rating. Computed by the matrix recursion u_k = u_{k-1} M,
    M(i,j) = Q(i,j) exp(-h(j)), S_k = sum_j u_k(j)."""
    n = Q.shape[0]
    M = Q * np.exp(-hazards)[None, :]
    u = np.zeros(n)
    u[start_rating] = 1.0
    curve = np.zeros(K + 1)
    curve[0] = 1.0
    for k in range(1, K + 1):
        u = u @ M
        curve[k] = u.sum()
    return curve


def intra_year_model_survival_curve(Q: np.ndarray, hazards: np.ndarray, start_rating: int, K: int) -> np.ndarray:
    """P_0(t_k) under the intra-year migration model's own law: the one-year
    survival factor between ratings i (pre-migration) and j (post-migration,
    migrating at a uniform time inside the year) is the closed form
    ``intra_year_conditional_price``; M(i,j) = Q(i,j) * that factor."""
    n = Q.shape[0]
    factor = np.array([[intra_year_conditional_price(hazards[i], hazards[j]) for j in range(n)] for i in range(n)])
    M = Q * factor
    u = np.zeros(n)
    u[start_rating] = 1.0
    curve = np.zeros(K + 1)
    curve[0] = 1.0
    for k in range(1, K + 1):
        u = u @ M
        curve[k] = u.sum()
    return curve


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


def rolling_hedge_review(hazards: np.ndarray, Q: np.ndarray, start_rating: int, pins: np.ndarray,
                          betas: np.ndarray, N: np.ndarray, n_paths: int, rng: np.random.Generator) -> dict:
    """Rolling bootstrap hedge under the review model (Sec. 4.7, Eq. 4.45-4.48):
    the next year's rating is revealed exactly at the left pin, so the
    post-review hazard is constant over the whole bucket and the realised
    residual, the Fréchet term E[phi] and the adapted bound E[Psi] coincide
    exactly bucket by bucket (Theorem 4.26, equality case)."""
    K = len(pins) - 1
    ratings = simulate_review_ratings(Q, start_rating, K, n_paths, rng)
    default_time = simulate_default_times_given_ratings(ratings, hazards, pins, rng, intra_year=False)
    return _bucket_aggregate(
        pins=pins, betas=betas, N=N, default_time=default_time,
        a_post=lambda k: hazards[ratings[:, k + 1]],
        cond_price=lambda k: np.exp(-hazards[ratings[:, k + 1]]),
        psi=lambda k, beta_k: np.exp(-hazards[ratings[:, k + 1]]) + np.exp(-beta_k)
        - 2.0 * np.exp(-np.maximum(hazards[ratings[:, k + 1]], beta_k)),
    )


def rolling_hedge_intra_year(hazards: np.ndarray, Q: np.ndarray, start_rating: int, pins: np.ndarray,
                              betas: np.ndarray, N: np.ndarray, n_paths: int, rng: np.random.Generator) -> dict:
    """Rolling bootstrap hedge under the intra-year migration model: the
    rating migrates once at a uniform time inside each year and the hazard
    switches immediately (Sec. 4.7). The pre-migration conditional price is
    the exact closed form of ``intra_year_conditional_price``, and the
    pathwise Psi is computed from the true piecewise-constant hazard path
    crossing the flat bootstrap rate within the bucket (Proposition 4.6)."""
    K = len(pins) - 1
    ratings = simulate_review_ratings(Q, start_rating, K, n_paths, rng)
    mig_times = rng.uniform(size=(n_paths, K))  # fraction of the year, in (0,1)
    default_time = simulate_default_times_given_ratings(ratings, hazards, pins, rng, intra_year=True)

    def a_post(k):
        h_before, h_after = hazards[ratings[:, k]], hazards[ratings[:, k + 1]]
        u = mig_times[:, k]
        return h_before * u + h_after * (1.0 - u)

    def cond_price(k):
        n = Q.shape[0]
        i = ratings[:, k]
        price = np.zeros(n_paths)
        for ii in range(n):
            mask = i == ii
            if not mask.any():
                continue
            h_i = hazards[ii]
            contrib = np.zeros(mask.sum())
            for jj in range(n):
                w = Q[ii, jj]
                if w <= 0:
                    continue
                contrib += w * intra_year_conditional_price(np.full(mask.sum(), h_i), np.full(mask.sum(), hazards[jj]))
            price[mask] = contrib
        return price

    def psi(k, beta_k):
        h_before, h_after = hazards[ratings[:, k]], hazards[ratings[:, k + 1]]
        u = mig_times[:, k]
        a = h_before * u + h_after * (1.0 - u)
        m = np.maximum(h_before, beta_k) * u + np.maximum(h_after, beta_k) * (1.0 - u)
        return np.exp(-a) + np.exp(-beta_k) - 2.0 * np.exp(-m)

    return _bucket_aggregate(pins=pins, betas=betas, N=N, default_time=default_time,
                              a_post=a_post, cond_price=cond_price, psi=psi)


def rolling_hedge_cir(kappa: float, theta: float, lambda0: float, sigma: float, pins: np.ndarray,
                       betas: np.ndarray, N: np.ndarray, n_paths: int, dt: float, rng: np.random.Generator) -> dict:
    """Rolling bootstrap hedge when the hazard follows a CIR process
    (Sec. 5.2-5.3). The exact affine formula gives the conditional price at
    each pin; the pathwise Psi integrates the pointwise maximum of the
    simulated path and the flat bootstrap rate over the fine time grid."""
    from src.cox_hazard import simulate_cir

    T = pins[-1]
    t_grid, lam = simulate_cir(kappa, theta, lambda0, sigma, T, dt, n_paths, rng)
    steps_per_year = int(round(1.0 / dt))
    K = len(pins) - 1
    cum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(0.5 * (lam[:, :-1] + lam[:, 1:]) * dt, axis=1)], axis=1)
    pin_idx = [int(round(t / dt)) for t in pins]
    Lambda_at_pins = cum[:, pin_idx]
    E = rng.exponential(size=n_paths)
    default_time = np.full(n_paths, np.inf)
    crossed = Lambda_at_pins >= E[:, None]
    first_cross = np.argmax(crossed, axis=1)
    has_crossed = crossed.any(axis=1)
    default_time[has_crossed] = pins[first_cross[has_crossed]]

    def lam_bucket(k):
        return lam[:, k * steps_per_year: (k + 1) * steps_per_year + 1]

    def a_post(k):
        seg = lam_bucket(k)
        return np.trapz(seg, dx=dt, axis=1)

    def cond_price(k):
        lam_tk = lam[:, k * steps_per_year]
        return cir_zero_coupon_price(pins[k + 1] - pins[k], lam_tk, kappa, theta, sigma)

    def psi(k, beta_k):
        seg = lam_bucket(k)
        a = np.trapz(seg, dx=dt, axis=1)
        m = np.trapz(np.maximum(seg, beta_k), dx=dt, axis=1)
        return np.exp(-a) + np.exp(-beta_k) - 2.0 * np.exp(-m)

    return _bucket_aggregate(pins=pins, betas=betas, N=N, default_time=default_time,
                              a_post=a_post, cond_price=cond_price, psi=psi)


def _bucket_aggregate(pins: np.ndarray, betas: np.ndarray, N: np.ndarray, default_time: np.ndarray,
                       a_post, cond_price, psi) -> dict:
    """Shared bucket-by-bucket aggregation of the rolling hedge (Eq. 4.45-4.48):
    for each bucket, average the realised residual, the Fréchet term E[phi]
    and the adapted bound E[Psi] over paths alive at the left pin, and sum
    across buckets. ``a_post``, ``cond_price`` and ``psi`` are callables of
    the bucket index (and, for ``psi``, the bucket's bootstrap rate)
    returning per-path arrays."""
    n_paths = len(default_time)
    K = len(pins) - 1
    realized_total = np.zeros(n_paths)
    bucket_rows = []
    for k in range(K):
        alive = default_time > pins[k]
        beta_k = betas[k]
        price = cond_price(k)
        eps_k = N[k] - N[k + 1] * price
        phi_k = np.abs(np.exp(-a_post(k)) - np.exp(-beta_k))
        psi_k = psi(k, beta_k)
        realized_total += np.where(alive, eps_k, 0.0)
        bucket_rows.append({
            "k": k, "n_alive": int(alive.sum()),
            "realized_abs": float(np.mean(np.where(alive, np.abs(eps_k), 0.0))),
            "frechet": float(N[k + 1] * np.mean(np.where(alive, phi_k, 0.0))),
            "adapted": float(N[k + 1] * np.mean(np.where(alive, psi_k, 0.0))),
        })
    total_realized_abs = float(np.mean(np.abs(realized_total)))
    sum_bucket_realized = float(sum(r["realized_abs"] for r in bucket_rows))
    sum_frechet = float(sum(r["frechet"] for r in bucket_rows))
    sum_adapted = float(sum(r["adapted"] for r in bucket_rows))
    return {
        "buckets": bucket_rows,
        "sum_bucket_realized_abs": sum_bucket_realized,
        "sum_frechet": sum_frechet,
        "sum_adapted": sum_adapted,
        "total_cash_abs_mean": total_realized_abs,
        "total_cash": realized_total,
        "mean_signed_residual": float(np.mean(realized_total)),
    }


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
    """Least-squares weights theta on the m window bonds maturing at
    t_{k+1},...,t_{k+m} (price vectors K_hat^1.1,...,K_hat^m.1) matching the
    liability's price vector K_hat^{K_total-k}.1 at the current pin t_k
    (Sec. 4.8, Eq. 4.51): theta solves min_theta ||basis @ theta - target||
    over the rating state space, so the portfolio value-matches the T-bond
    for every current rating when m spans the Krylov subspace."""
    n = K_hat.shape[0]
    vec = np.ones(n)
    basis = []
    for _ in range(m):
        vec = K_hat @ vec
        basis.append(vec.copy())
    basis = np.column_stack(basis)
    target_power = max(K_total - k, 0)
    target_vec = np.ones(n)
    for _ in range(target_power):
        target_vec = K_hat @ target_vec
    theta, *_ = np.linalg.lstsq(basis, target_vec, rcond=None)
    return theta
