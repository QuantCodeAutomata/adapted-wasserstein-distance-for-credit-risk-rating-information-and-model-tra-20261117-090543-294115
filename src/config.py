"""Experiment parameters for the adapted-Wasserstein-distance replication.

All numeric constants used by exp/*.py live here as module-level UPPER_CASE
variables, grouped by experiment, each with a comment on its meaning, unit
and source (paper section/equation, or "not specified in the paper").
No classes or dicts: every parameter is a plain variable so it can be
imported individually.
"""

import numpy as np

RANDOM_SEED = 12345  # global seed for reproducibility; not specified in the paper

# ---------------------------------------------------------------------------
# exp_1: single rating review, timing of information, calibration-preserving
# estimation (Sec. 3.1-3.2, Theorems 3.6, 3.8, 3.10, 3.13, 3.15, Cor. 3.9)
# ---------------------------------------------------------------------------
EXP1_P = 0.90  # initial survival probability p, dimensionless; illustrative, not specified in the paper
EXP1_DELTAS = np.array([0.099, 0.05, 0.02, 0.005, 0.0015, 0.0005])  # symmetric revision sizes delta for Q = p +/- delta, delta < min(p,1-p)=0.1 (Corollary 3.9)
EXP1_BETA_A = 5.0  # Beta-distribution shape parameter a for the alternative posterior law mu (rescaled to [0,1] with mean p); not specified in the paper
EXP1_N_LIST = np.array([10, 30, 100, 300, 1000, 3000, 10000])  # sample sizes n for the calibration-preserving estimator (Theorem 3.10)
EXP1_N_REPLICATIONS = 2000  # Monte Carlo replications per sample size for the O(n^{-1/2}) rate check; count not specified in the paper
EXP1_TIMING_P = 0.85  # initial price p used for the early/late timing comparison (Theorem 3.13); illustrative

# ---------------------------------------------------------------------------
# exp_2: multi-period migration DP recursion and synchronous coupling
# (Theorem 3.16, 3.18)
# ---------------------------------------------------------------------------
EXP2_MATURITIES = np.array([2, 4, 6, 8, 10])  # bond maturities N in years, Sec. 5.1
EXP2_SYNC_PATHS = 200_000  # Monte Carlo paths for the synchronous-coupling estimate; count not specified in the paper
EXP2_INITIAL_RATINGS = ["A", "BBB", "BB", "B", "CCC/C"]  # initial ratings compared, Table 1 (page 46)

# ---------------------------------------------------------------------------
# exp_3: momentum payoff replication and stability bound (Theorem 3.19,
# Corollary 3.20, Proposition 3.22)
# ---------------------------------------------------------------------------
EXP3_P = 0.90  # initial survival probability, same convention as exp_1; illustrative
EXP3_MU_DELTA = 0.05  # symmetric two-point posterior law mu: Q = p +/- delta, not specified in the paper
EXP3_NU_DELTA = 0.08  # symmetric two-point posterior law nu for the second (recalibrated) model, not specified in the paper
EXP3_N_MC = 500_000  # Monte Carlo sample size for the L^2 hedging-error estimate; count not specified in the paper
# Stability test (Proposition 3.22): two finite three-point posterior laws mu, nu
# with mean exactly EXP3_P, used for the claim G=|Q-p| ("rating-date straddle",
# page 23) and its variance-optimal hedge transferred between the two models.
# Supports and probabilities are illustrative (not specified in the paper beyond
# the straddle example) and chosen only so that each law has mean EXP3_P exactly.
EXP3_STABILITY_MU_SUPPORT = np.array([0.70, 0.90, 0.95])
EXP3_STABILITY_MU_PROBS = np.array([0.10, 0.50, 0.40])
EXP3_STABILITY_NU_SUPPORT = np.array([0.60, 0.90, 0.93])
EXP3_STABILITY_NU_PROBS = np.array([0.05, 0.45, 0.50])

# ---------------------------------------------------------------------------
# exp_4: Cox bond prices with deterministic hazards (Theorem 4.3, 4.7, 4.9,
# Proposition 4.6, Theorem 4.11)
# ---------------------------------------------------------------------------
EXP4_T = 5.0  # bucket length T in years, Sec. 4 illustrations
EXP4_HAZARD_CAP_L = 0.25  # intensity cap L (per year) for the sharp range (Theorem 4.7); not specified in the paper, chosen above typical CCC hazards
EXP4_HA = 0.06  # integrated-hazard target h_a = integral_0^T a, per year average 0.06*... ; illustrative, not specified in the paper
EXP4_HB = 0.09  # integrated-hazard target h_b for the second system; illustrative
EXP4_COMMON_SHOCK_PATHS = 200_000  # Monte Carlo paths for the common-shock coupling check (Eq. 4.10-4.11)
EXP4_STABILITY_EPS = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])  # perturbation sizes epsilon for the L^1 stability sharpness check
EXP4_CIR_KAPPA = 0.3  # CIR mean-reversion speed, per year (Sec. 5.2 parameters, used for exp_4 random-intensity illustration)
EXP4_CIR_THETA = 0.03  # CIR long-run mean hazard level, per year
EXP4_CIR_LAMBDA0 = 0.02  # CIR initial hazard, per year
EXP4_CIR_SIGMA = 0.08  # CIR volatility parameter, mid value of {0.04,0.08,0.12} (Sec. 5.2-5.3)
EXP4_CIR_DT = 1.0 / 250  # Euler-Maruyama time step, per year (Sec. 5.2: "time step dt = 1/250")
EXP4_CIR_PATHS = 50_000  # Monte Carlo paths for the random-intensity default-layer reduction (Theorem 4.11); reduced from a larger count to keep the run time short, see results/RESULTS.md

# ---------------------------------------------------------------------------
# exp_5: single calibrated rating-review bucket (Theorem 4.16, 4.23,
# Proposition 4.21/4.37, Example 4.22)
# ---------------------------------------------------------------------------
EXP5_T1 = 0.0  # review date t1 (bucket start), years
EXP5_T = 5.0  # bucket maturity T, years (Example 4.22: five-year bucket)
EXP5_HAZARDS_L = np.array([0.02, 0.06, 0.20])  # flat post-review hazards L (per year) for the three ratings, Example 4.22
EXP5_PROBS = np.array([0.5, 0.3, 0.2])  # probabilities of the three post-review ratings, Example 4.22
EXP5_XI = 0.0  # pre-review cumulative hazard Xi, so exp(-Xi) = 1 (Example 4.22)

# ---------------------------------------------------------------------------
# exp_6: rolling bootstrap along a term structure + liquidity window
# (Section 4.7-4.8, Theorem 4.26, 4.27, Proposition 4.28)
# ---------------------------------------------------------------------------
EXP6_K_YEARS = 7  # bond maturity in years with annual pins (Sec. 5.2-5.3 use K=7)
EXP6_PATHS = 200_000  # Monte Carlo paths for the rolling bootstrap hedge (Sec. 5.2: "2*10^5 paths")
EXP6_PATHS_FULL = 200_000  # paper configuration, same as the executed one for this experiment
EXP6_SVD_THRESHOLD = 1e-3  # relative singular-value threshold used to read off the effective Krylov dimension d; not specified in the paper

# ---------------------------------------------------------------------------
# exp_7: robust pricing over adapted balls and small-dispersion asymptotics
# (Theorem 4.30, 4.35, Corollary 4.31, 4.36, Proposition 4.29, 4.32)
# ---------------------------------------------------------------------------
EXP7_P = 0.90  # initial survival probability for the robust-pricing illustration; illustrative, not specified in the paper
EXP7_ETAS = np.array([0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15])  # adapted-ball radii eta (bounded by 2p(1-p))
EXP7_STRIKES = np.array([0.70, 0.80, 0.85, 0.90, 0.95, 0.98])  # option strikes k used for the robust call premium
EXP7_DELTA = 2.0  # time-to-maturity Delta = T - t_r for the small-dispersion asymptotics, years; illustrative
EXP7_EPSILONS = np.geomspace(1e-1, 1e-4, 12)  # perturbation sizes epsilon for the small-dispersion log-log check (Corollary 4.36); range not specified in the paper
EXP7_ZETA_VALUES = np.array([-1.0, 1.0])  # symmetric bounded centered shock zeta = +/-1 with equal probability; choice not specified in the paper
EXP7_ZETA_PROBS = np.array([0.5, 0.5])
EXP7_N_MC_SMALL_DISP = 400_000  # Monte Carlo sample size per epsilon for the AW2/W2 asymptotics check

# ---------------------------------------------------------------------------
# exp_8: empirical comparison of S&P-style calibrated rating systems
# (Sec. 5.1, Table 1, Figure 1)
# ---------------------------------------------------------------------------
EXP8_MATURITIES = np.array([2, 4, 6, 8, 10])  # bond maturities N in years, Table 1
EXP8_INITIAL_RATINGS = ["BBB", "BB", "B"]  # ratings tabulated in Table 1 (page 46)
EXP8_SYNC_PATHS = 1_000_000  # Monte Carlo paths for the synchronous-coupling comparison (Sec. 5.1 uses 10^6 for distributional quantities elsewhere)
EXP8_RECESSION_YEARS = [1990, 1991, 2001, 2002, 2008, 2009, 2020]  # recession years defining the regime chain (page 46)
EXP8_P_EXPANSION_TO_RECESSION = 0.111  # P[E -> R], regime transition probability (page 46)
EXP8_P_RECESSION_TO_RECESSION = 0.429  # P[R -> R], regime transition probability (page 46)

# ---------------------------------------------------------------------------
# exp_9: Monte Carlo rolling-bootstrap study (review / intra-year / CIR)
# (Sec. 5.2, Figure 2)
# ---------------------------------------------------------------------------
EXP9_K_YEARS = 7  # bond maturity, annual pins (Sec. 5.2)
EXP9_PATHS_PAPER = 200_000  # paper's Monte Carlo sample size (Sec. 5.2: "2*10^5 paths")
EXP9_PATHS_RUN = 200_000  # executed sample size (equal to the paper's; the model is cheap to simulate)
EXP9_CIR_SIGMAS = np.array([0.04, 0.08, 0.12])  # CIR volatility levels tested (Sec. 5.2)
EXP9_CIR_KAPPA = 0.3  # CIR mean-reversion speed, per year
EXP9_CIR_THETA = 0.03  # CIR long-run mean hazard, per year
EXP9_CIR_LAMBDA0 = 0.02  # CIR initial hazard, per year
EXP9_CIR_DT = 1.0 / 250  # Euler-Maruyama step, per year (Sec. 5.2)
EXP9_CIR_PATHS_PAPER = 200_000  # paper's Monte Carlo sample size for the CIR case (Sec. 5.2)
EXP9_CIR_PATHS_RUN = 50_000  # reduced so the fine (dt=1/250) path array stays within the process memory limit; see results/RESULTS.md

# ---------------------------------------------------------------------------
# exp_10: hedging with a liquidity window of bonds (Sec. 4.8, Sec. 5.3)
# ---------------------------------------------------------------------------
EXP10_K_YEARS = 7  # bond maturity, annual pins
EXP10_PATHS_PAPER = 1_000_000  # paper's Monte Carlo sample size for distributional quantities (Sec. 5.3: "10^6 paths")
EXP10_PATHS_RUN = 200_000  # executed sample size, reduced to keep the run time short; see results/RESULTS.md
EXP10_WINDOW_RANGE = np.arange(1, EXP10_K_YEARS)  # liquidity windows m = 1,...,K-1
EXP10_SVD_THRESHOLD = 1e-3  # relative threshold to read off the Krylov dimension d; not specified in the paper
