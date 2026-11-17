# Repository guide for coding agents

This repository reproduces ten numerical experiments from *Adapted
Wasserstein Distance for Credit Risk: Rating Information and Model-Transfer
Hedging*. It verifies the paper's closed-form formulas (Theorems 3.6, 3.8,
3.13, 3.15, 3.16, 3.18, 4.3, 4.7, 4.9, 4.11, 4.16, 4.23, 4.26–4.28, 4.30,
4.32, 4.35) with exact computations where the paper gives a closed form, and
with Monte Carlo simulation where the paper itself specifies a simulated
market model (rating-review, intra-year migration, CIR intensity).

## Layout

```
src/        flat library modules: single-review formulas (src/single_review.py),
            dynamic-programming adapted distance and synchronous coupling
            (src/dp_coupling.py), momentum-payoff hedging (src/momentum_hedge.py),
            Cox-hazard mismatch (src/cox_hazard.py), single-bucket bootstrap
            hedging (src/bootstrap_hedge.py), multi-bucket rolling and window
            hedges (src/rolling_hedge.py), robust pricing and small-dispersion
            asymptotics (src/robust_pricing.py), rating-transition utilities
            and calibration transforms (src/transition_utils.py,
            src/rating_data.py), and src/config.py with every experiment
            parameter as a module-level variable.
exp/        one script per experiment, exp1.py .. exp10.py, run as
            `python -m exp.expN`. Each script only calls functions exposed by
            src/ and writes its tables/figures to results/.
data/       data/generate_synthetic.py generates every simulated input used by
            the experiments: Monte Carlo samples for the single-review
            estimator (exp_1), the stylized rating-transition matrix consumed
            by src/rating_data.py (exp_2, exp_6, exp_8, exp_9, exp_10), and
            CIR/hazard-perturbation draws (exp_4, exp_6, exp_7, exp_9, exp_10).
            No market-data loader is included: the paper's only external
            input, the S&P Global Ratings transition study, is a proprietary
            report not served by any market-data API available to this
            replication (see results/RESULTS.md, "Deviations from the
            paper").
tests/      pytest unit tests for every src/ module, using small fixtures or
            synthetic inputs (no tests depend on a live API call).
results/    every table, figure and log the experiments produce, plus
            results/RESULTS.md with the comparison against the paper's
            reported values.
```

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -m "not market_data"
python -m exp.exp1
python -m exp.exp2
...
python -m exp.exp10
bash reproduce.sh   # runs the tests and every experiment above, in order
```

`exp6`, `exp9` and `exp10` accept a `--full` flag that switches from the
reduced Monte Carlo sample sizes used by default to the paper's reported
sample sizes (`EXP*_PATHS_PAPER` in `src/config.py`); see
`results/RESULTS.md` for the executed vs. paper configuration of every
reduced run.

## Conventions

* Every experiment parameter lives in `src/config.py` as an UPPER_CASE
  module-level variable with a comment naming its meaning, unit and the
  paper section or equation it comes from (or stating that the paper does
  not specify it). No classes or parameter dictionaries.
* All data preparation lives in `data/`. `data/generate_synthetic.py` is the
  only place that generates simulated inputs; `src/` and `exp/` never
  generate or load data themselves, they only call `data/generate_synthetic.py`
  or deterministic closed-form routines in `src/`.
* No market-data loader is included because this paper uses no asset
  prices; its only external input is a rating-transition table, which is
  replaced by a stylized synthetic matrix as documented in
  `results/RESULTS.md`. If a market-data loader is ever added, API keys
  belong only in an uncommitted `.env` (see `.env.example` conventions
  used across this project family), never in code, logs or commits.
* Tests run as `python -m pytest`. Tests that would depend on a live market
  data call are marked `market_data` (registered in `tests/conftest.py`) and
  excluded by `python -m pytest -m "not market_data"`; this repository
  currently has no such tests since it uses no market data.
* After rerunning an experiment, the files it writes in `results/` are
  overwritten. The numbers and verdicts written by hand in
  `results/RESULTS.md` are not regenerated automatically — check them
  against the new files and update them manually after any rerun that
  changes the results.
* Deviations from the paper (data source substitution, interpretations of
  under-specified formulas, reduced Monte Carlo sample sizes) are recorded
  only in `results/RESULTS.md`, never as silent changes to `src/config.py`
  defaults.
