# Adapted Wasserstein Distance for Credit Risk: Rating Information and Model-Transfer Hedging

Replication of ten numerical experiments from *Adapted Wasserstein Distance
for Credit Risk: Rating Information and Model-Transfer Hedging*.

The paper studies the adapted (bicausal, nested) Wasserstein distance between
discrete- and continuous-time defaultable-bond price models driven by rating
information, derives closed-form formulas and sharp bounds for this distance,
and uses it to control the error of model-transfer hedging strategies for
credit-sensitive claims.

## What is implemented

| Experiment | Title |
| --- | --- |
| exp_1 | Discrete-time adapted Wasserstein distance for a single rating review and timing of information |
| exp_2 | Multi-period rating migration: dynamic programming and synchronous coupling |
| exp_3 | Discrete-time hedging: momentum payoff replication and stability bounds |
| exp_4 | Adapted Wasserstein distance between Cox bond prices with deterministic hazards |
| exp_5 | Model-transfer hedging in a single calibrated rating-review bucket |
| exp_6 | Rolling bootstrap along a quoted term structure and hedging with a window of liquid bonds |
| exp_7 | Robust pricing over adapted Wasserstein balls and small-dispersion asymptotics |
| exp_8 | Empirical comparison of calibrated rating systems using S&P-style transition data |
| exp_9 | Monte Carlo study of the rolling bootstrap hedge (review / intra-year / CIR) |
| exp_10 | Hedging with multiple instruments: bootstrap desk vs. market desk |

Every experiment is a numerical verification of a closed-form theorem or a
Monte Carlo simulation of a model fully specified by the paper (two- and
three-date rating-review models, finite-state rating chains, Cox bond prices
with deterministic or CIR hazards). No historical market prices are used:
the only external ingredient is a rating-transition matrix, and the paper's
actual source for it — the S&P Global Ratings 2024 annual corporate default
and transition study — is a proprietary PDF report that is not served by the
QuantX Dataloader or by the Massive REST API. `data/rating data` therefore
uses a stylized seven-category transition matrix with the qualitative shape
the paper describes (see `results/RESULTS.md`, "Deviations from the paper").

## Repository layout

```
src/            core library: closed-form formulas, dynamic programming,
                Cox-hazard mismatch, bootstrap/rolling hedges, robust pricing
exp/            one script per experiment, exp_1 .. exp_10
data/           data/generate_synthetic.py (Monte Carlo sample generation)
                and data/rating_matrices.py (stylized S&P-style rating data)
tests/          pytest unit tests for every src/ module
results/        tables, figures and RESULTS.md written by the experiments
```

## Setup

```bash
python -m pip install -r requirements.txt
```

No API keys are required: this replication uses no market data.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
python -m exp.exp7
python -m exp.exp8
python -m exp.exp9
python -m exp.exp10
```

Each experiment writes its tables and figures directly to `results/` and
prints a short log of the realized configuration (sample sizes, maturities,
grid sizes). `exp6`, `exp9` and `exp10` accept a `--full` flag that selects
the paper's Monte-Carlo sample sizes; see `results/RESULTS.md` for the
executed vs. paper configuration of every reduced run.

## Reproducing everything

```bash
bash reproduce.sh
```

## Paper reference

Adapted Wasserstein Distance for Credit Risk: Rating Information and
Model-Transfer Hedging. See `results/RESULTS.md` for the experiment-by-experiment
comparison with the paper's reported values.
