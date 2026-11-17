#!/usr/bin/env bash
# Runs the test suite and every experiment of the replication, in order,
# exactly as documented in README.md.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Installing dependencies ==="
python -m pip install -r requirements.txt

echo "=== Running tests (excluding market-data tests; none are required here) ==="
python -m pytest -m "not market_data"

for exp in exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9 exp10; do
    echo "=== Running python -m exp.${exp} ==="
    python -m "exp.${exp}"
done

echo "=== Done. See results/RESULTS.md for the experiment-by-experiment comparison. ==="
