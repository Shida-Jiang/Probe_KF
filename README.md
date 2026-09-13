# Probe Sets for Nonlinear Kalman Filtering

Python code for **“Probe Sets for Nonlinear Kalman Filtering: Multi-Center
Gains and Covariance Recalibration.”** See the paper for the methods and
experimental setup.

Covariances are propagated as square-root factors using QR and Joseph updates.
NEES is calculated directly from these factors.

## Install and test

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python test_filters.py
```

## Run

Run the paper experiments, saving summaries and per-run data for later reuse:

```bash
python run_paper.py --output-dir results/revised --jobs 8 --save-data --keep-cache
```

Adjust the worker count as needed. For a quick workflow check:

```bash
python run_paper.py --quick --output-dir results/quick --jobs 2
```

The output folder contains seven figures and two LaTeX tables. Summary CSVs
are included with `--save-data`. With `--keep-cache`, per-run data remain in
`results/.revised_cache/` for the example above.

## Reuse results

Regenerate the tables or sweep figures from saved CSVs:

```bash
python paper_tables.py results/revised/setups_v4_order_r1000.csv --output-dir results/revised
python sweep_all.py all --figures-only --input-dir results/revised --output-dir results/revised
```

These commands reuse the completed simulations. Add `--setup-cache-dir PATH`
to `run_paper.py` to reuse compatible randomized-study NPZ files and run
missing cells. Its `--from-results PATH` option rebuilds the complete output
set from summary CSVs and performs a fresh runtime benchmark.

Use each script's `--help` option for further arguments.
