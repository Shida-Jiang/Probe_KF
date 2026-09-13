"""Run the randomized F0-F4/IPLF comparison and higher-order controls.

Each physical setup is evaluated with paired initial errors and noise
realizations across all methods.  With ``--order-controls``, the script also
runs EKF2 versus EKF+probe, EKF3 versus EKF2+probe, Gauss-Hermite versus
UKF+probe, and CKF-5 versus CKF+probe.

Usage
-----
python compare_setups.py [n_setups] [runs] [n_jobs] [variants] [seed] [systems]
python compare_setups.py 300 1000 8 --order-controls
"""

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from filters import COVARIANCE_POLICY, FILTERS
from framework import BACKOUT_POLICY
from metrics import (METRIC_FIELDS, SUMMARY_VERSION, anees_of, geometric_mean,
                     per_run_rmse, summarize_rmse)
from sweep_all import (BURN_IN, SEED0, VARIANTS, CONFIGS, MANIFEST,
                       LABELS_VERSION, ORDER_CONTROL_LABEL, make_filter,
                       moment_calls_per_step, order_control_for,
                       run_configuration)
from systems import BatteryEstimation, FractalTerrain

KEY = {"batt": (0, 2), "trn": (0, 1)}
SYSTEMS = ["batt", "trn"]
SEED_OFFSET = {"batt": 0, "trn": 2}
TITLE = {"batt": "battery state estimation",
         "trn": "terrain-referenced navigation"}
# Variant-specific single-center higher-order measurement control.  The arrays
# are stored in one generic ORDER column because exactly one control applies to
# each row.  This keeps the comparison table symmetric across all four wrapped
# variants while keeping the six principal methods comparable.
ORDER = "ORDER"
OUTDIR_FMT = "setups_v4_cells_r{runs}"
CSV_FMT = "setups_v4{suffix}_r{runs}.csv"
RUN_POLICY = "all-runs-v1"



# ------------------------------------------------------------------ setups
def _lu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_setup(system, rng):
    """Return one sampled physical setup and its constructor arguments."""
    if system == "batt":
        p = dict(sigma_meas=_lu(rng, 1e-6, 1e-2),
                 prior_scale=_lu(rng, 0.25, 4.0),
                 proc_scale=_lu(rng, 0.25, 4.0),
                 soh0=float(rng.uniform(0.84, 0.96)),
                 amp=float(rng.uniform(1.0, 3.0)))
        dsoc = p["amp"] / (6.0 * p["soh0"])
        p["soc0"] = float(rng.uniform(max(0.25, dsoc + 0.05), 0.75))
        kw = dict(sigma_meas=p["sigma_meas"], rest=15,
                  ramp=30, dt=20.0, Q0=1.0, amp=p["amp"],
                  x0=[p["soc0"], 0.0, p["soh0"]],
                  sig_soc=0.05 * p["prior_scale"],
                  sig_uc=5e-3 * p["prior_scale"],
                  sig_soh=0.03 * p["prior_scale"],
                  sig_pro=[2e-6 * p["proc_scale"], 2e-6 * p["proc_scale"],
                           0.0])
        return p, kw
    if system == "trn":
        p = dict(sigma_meas=_lu(rng, 1e-1, 1e3),
                 prior_scale=_lu(rng, 0.5, 2.0),
                 proc_scale=_lu(rng, 0.25, 4.0),
                 hurst=float(rng.uniform(0.75, 0.99)),
                 A0=float(rng.uniform(120.0, 320.0)),
                 map_seed=int(rng.integers(0, 1_000_000)))
        kw = dict(sigma_meas=p["sigma_meas"], n_oct=4, L0=20.0, A0=p["A0"],
                  hurst=p["hurst"], speed=0.25, N=60,
                  sig_init=0.30 * p["prior_scale"],
                  sig_pro=2e-3 * p["proc_scale"], seed=p["map_seed"])
        return p, kw
    raise ValueError(system)


def all_setups(n_setups, master_seed, systems):
    out = {}
    for name in systems:
        rng = np.random.default_rng(master_seed * 1000 + SEED_OFFSET[name])
        out[name] = [sample_setup(name, rng) for _ in range(n_setups)]
    return out


def cell_path(outdir, name, idx, variant, p):
    s = json.dumps({k: float(v) for k, v in sorted(p.items())}, sort_keys=True)
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    return os.path.join(outdir, f"{name}_{idx:03d}_{h}_{variant}.npz")


def build(name, kw):
    if name == "batt":
        return BatteryEstimation(**kw)
    return FractalTerrain(**kw)


# -------------------------------------------------------------------- runs
def run_cell(args):
    name, idx, variant, runs, n_setups, master_seed, outdir, include_order = args
    p, kw = all_setups(n_setups, master_seed, [name])[name][idx]
    path = cell_path(outdir, name, idx, variant, p)
    rule = order_control_for(variant) if include_order else None
    todo = list(CONFIGS) + ([ORDER] if rule else [])

    # Reuse completed configurations and compute the missing ones.
    payload = {}
    if os.path.exists(path):
        with np.load(path) as old:
            payload = {key: old[key] for key in old.files}

    for key, expected in (("backout_policy", BACKOUT_POLICY),
                          ("run_policy", RUN_POLICY),
                          ("covariance_policy", COVARIANCE_POLICY)):
        stored = payload.get(key)
        if (stored is None or np.asarray(stored).shape != ()
                or str(np.asarray(stored).item()) != expected):
            payload = {}
            break
    payload.update(backout_policy=np.asarray(BACKOUT_POLICY),
                   run_policy=np.asarray(RUN_POLICY),
                   covariance_policy=np.asarray(COVARIANCE_POLICY))

    # ORDER is a generic column whose meaning depends on the wrapped variant.
    # Record the concrete rule in the checkpoint and invalidate any legacy
    # ORDER arrays whose rule cannot be verified.
    if rule:
        stored = payload.get("order_rule")
        if stored is not None:
            stored = str(np.asarray(stored).item())
        if stored != rule:
            for key in list(payload):
                if key.endswith(f"_{ORDER}"):
                    payload.pop(key, None)
        payload["order_rule"] = np.asarray(rule)

    def complete(cfg):
        keys = (f"rmse_{cfg}", f"anees_{cfg}", f"fire_{cfg}")
        return all(key in payload and np.asarray(payload[key]).shape == (runs,)
                   for key in keys)

    missing = [cfg for cfg in todo if not complete(cfg)]
    if not missing:
        return path

    K = list(KEY[name])
    sysm = build(name, kw)
    elapsed0 = time.perf_counter()

    for cfg in missing:
        r = np.empty(runs)
        a = np.empty(runs)
        fire = np.empty(runs)
        real = rule if cfg == ORDER else cfg
        for m in range(runs):
            # Identical realization across configs: same seed and draw order.
            rng = np.random.default_rng(SEED0 + m)
            flt = make_filter(variant, real, sysm)
            errs, Ps, aux, factors = run_configuration(
                flt, sysm, rng, real, return_factors=True)
            r[m] = per_run_rmse(errs, K, burn_in=BURN_IN)
            a[m] = anees_of(errs, Ps, K, factors=factors)
            fire[m] = float(aux)
        payload[f"rmse_{cfg}"] = r
        payload[f"anees_{cfg}"] = a
        payload[f"fire_{cfg}"] = fire

    old_secs = float(np.asarray(payload.get("secs", 0.0)))
    payload["secs"] = np.asarray(old_secs + time.perf_counter() - elapsed0)
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


# ---------------------------------------------------------------- reporting
def report(rows, systems):
    """Report geometric means across setups separately for each variant."""
    for name in systems:
        for variant in VARIANTS:
            subset = [row for row in rows
                      if row["system"] == name and row["variant"] == variant]
            if not subset:
                continue
            print(f"\n=== {TITLE[name]} / {variant} === "
                  f"{len(subset)} physical setups")
            print("  Geometric means across setups")
            print(f"    {'method':>14s} {'RMSE RMS':>12s} {'RMSE median':>12s} "
                  f"{'RMSE p95':>12s} {'ANEES':>12s}")
            methods = list(CONFIGS)
            if all(f"rmse_rms_{ORDER}" in row for row in subset):
                methods.append(ORDER)
            for cfg in methods:
                values = [geometric_mean([row[f"{metric}_{cfg}"]
                                          for row in subset])
                          for metric in METRIC_FIELDS]
                label = subset[0]["order_rule"] if cfg == ORDER else cfg
                print(f"    {label:>14s} "
                      + " ".join(f"{value:12.6g}" for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_setups", nargs="?", type=int, default=300)
    parser.add_argument("runs", nargs="?", type=int, default=1000)
    parser.add_argument("n_jobs", nargs="?", type=int, default=1)
    parser.add_argument("variants", nargs="?", default=",".join(VARIANTS))
    parser.add_argument("seed", nargs="?", type=int, default=7)
    parser.add_argument("systems", nargs="?", default=",".join(SYSTEMS))
    parser.add_argument("--order-controls", action="store_true",
                        help="run EKF2/EKF3/GH/CKF-5 single-center controls")
    parser.add_argument("--cache-dir",
                        help="directory of compatible per-cell NPZ checkpoints")
    parser.add_argument("--output-dir", default=".",
                        help="directory for the regenerated summary CSV")
    args = parser.parse_args()

    n_setups, runs, n_jobs = args.n_setups, args.runs, args.n_jobs
    variants = args.variants.split(",")
    seed = args.seed
    systems = args.systems.split(",")
    bad_variants = [v for v in variants if v not in VARIANTS]
    bad_systems = [name for name in systems if name not in SYSTEMS]
    if bad_variants:
        raise SystemExit(f"unknown variant(s) {bad_variants}")
    if bad_systems:
        raise SystemExit(f"unknown system(s) {bad_systems}")
    if min(n_setups, runs, n_jobs) < 1:
        raise SystemExit("n_setups, runs, and n_jobs must be positive")

    include_order = bool(args.order_controls)
    suffix = "_order" if include_order else ""
    outdir = args.cache_dir or OUTDIR_FMT.format(runs=runs)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"label schema {LABELS_VERSION}; configs {CONFIGS}; "
          f"higher-order controls={include_order}")
    print(f"{n_setups} setups x {runs} runs x {len(variants)} variants x "
          f"{len(systems)} systems, checkpoints in {outdir}/", flush=True)

    jobs = [(nm, i, v, runs, n_setups, seed, outdir, include_order)
            for nm in systems for i in range(n_setups) for v in variants]
    t0 = time.perf_counter()
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            paths = list(ex.map(run_cell, jobs, chunksize=1))
    else:
        paths = [run_cell(j) for j in jobs]
    print(f"cells done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    setups = all_setups(n_setups, seed, systems)
    rows = []
    for (nm, i, v, *_), path in zip(jobs, paths):
        d = np.load(path)
        p, _ = setups[nm][i]
        row = dict(system=nm, idx=i, variant=v, labels=LABELS_VERSION,
                   metrics=SUMMARY_VERSION,
                   covariance_policy=COVARIANCE_POLICY,
                   n_paired=runs, secs=float(d["secs"]))
        row.update({k: float(val) for k, val in p.items()})
        rule = order_control_for(v) if include_order else None
        row["order_rule"] = ORDER_CONTROL_LABEL.get(rule, "") if rule else ""
        if rule:
            sysm = build(nm, setups[nm][i][1])
            base = FILTERS[v](sysm)
            control = make_filter(v, rule, sysm)
            row["pts_F4"] = (moment_calls_per_step(sysm, "F4")
                              * int(base.n_points))
            row["pts_ORDER"] = (moment_calls_per_step(sysm, rule)
                                 * int(control.n_points))
            row["spec_ORDER"] = "|".join(MANIFEST[rule])
        for c in list(CONFIGS) + ([ORDER] if rule else []):
            g, rc, mt = MANIFEST[rule if c == ORDER else c]
            rr, aa = d[f"rmse_{c}"], d[f"anees_{c}"]
            row.update({f"{metric}_{c}": value
                        for metric, value in summarize_rmse(rr).items()})
            row[f"anees_{c}"] = float(np.mean(aa))
            row[f"fire_{c}"] = float(np.mean(d[f"fire_{c}"]))
            row[f"ncovfail_{c}"] = int(np.sum(~np.isfinite(aa)))
            row[f"spec_{c}"] = f"{g}|{rc}|{mt}"
        rows.append(row)
        d.close()

    # Systems sample different parameters (battery has soc0/amp, trn has
    # hurst/A0/map_seed), so the header is the union in first-seen order and
    # absent parameters are written blank rather than dropped.
    out = os.path.join(args.output_dir, CSV_FMT.format(runs=runs, suffix=suffix))
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    report(rows, systems)
    print(f"\nwrote {out}  ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
