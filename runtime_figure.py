"""Generate the runtime figure used in the paper.

The figure contains F0--F4, IPLF, and one higher-order single-center control in
each filter group.  Runtime is normalized by the conventional EKF within each
application and then averaged across the requested applications.

Usage:
    python runtime_figure.py [runs] [n_jobs] [systems] [repeats]

Examples:
    python runtime_figure.py 200 1 batt,trn 1
    python runtime_figure.py 20 1 batt,trn 1 --output-dir paper_outputs

The default command writes only ``sweep_runtime.png``.  Pass ``--save-data``
to retain the two timing CSV files.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from sweep_all import (CONFIGS, LABEL, SEED0, SPECS, STYLE, VARIANTS, run_configuration,
                       make_filter, ORDER_CONTROL_FOR, runtime_configs_for)

RT_ALL = list(CONFIGS) + list(dict.fromkeys(ORDER_CONTROL_FOR.values()))
TIME_SIGMA = {"batt": 1e-4, "trn": 1.0}
WARMUP = 2


def time_cell(args):
    """Average wall time per Monte Carlo run for one method."""
    name, variant, cfg, runs, repeats = args
    spec = SPECS[name]
    sysm = spec["make"](TIME_SIGMA[name])

    for m in range(WARMUP):
        rng = np.random.default_rng(SEED0 + m)
        run_configuration(make_filter(variant, cfg, sysm), sysm, rng, cfg)

    block_times = []
    for rep in range(repeats):
        t0 = time.perf_counter()
        seed_offset = rep * runs
        for m in range(runs):
            rng = np.random.default_rng(SEED0 + seed_offset + m)
            run_configuration(make_filter(variant, cfg, sysm), sysm, rng, cfg)
        block_times.append((time.perf_counter() - t0) / runs)
    return name, variant, cfg, float(np.mean(block_times))


def make_figure(avg, fname="sweep_runtime.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    xs = np.arange(len(VARIANTS))
    present = {variant: runtime_configs_for(variant) for variant in VARIANTS}
    max_bars = max(len(cfgs) for cfgs in present.values())
    width = 0.84 / max_bars
    labelled = set()

    for x, variant in zip(xs, VARIANTS):
        cfgs = present[variant]
        offsets = (np.arange(len(cfgs)) - (len(cfgs) - 1) / 2.0) * width
        for offset, cfg in zip(offsets, cfgs):
            label = LABEL[cfg] if cfg not in labelled else None
            ax.bar(
                x + offset,
                avg[(variant, cfg)],
                width * 0.80,
                color=STYLE[cfg][0],
                label=label,
                edgecolor="0.25",
                linewidth=0.4,
            )
            labelled.add(cfg)

    ax.set_xticks(xs)
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("Normalized runtime")
    ax.axhline(1.0, color="0.55", lw=0.8, alpha=0.5, zorder=0)
    ax.grid(True, axis="y", which="major", alpha=0.5, linewidth=0.7)
    ax.grid(True, axis="y", which="minor", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.savefig(fname)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="?", type=int, default=100)
    parser.add_argument("n_jobs", nargs="?", type=int, default=1)
    parser.add_argument("systems", nargs="?", default="batt,trn")
    parser.add_argument("repeats", nargs="?", type=int, default=1)
    parser.add_argument("--output-dir", default=".",
                        help="directory for the runtime figure and optional CSVs")
    parser.add_argument("--save-data", action="store_true",
                        help="also retain sweep_runtime.csv and sweep_runtime_raw.csv")
    args = parser.parse_args()
    runs, n_jobs, repeats = args.runs, args.n_jobs, args.repeats
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    if runs < 1 or repeats < 1:
        raise SystemExit("runs and repeats must both be positive")
    bad = [system for system in systems if system not in SPECS]
    if bad:
        raise SystemExit(f"unknown system(s) {bad}; choose from {list(SPECS)}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    jobs = [
        (system, variant, cfg, runs, repeats)
        for system in systems
        for variant in VARIANTS
        for cfg in runtime_configs_for(variant)
    ]
    print(
        f"=== timing {len(jobs)} cells "
        f"({len(systems)} systems x {len(VARIANTS)} variant groups), "
        f"{runs} runs per block, repeats={repeats}, n_jobs={n_jobs} ===",
        flush=True,
    )

    t0 = time.perf_counter()
    raw = {}
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for name, variant, cfg, seconds in executor.map(time_cell, jobs, chunksize=1):
                raw[(name, variant, cfg)] = seconds
    else:
        for job in jobs:
            name, variant, cfg, seconds = time_cell(job)
            raw[(name, variant, cfg)] = seconds
            print(f"  {name:>4s} {variant:<5s} {LABEL[cfg]:<7s} "
                  f"{1e3 * seconds:8.2f} ms/run", flush=True)

    normalized = {}
    for system in systems:
        base = raw[(system, "EKF", "F0")]
        for variant in VARIANTS:
            for cfg in runtime_configs_for(variant):
                normalized[(system, variant, cfg)] = raw[(system, variant, cfg)] / base

    average = {}
    for variant in VARIANTS:
        for cfg in runtime_configs_for(variant):
            average[(variant, cfg)] = float(np.mean([
                normalized[(system, variant, cfg)] for system in systems
            ]))

    if args.save_data:
        with open(os.path.join(output_dir, "sweep_runtime_raw.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["system", "variant", "configuration", "display_label",
                            "seconds_per_run", "normalized_within_system"],
            )
            writer.writeheader()
            for system in systems:
                for variant in VARIANTS:
                    for cfg in runtime_configs_for(variant):
                        writer.writerow({
                            "system": system,
                            "variant": variant,
                            "configuration": cfg,
                            "display_label": LABEL[cfg],
                            "seconds_per_run": raw[(system, variant, cfg)],
                            "normalized_within_system": normalized[(system, variant, cfg)],
                        })
        with open(os.path.join(output_dir, "sweep_runtime.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["variant", "configuration", "display_label",
                            "normalized_runtime"],
            )
            writer.writeheader()
            for variant in VARIANTS:
                for cfg in runtime_configs_for(variant):
                    writer.writerow({
                        "variant": variant,
                        "configuration": cfg,
                        "display_label": LABEL[cfg],
                        "normalized_runtime": average[(variant, cfg)],
                    })

    figure_path = os.path.join(output_dir, "sweep_runtime.png")
    make_figure(average, figure_path)

    colw = max(10, max(len(LABEL[cfg]) for cfg in RT_ALL) + 2)
    print("\nnormalized runtime (conventional EKF = 1), averaged over "
          f"{len(systems)} systems:")
    print("        " + "".join(f"{LABEL[cfg]:>{colw}s}" for cfg in RT_ALL))
    for variant in VARIANTS:
        print(f"  {variant:<6s}" + "".join(
            f"{average[(variant, cfg)]:{colw}.2f}"
            if (variant, cfg) in average else f"{'-':>{colw}s}"
            for cfg in RT_ALL
        ))
    retained = "sweep_runtime.png"
    if args.save_data:
        retained += ", sweep_runtime.csv, and sweep_runtime_raw.csv"
    print(f"\nwrote {retained} to {output_dir} "
          f"({(time.perf_counter() - t0) / 60:.1f} min)")



if __name__ == "__main__":
    main()
