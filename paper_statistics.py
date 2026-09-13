"""Print the performance summaries used by the paper's figures and tables.

Usage
-----
python paper_statistics.py sweep_batt_results.csv sweep_trn_results.csv \
    setups_v4_order_r1000.csv

The script writes no files. Randomized-study entries use exactly the same four
geometric means as the tables, separately for each application and KF variant.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from filters import COVARIANCE_POLICY
from metrics import METRIC_FIELDS, SUMMARY_VERSION, geometric_mean
from paper_tables import (
    CONFIGS, DISPLAY, SYSTEM_LABEL, VARIANT_ORDER,
    framework_summary, order_summary, validate_randomized_rows,
)

EXPECTED_NOISE = {
    "batt": np.logspace(-6, -2, 9),
    "trn": np.logspace(0, 3, 9),
}


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or nonnumeric metric {key!r}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if any(row.get("covariance_policy") != COVARIANCE_POLICY for row in rows):
        raise ValueError(
            f"{path}: expected covariance_policy={COVARIANCE_POLICY}; "
            "rerun the simulations before reporting statistics."
        )
    if any(row.get("metrics") != SUMMARY_VERSION for row in rows):
        raise ValueError(
            f"{path}: expected metrics={SUMMARY_VERSION}; regenerate summaries "
            "from per-run results before reporting statistics"
        )
    return rows


def _format_metrics(stats: dict[str, float]) -> str:
    return "; ".join(f"{label}={stats[field]:.6g}" for field, label in zip(
        METRIC_FIELDS, ("RMS", "median", "95th percentile", "ANEES")
    ))


def _analyze_randomized(rows: list[dict[str, str]]) -> list[str]:
    n_setups, n_runs = validate_randomized_rows(rows)
    principal_tables = framework_summary(rows)
    order_tables = order_summary(rows)
    lines: list[str] = []
    for system in ("batt", "trn"):
        lines.append(
            f"=== {SYSTEM_LABEL[system]}: randomized study "
            f"({n_setups} setups per variant, {n_runs} paired runs per setup) ==="
        )
        lines.append("Table 2: geometric means over setups, in original units:")
        for variant in VARIANT_ORDER:
            lines.append(f"  {variant}:")
            for cfg in CONFIGS:
                stats = principal_tables[system][variant][cfg]
                lines.append(f"    {DISPLAY[cfg]:>5s}: {_format_metrics(stats)}")

        lines.append("Table 3: geometric means over setups, in original units:")
        for comparison in order_tables:
            if comparison["system"] != system:
                continue
            for label_key, metrics_key in (("higher", "control_metrics"), ("probe", "probe_metrics")):
                lines.append(
                    f"  {comparison[label_key]:>10s}: "
                    + _format_metrics(comparison[metrics_key])
                )

        lines.append("Paired RMS comparisons (geometric mean of setup-specific ratios):")
        for variant in VARIANT_ORDER:
            subset = [row for row in rows
                      if row["system"] == system and row["variant"] == variant]
            comparisons = []
            for partner in ("F0", "F2", "F3", "ORDER"):
                ratio = geometric_mean([
                    _number(row, "rmse_rms_F4") / _number(row, f"rmse_rms_{partner}")
                    for row in subset
                ])
                comparisons.append(f"F4/{partner}={ratio:.6g}")
            lines.append(f"  {variant}: " + "; ".join(comparisons))
        lines.append("")
    return lines


def _validate_sweep(system: str, rows: list[dict[str, str]]) -> None:
    if any(row.get("covariance_policy") != COVARIANCE_POLICY for row in rows):
        raise ValueError(
            f"{system} sweep must use covariance_policy={COVARIANCE_POLICY}; "
            "rerun the simulations before reporting statistics."
        )
    if any(row.get("metrics") != SUMMARY_VERSION for row in rows):
        raise ValueError(f"{system} sweep must use metrics={SUMMARY_VERSION}")
    expected_noise = EXPECTED_NOISE[system]
    expected = len(expected_noise) * len(VARIANT_ORDER) * len(CONFIGS)
    if len(rows) != expected:
        raise ValueError(f"{system} sweep contains {len(rows)} rows; expected {expected}")
    if any(row.get("system") != system for row in rows):
        raise ValueError(f"{system} sweep contains a row from another application")
    for sigma in expected_noise:
        for variant in VARIANT_ORDER:
            for cfg in CONFIGS:
                matches = [
                    row for row in rows
                    if np.isclose(_number(row, "sigma"), sigma, rtol=1e-12, atol=0.0)
                    and row.get("variant") == variant and row.get("cfg") == cfg
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"{system}: expected one row for sigma={sigma:g}, "
                        f"variant={variant}, cfg={cfg}; found {len(matches)}"
                    )
                for metric in METRIC_FIELDS:
                    _number(matches[0], metric)


def _sweep_value(
    rows: list[dict[str, str]], sigma: float, variant: str, cfg: str, metric: str
) -> float:
    for row in rows:
        if (
            np.isclose(_number(row, "sigma"), sigma, rtol=1e-12, atol=0.0)
            and row.get("variant") == variant and row.get("cfg") == cfg
        ):
            return _number(row, metric)
    raise KeyError((sigma, variant, cfg, metric))


def _analyze_sweep(system: str, rows: list[dict[str, str]]) -> list[str]:
    _validate_sweep(system, rows)
    noise = EXPECTED_NOISE[system]
    low = float(noise[0])
    lines = [f"=== {SYSTEM_LABEL[system]}: fixed noise sweep ==="]
    lines.append(
        f"Within-setup statistics at the lowest measurement-noise level ({low:g}), "
        "in original units:"
    )
    for variant in VARIANT_ORDER:
        lines.append(f"  {variant}:")
        for cfg in CONFIGS:
            stats = {metric: _sweep_value(rows, low, variant, cfg, metric)
                     for metric in METRIC_FIELDS}
            lines.append(f"    {DISPLAY[cfg]:>5s}: {_format_metrics(stats)}")
        ratios = np.array([
            _sweep_value(rows, sigma, variant, "F4", "rmse_rms")
            / _sweep_value(rows, sigma, variant, "F0", "rmse_rms")
            for sigma in noise
        ])
        lines.append(
            f"    F4/F0 RMS ratio range across the {len(noise)} noise levels="
            f"{np.min(ratios):.6g}-{np.max(ratios):.6g}"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_batt_csv", type=Path)
    parser.add_argument("sweep_trn_csv", type=Path)
    parser.add_argument("randomized_csv", type=Path)
    args = parser.parse_args()

    batt = _read_csv(args.sweep_batt_csv)
    trn = _read_csv(args.sweep_trn_csv)
    randomized = _read_csv(args.randomized_csv)
    lines = [
        "Each per-run RMSE averages the two key-state RMSEs.",
        "The 95th percentile is calculated within each setup, over its runs.",
        "ANEES follows the paper's definition and has nominal value one.",
        "",
    ]
    lines.extend(_analyze_sweep("batt", batt))
    lines.extend(_analyze_sweep("trn", trn))
    lines.extend(_analyze_randomized(randomized))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
