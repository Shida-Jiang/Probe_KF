"""Generate the paper's two tables from per-setup performance summaries.

Usage
-----
python paper_tables.py setups_v4_order_r1000.csv --output-dir paper_outputs

Each table entry is the geometric mean, over physical setups, of one of four
within-setup statistics: RMS, median, or 95th percentile of per-run RMSE, or
ANEES. The four filter variants remain separate in the framework table.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from filters import COVARIANCE_POLICY
from metrics import METRIC_FIELDS, SUMMARY_VERSION, geometric_mean

CONFIGS = ["F0", "F1", "F2", "F3", "F4", "IPLF2"]
DISPLAY = {"IPLF2": "IPLF", **{f"F{i}": f"F{i}" for i in range(5)}}
SYSTEMS = ["batt", "trn"]
SYSTEM_LABEL = {
    "batt": "Battery state estimation",
    "trn": "Terrain-referenced navigation",
}
VARIANT_ORDER = ["EKF", "EKF2", "UKF", "CKF"]
PAIR_LABEL = {
    "EKF": ("EKF2", "EKF+probe"),
    "EKF2": ("EKF3", "EKF2+probe"),
    "UKF": ("GH", "UKF+probe"),
    "CKF": ("CKF-5", "CKF+probe"),
}
METRICS = METRIC_FIELDS
HIGHLIGHT_MARGIN = 0.05
MetricSummary = Dict[str, float]
FrameworkSummary = Dict[str, Dict[str, Dict[str, MetricSummary]]]


def _tex_integer(value: int) -> str:
    return f"{value:,}".replace(",", r"{,}")


def _number(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or nonnumeric metric {key!r}") from exc


def _validate_schema(rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("empty randomized-study data")
    if any(row.get("labels") != "v4" for row in rows):
        raise ValueError("expected label schema v4")
    if any(row.get("covariance_policy") != COVARIANCE_POLICY for row in rows):
        raise ValueError(
            f"expected covariance_policy={COVARIANCE_POLICY}; rerun "
            "compare_setups.py --order-controls. Earlier filtering results "
            "cannot be updated by reaggregating their per-run data."
        )
    if any(row.get("metrics") != SUMMARY_VERSION for row in rows):
        raise ValueError(
            f"expected metrics={SUMMARY_VERSION}; regenerate the CSV from "
            "per-run results with compare_setups.py"
        )


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    _validate_schema(rows)
    return rows


def validate_randomized_rows(rows: List[Dict[str, str]]) -> tuple[int, int]:
    """Require one current-summary cell per setup and filter variant."""
    _validate_schema(rows)
    if any(not row.get("idx") for row in rows):
        raise ValueError("every randomized-study row requires a setup index")
    setups = {
        system: {row["idx"] for row in rows if row.get("system") == system}
        for system in SYSTEMS
    }
    counts = {system: len(ids) for system, ids in setups.items()}
    if not all(counts.values()) or len(set(counts.values())) != 1:
        raise ValueError(f"expected the same number of setups per application, got {counts}")
    cells = {(row.get("system"), row["idx"], row.get("variant")) for row in rows}
    expected = {
        (system, setup, variant)
        for system in SYSTEMS for setup in setups[system] for variant in VARIANT_ORDER
    }
    if cells != expected or len(cells) != len(rows):
        raise ValueError("expected exactly one cell per system, setup, and each of the four filter variants")
    if any(not row.get("order_rule") for row in rows):
        raise ValueError("higher-order results are required for every cell; use --order-controls")
    try:
        runs = {_number(row, "n_paired") for row in rows}
        if len(runs) != 1:
            raise ValueError
        n_runs = runs.pop()
        if not np.isfinite(n_runs) or n_runs < 1 or n_runs != int(n_runs):
            raise ValueError
    except ValueError as exc:
        raise ValueError("expected one positive integer Monte Carlo count in n_paired") from exc
    for row in rows:
        for cfg in CONFIGS + ["ORDER"]:
            for metric in METRICS:
                _number(row, f"{metric}_{cfg}")
    return counts[SYSTEMS[0]], int(n_runs)


def _aggregate(subset: List[Dict[str, str]], cfg: str) -> MetricSummary:
    return {
        metric: geometric_mean([_number(row, f"{metric}_{cfg}") for row in subset])
        for metric in METRICS
    }


def framework_summary(rows: Iterable[Dict[str, str]]) -> FrameworkSummary:
    """Geometrically average each statistic over setups, separately by variant."""
    rows = list(rows)
    _validate_schema(rows)
    result: FrameworkSummary = {}
    for system in SYSTEMS:
        system_result = {}
        for variant in VARIANT_ORDER:
            subset = [row for row in rows
                      if row.get("system") == system and row.get("variant") == variant]
            if subset:
                system_result[variant] = {cfg: _aggregate(subset, cfg) for cfg in CONFIGS}
        if system_result:
            result[system] = system_result
    return result


def order_summary(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    """Geometrically average both members of each higher-order comparison."""
    rows = list(rows)
    _validate_schema(rows)
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        system, variant = row.get("system", ""), row.get("variant", "")
        if system in SYSTEM_LABEL and variant in PAIR_LABEL and row.get("order_rule"):
            grouped[(system, variant)].append(row)
    output: List[Dict[str, object]] = []
    for system in SYSTEMS:
        for variant in VARIANT_ORDER:
            subset = grouped.get((system, variant), [])
            if not subset:
                continue
            higher_label, probe_label = PAIR_LABEL[variant]
            output.append({
                "system": system, "variant": variant,
                "higher": higher_label, "probe": probe_label,
                "control_metrics": _aggregate(subset, "ORDER"),
                "probe_metrics": _aggregate(subset, "F4"),
            })
    return output


def _bold_winners(stats: Dict[str, MetricSummary]) -> Dict[str, set[str]]:
    """Bold a metric only when its unrounded value is >5% below the runner-up."""
    winners = {name: set() for name in stats}
    for metric in METRICS:
        ranked = sorted((values[metric], name) for name, values in stats.items()
                        if np.isfinite(values[metric]))
        if len(ranked) >= 2 and ranked[0][0] < (1.0 - HIGHLIGHT_MARGIN) * ranked[1][0]:
            winners[ranked[0][1]].add(metric)
    return winners


def _bold_near_best(stats: Dict[str, MetricSummary]) -> Dict[str, set[str]]:
    """Bold every finite value at most 5% above its column's minimum."""
    highlighted = {name: set() for name in stats}
    for metric in METRICS:
        finite = [(values[metric], name) for name, values in stats.items()
                  if np.isfinite(values[metric])]
        if not finite:
            continue
        cutoff = (1.0 + HIGHLIGHT_MARGIN) * min(value for value, _ in finite)
        for value, name in finite:
            if value <= cutoff:
                highlighted[name].add(metric)
    return highlighted


def _cells(stats: MetricSummary, bold: set[str]) -> List[str]:
    cells = []
    for metric in METRICS:
        number = stats[metric]
        if not np.isfinite(number):
            cells.append("--")
            continue
        formatted = f"{number:.3g}"
        if "e" in formatted:
            mantissa, exponent = formatted.split("e")
            formatted = rf"{mantissa}\mathbin{{\times}}10^{{{int(exponent)}}}"
        if metric in bold:
            formatted = rf"\boldsymbol{{{formatted}}}"
        cells.append(f"${formatted}$")
    return cells


def _table_start(caption: str, label: str, spacing: str,
                 *, variant_column: bool = False) -> List[str]:
    columns = "llcccc|cccc" if variant_column else "lcccc|cccc"
    left_columns = 6 if variant_column else 5
    row_header = "KF & Method" if variant_column else "Method"
    return [
        r"\begin{table*}[htbp]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\scriptsize", rf"\setlength{{\tabcolsep}}{{{spacing}pt}}",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\aboverulesep}{0pt}", r"\setlength{\belowrulesep}{0pt}",
        r"\vspace{3pt}",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        rf"\multicolumn{{{left_columns}}}{{c|}}{{Battery state estimation}}",
        r"& \multicolumn{4}{c}{Terrain-referenced navigation}\\",
        row_header + r" & RMS & Median & 95th percentile & ANEES",
        r"& RMS & Median & 95th percentile & ANEES\\",
        r"\midrule",
    ]


def _write_table(lines: List[str], path: str) -> None:
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _metric_caption(n_setups: int) -> str:
    return (
        "Each per-run RMSE averages the two key-state RMSEs. Within each setup, we calculate "
        "the RMS, median, and 95th percentile of these values, and ANEES from \\eqref{eq:anees}. "
        f"Each entry is the geometric mean of the corresponding statistic over {n_setups} setups. "
        "Battery RMSE is dimensionless. Navigation RMSE is in kilometres. ANEES has nominal value one. "
    )


def write_framework_table(summary: FrameworkSummary, path: str,
                          *, n_setups: int, n_runs: int) -> None:
    if any(system not in summary or set(summary[system]) != set(VARIANT_ORDER)
           for system in SYSTEMS):
        raise ValueError("both applications and all four variants are required")
    caption = (
        f"Randomized study with {n_setups} physical setups per application and "
        rf"${_tex_integer(n_runs)}$ paired runs per filter-setup cell. "
        + _metric_caption(n_setups)
        + "The four nonlinear-KF variants are reported separately. Within each variant and application, "
        r"all values no more than 5\% above the minimum for that metric are bolded."
    )
    lines = _table_start(caption, "tab:randomized", "2.4", variant_column=True)
    for index, variant in enumerate(VARIANT_ORDER):
        if index:
            lines.append(r"\midrule")
        bold = {system: _bold_near_best(summary[system][variant]) for system in SYSTEMS}
        for row_index, cfg in enumerate(CONFIGS):
            cells = [rf"\textbf{{{variant}}}" if row_index == 0 else "", DISPLAY[cfg]]
            for system in SYSTEMS:
                cells.extend(_cells(summary[system][variant][cfg], bold[system][cfg]))
            lines.append(" & ".join(cells) + r"\\")
    _write_table(lines, path)


def write_order_table(summary: List[Dict[str, object]], path: str,
                      *, n_setups: int) -> None:
    indexed = {(row["system"], row["variant"]): row for row in summary}
    expected = {(system, variant) for system in SYSTEMS for variant in VARIANT_ORDER}
    if len(summary) != 8 or set(indexed) != expected:
        raise ValueError(f"expected eight higher-order comparisons, found {len(summary)}")
    caption = (
        f"Randomized higher-order comparison over {n_setups} setups per application. "
        + _metric_caption(n_setups)
        + r"Within each two-row comparison, a value is bolded when it is more than 5\% "
        "below its partner's value."
    )
    lines = _table_start(caption, "tab:order_controls", "2.2")
    for index, variant in enumerate(VARIANT_ORDER):
        if index:
            lines.append(r"\midrule")
        pairs = {system: {
            "control": indexed[(system, variant)]["control_metrics"],
            "probe": indexed[(system, variant)]["probe_metrics"],
        } for system in SYSTEMS}
        bold = {system: _bold_winners(pairs[system]) for system in SYSTEMS}
        for method, label in zip(("control", "probe"), PAIR_LABEL[variant]):
            cells = [label]
            for system in SYSTEMS:
                cells.extend(_cells(pairs[system][method], bold[system][method]))
            lines.append(" & ".join(cells) + r"\\")
    _write_table(lines, path)


def generate_tables(csv_path: str, output_dir: str) -> tuple[str, str]:
    rows = _read_rows(csv_path)
    n_setups, n_runs = validate_randomized_rows(rows)
    os.makedirs(output_dir, exist_ok=True)
    framework_path = os.path.join(output_dir, "table_randomized.tex")
    order_path = os.path.join(output_dir, "table_order_controls.tex")
    write_framework_table(framework_summary(rows), framework_path,
                          n_setups=n_setups, n_runs=n_runs)
    write_order_table(order_summary(rows), order_path, n_setups=n_setups)
    return framework_path, order_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="current setup summaries from compare_setups.py")
    parser.add_argument("--output-dir", default="paper_outputs")
    args = parser.parse_args()
    paths = generate_tables(args.csv_path, args.output_dir)
    print("wrote " + " and ".join(paths))


if __name__ == "__main__":
    main()
