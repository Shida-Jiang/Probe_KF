"""Reproduce the figures, tables, and quoted statistics in the paper.

A standard run leaves exactly seven PNG figures and two LaTeX tables in the
output directory.  The compact CSV files used to calculate the paper's text
statistics are kept in a temporary cache, analyzed, and removed after success.
Pass ``--save-data`` to retain those three CSVs for an independent audit,
or ``--keep-cache`` to preserve the per-run arrays for later reaggregation.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SWEEP_FIGURES = [
    f"sweep_{system}_fig{index}_{suffix}.png"
    for system in ("batt", "trn")
    for index, suffix in ((1, "rmse"), (2, "anees"), (3, "tail"))
]
PUBLICATION_FILES = SWEEP_FIGURES + [
    "sweep_runtime.png",
    "table_randomized.tex",
    "table_order_controls.tex",
]
DATA_FILES = [
    "sweep_batt_results.csv",
    "sweep_trn_results.csv",
    "setups_v4_order_r1000.csv",
]


def _run(arguments: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(item) for item in arguments)
    print(f"\n> {printable}", flush=True)
    subprocess.run(arguments, cwd=cwd, check=True)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"output directory is not empty: {path}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _validate_output(output: Path, source: Path | None) -> None:
    """Keep output replacement separate from the code and source results."""
    if PACKAGE_DIR == output or output in PACKAGE_DIR.parents:
        raise SystemExit("The output directory must not contain the source code.")
    if source is not None and (source == output or output in source.parents):
        raise SystemExit("The output directory must not contain --from-results.")


def _copy_sweep_figures(source: Path, output: Path) -> None:
    for name in SWEEP_FIGURES:
        source_path = source / name
        if not source_path.exists():
            raise RuntimeError(f"missing sweep figure: {source_path}")
        shutil.copy2(source_path, output / name)


def _find_randomized_csv(source: Path) -> Path | None:
    preferred = source / "setups_v4_order_r1000.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(
        path for path in source.glob("setups_v4_order_r*.csv")
        if "comparison" not in path.name
    )
    return candidates[-1] if candidates else None


def _regenerate_sweeps(source: Path, output: Path, cache: Path) -> dict[str, Path]:
    raw: dict[str, Path] = {}
    cache.mkdir(parents=True, exist_ok=True)
    for system in ("batt", "trn"):
        csv_path = source / f"sweep_{system}_results.csv"
        if not csv_path.exists():
            raise SystemExit(
                f"Missing fixed-sweep summary: {csv_path}. "
                "Rerun the sweep with --save-data. Existing figures cannot "
                "verify or reconstruct the current RMSE summaries."
            )
        _run([
            sys.executable,
            str(PACKAGE_DIR / "sweep_all.py"),
            system,
            "--figures-only", "--save-csv",
            "--input-dir", str(source),
            "--output-dir", str(cache),
        ])
        raw[system] = cache / csv_path.name
    _copy_sweep_figures(cache, output)
    return raw


def _generate_tables(source: Path, output: Path) -> Path:
    randomized_csv = _find_randomized_csv(source)
    if randomized_csv is not None:
        _run([
            sys.executable,
            str(PACKAGE_DIR / "paper_tables.py"),
            str(randomized_csv),
            "--output-dir", str(output),
        ])
        return randomized_csv

    raise SystemExit(
        f"No randomized-study summary CSV was found in {source}. "
        "Run compare_setups.py --order-controls with --cache-dir pointing "
        "to compatible saved per-cell NPZ files to regenerate the summaries, "
        "or rerun the simulations. Existing tables cannot reconstruct them."
    )


def _print_statistics(sweeps: dict[str, Path], randomized_csv: Path | None) -> bool:
    if set(sweeps) != {"batt", "trn"} or randomized_csv is None:
        print(
            "\nRaw CSVs were not all available, so quoted text statistics were not "
            "recalculated. The figures and tables were still reproduced.",
            flush=True,
        )
        return False
    _run([
        sys.executable,
        str(PACKAGE_DIR / "paper_statistics.py"),
        str(sweeps["batt"]),
        str(sweeps["trn"]),
        str(randomized_csv),
    ])
    return True


def _retain_data(
    sweeps: dict[str, Path], randomized_csv: Path | None, output: Path
) -> None:
    if set(sweeps) != {"batt", "trn"} or randomized_csv is None:
        raise SystemExit("--save-data requires all three completed result CSVs")
    sources = {
        "sweep_batt_results.csv": sweeps["batt"],
        "sweep_trn_results.csv": sweeps["trn"],
        "setups_v4_order_r1000.csv": randomized_csv,
    }
    for name, source in sources.items():
        shutil.copy2(source, output / name)


def _from_results(
    source: Path,
    output: Path,
    cache: Path,
    *,
    runtime_runs: int,
    runtime_repeats: int,
    save_data: bool,
) -> None:
    sweeps = _regenerate_sweeps(source, output, cache)
    randomized_csv = _generate_tables(source, output)
    _print_statistics(sweeps, randomized_csv)
    if save_data:
        _retain_data(sweeps, randomized_csv, output)

    _runtime_figure(output, runtime_runs, runtime_repeats)


def _runtime_figure(output: Path, runs: int, repeats: int) -> None:
    _run([
        sys.executable,
        str(PACKAGE_DIR / "runtime_figure.py"),
        str(runs), "1", "batt,trn", str(repeats),
        "--output-dir", str(output),
    ])


def _full_run(
    output: Path,
    cache: Path,
    *,
    jobs: int,
    sweep_runs: int,
    setups: int,
    setup_runs: int,
    runtime_runs: int,
    runtime_repeats: int,
    save_data: bool,
    keep_cache: bool = False,
    setup_cache: Path | None = None,
) -> None:
    cache.mkdir(parents=True, exist_ok=True)

    # Keep the summary CSVs in the temporary cache so every textual statistic
    # can be reproduced without leaving intermediate data in the final folder.
    _run([
        sys.executable,
        str(PACKAGE_DIR / "sweep_all.py"),
        "all", str(sweep_runs), str(jobs),
        "--output-dir", str(cache),
        "--save-data" if keep_cache else "--save-csv",
    ])
    _copy_sweep_figures(cache, output)

    _run([
        sys.executable,
        str(PACKAGE_DIR / "compare_setups.py"),
        str(setups), str(setup_runs), str(jobs),
        "--order-controls",
        "--cache-dir", str(setup_cache or cache / f"setups_v4_cells_r{setup_runs}"),
        "--output-dir", str(cache),
    ], cwd=cache)
    randomized_csv = cache / f"setups_v4_order_r{setup_runs}.csv"
    if not randomized_csv.exists():
        raise RuntimeError(f"randomized-study output was not created: {randomized_csv}")

    _run([
        sys.executable,
        str(PACKAGE_DIR / "paper_tables.py"),
        str(randomized_csv),
        "--output-dir", str(output),
    ])
    sweeps = {
        system: cache / f"sweep_{system}_results.csv"
        for system in ("batt", "trn")
    }
    _print_statistics(sweeps, randomized_csv)
    if save_data:
        _retain_data(sweeps, randomized_csv, output)

    _runtime_figure(output, runtime_runs, runtime_repeats)


def _verify_output(output: Path, *, save_data: bool) -> list[str]:
    expected = PUBLICATION_FILES + (DATA_FILES if save_data else [])
    missing = [name for name in expected if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"missing final output(s): {', '.join(missing)}")
    extras = sorted(path.name for path in output.iterdir() if path.name not in expected)
    if extras:
        raise RuntimeError(
            "unexpected files were written to the paper output folder: "
            + ", ".join(extras)
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper_outputs")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--sweep-runs", type=int, default=10_000)
    parser.add_argument("--setups", type=int, default=300)
    parser.add_argument("--setup-runs", type=int, default=1_000)
    parser.add_argument("--runtime-runs", type=int, default=200)
    parser.add_argument("--runtime-repeats", type=int, default=1)
    parser.add_argument(
        "--from-results", type=Path,
        help="regenerate figures/tables from current-policy CSVs and rerun the runtime benchmark",
    )
    parser.add_argument(
        "--setup-cache-dir", type=Path,
        help="reuse and preserve compatible randomized-study per-cell NPZ checkpoints",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="small smoke run that checks the complete pipeline",
    )
    parser.add_argument(
        "--save-data", action="store_true",
        help="retain the three compact result CSVs in the output directory",
    )
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="retain fixed-sweep per-run arrays and randomized-study checkpoints after success",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    output = Path(args.output_dir).resolve()
    source = args.from_results.resolve() if args.from_results is not None else None
    setup_cache = args.setup_cache_dir.resolve() if args.setup_cache_dir is not None else None
    _validate_output(output, source)
    if setup_cache is not None and (output == setup_cache or output in setup_cache.parents):
        raise SystemExit("The output directory must not contain a reused checkpoint cache.")
    if source is not None and setup_cache is not None:
        raise SystemExit("Use --setup-cache-dir with a full run; --from-results requires updated CSVs.")
    cache = output.parent / f".{output.name}_cache"
    for protected in (source, setup_cache):
        if protected is not None and (cache == protected or cache in protected.parents):
            raise SystemExit(f"The temporary cache {cache} would contain a reused input. Choose another output directory.")
    _prepare_output(output, args.overwrite)

    if args.quick:
        args.sweep_runs = 1
        args.setups = 1
        args.setup_runs = 1
        args.runtime_runs = 1
        args.runtime_repeats = 1

    try:
        if source is not None:
            _from_results(
                source,
                output,
                cache,
                runtime_runs=args.runtime_runs,
                runtime_repeats=args.runtime_repeats,
                save_data=args.save_data,
            )
        else:
            _full_run(
                output,
                cache,
                jobs=args.jobs,
                sweep_runs=args.sweep_runs,
                setups=args.setups,
                setup_runs=args.setup_runs,
                runtime_runs=args.runtime_runs,
                runtime_repeats=args.runtime_repeats,
                save_data=args.save_data,
                keep_cache=args.keep_cache,
                setup_cache=setup_cache,
            )
        expected = _verify_output(output, save_data=args.save_data)
    except Exception:
        print(
            f"\nThe run did not finish. Intermediate data remain in {cache}",
            file=sys.stderr,
        )
        raise
    else:
        if not args.keep_cache and cache.exists():
            shutil.rmtree(cache)
        print("\nPaper outputs:")
        for name in expected:
            print(f"  {output / name}")


if __name__ == "__main__":
    main()
