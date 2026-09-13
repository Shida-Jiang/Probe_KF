"""Fixed measurement-noise sweeps for the probe-set nonlinear KF study.

Usage:
    python sweep_all.py {batt|trn|all} [runs] [n_jobs]

The default command writes only the three PNG figures used in the paper for
each requested system.  Pass ``--save-csv`` to retain the compact summary CSV,
``--save-perrun`` to retain the per-run NPZ, or ``--save-data`` for both.
Existing complete CSV files can be replotted with ``--figures-only``.

The fixed sweeps compare F0--F4 and IPLF on identical noise realizations.  The
higher-order single-center controls are evaluated in ``compare_setups.py`` and
``runtime_figure.py``.
"""
import argparse
import csv
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from systems import FractalTerrain, BatteryEstimation
from systems import hard_terrain_map
from filters import FILTERS, COVARIANCE_POLICY, chol_lower, sym
from metrics import (SUMMARY_VERSION, METRIC_FIELDS, per_run_rmse,
                     summarize_rmse, nees_curve)
import filters as F
import framework as FW

# ----------------------------------------------------------------- settings
M_DEFAULT = 1000
N_JOBS_DEFAULT = max(1, os.cpu_count() or 1)
SEED0 = 1000
BURN_IN = 0.75          # fraction of the trajectory discarded for the RMSE
IPLF_TOL = 1e-4
IPLF_MAX_ITER = 20

VARIANTS = ["EKF", "EKF2", "UKF", "CKF"]
CONFIGS = ["F0", "F1", "F2", "F3", "F4", "IPLF2"]
LABELS_VERSION = "v4"
# One single-center higher-order measurement control for every wrapped variant.
# These controls are used by compare_setups.py and runtime_figure.py, not by
# the fixed measurement-noise sweep:
#   EKF2C vs EKF+F4, EKF3C vs EKF2+F4, GH vs UKF+F4, CKF5 vs CKF+F4.
ORDER_CONTROL_FOR = {
    "EKF": "EKF2C",
    "EKF2": "EKF3C",
    "UKF": "GH",
    "CKF": "CKF5",
}
ORDER_CONTROL_LABEL = {
    "EKF2C": "EKF2",
    "EKF3C": "EKF3",
    "GH": "GH",
    "CKF5": "CKF-5",
}
# gain / recal / metric of every config, written into the CSV so a file can
# never be read under the wrong label schema.
MANIFEST = {
    "F0": ("conventional", "none", "none"),
    "F1": ("single", "single", "I"),
    "F2": ("single", "set", "Pinv"),
    "F3": ("mean", "single", "Pinv"),
    "F4": ("mean", "set", "Pinv"),
    "IPLF2": ("iplf", "none", "none"),
    "EKF2C": ("single", "single", "Pinv"),
    "EKF3C": ("single", "single", "Pinv"),
    "GH": ("single", "single", "Pinv"),
    "CKF5": ("single", "single", "Pinv"),
}
def configs_for(variant):
    """Principal frameworks shown in every fixed-sweep panel.

    ``variant`` is retained in the interface because the runtime and randomized
    comparison helpers use variant-specific higher-order controls.  The fixed
    sweep itself intentionally uses the same six configurations in every panel.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return list(CONFIGS)


def order_control_for(variant):
    """Single-center higher-order measurement control paired with one variant."""
    return ORDER_CONTROL_FOR[variant]


def runtime_configs_for(variant):
    """Frameworks and the one paired higher-order control for a runtime group."""
    return list(CONFIGS) + [order_control_for(variant)]


def moment_calls_per_step(sysm, cfg):
    """Number of measurement moment-map calls made by one configuration."""
    n = sysm.n_x
    if cfg == "F0":
        return 1
    if cfg in {"F1", "EKF2C", "EKF3C", "GH", "CKF5"}:
        return 2
    if cfg == "F2":
        return 1 + (n + 1)
    if cfg == "F3":
        return (n + 1) + 1 + 1  # update probes + centre zhat + recalibration
    if cfg == "F4":
        return (n + 1) + 1 + (n + 1)
    raise ValueError(f"fixed call count unavailable for {cfg}")


def gh_order(n, target_points):
    """Order whose tensor grid is closest to the requested points per call."""
    return min(range(2, 8), key=lambda order: abs(order ** n - target_points))


def make_filter(variant, cfg, sysm):
    """Construct a wrapped filter or a single-center higher-order control."""
    if cfg == "EKF2C":
        return F.ekf2_control(sysm)
    if cfg == "EKF3C":
        return F.ekf3_control(sysm)
    if cfg == "GH":
        n = sysm.n_x
        base_points = 2 * n + 1
        target_per_call = moment_calls_per_step(sysm, "F4") * base_points / 2.0
        return F.gh_control(sysm, gh_order(n, target_per_call))
    if cfg == "CKF5":
        return F.ckf5_control(sysm)
    return FILTERS[variant](sysm)


def budget_table(sysm):
    """Measurement-function evaluations per step for the fixed-sweep methods."""
    rows = []
    for variant in VARIANTS:
        for cfg in configs_for(variant):
            if cfg == "IPLF2":
                continue
            flt = make_filter(variant, cfg, sysm)
            points_per_call = int(flt.n_points)
            calls = moment_calls_per_step(sysm, cfg)
            rows.append((variant, cfg, calls, points_per_call,
                         calls * points_per_call))
    return rows

# Display names.  Label schema v4: F0..F4 plus the IPLF baseline and the
# keys and the labels coincide; only the IPLF baseline needs a longer label.
LABEL = {"F0": "F0", "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4",
         "EKF2C": "EKF2", "EKF3C": "EKF3", "GH": "GH", "CKF5": "CKF-5",
         "IPLF2": r"IPLF ($\gamma=10^{-4}$)"}
# Colour and marker. Figures 1-2 use solid lines; Figure 3 uses solid for the
# median and dotted for the p95 within each colour. F4 is emphasized so it
# remains visible when several curves overlap.
STYLE = {"F0": ("0.15", "o"), "F1": ("0.55", "s"), "F2": ("tab:blue", "^"),
         "EKF2C": ("tab:cyan", "P"), "EKF3C": ("tab:red", "h"),
         "GH": ("tab:brown", "d"), "CKF5": ("tab:olive", "*"),
         "F3": ("tab:orange", "v"), "F4": ("tab:pink", "X"),
         "IPLF2": ("tab:purple", "+")}


def decades(a, n=9, span=4.0):
    """n points at even log spacing over `span` decades starting at 10**a."""
    return list(10.0 ** np.linspace(a, a + span, n))


# --------------------------------------------------------------- the systems
# ``key`` lists the state indices that share a unit and define the metrics.
TRN_FIXED_KW = dict(
    n_oct=4, L0=20.0, A0=200.0, hurst=0.97, speed=0.25, N=60,
    sig_init=0.45, sig_pro=0.0005,
    angles=hard_terrain_map()[0], phases=hard_terrain_map()[1],
)
BATT_FIXED_KW = dict(
    N=60, rest=15, ramp=30, dt=20.0, Q0=1.0, amp=1.3,
    x0=[0.35, 0.0, 0.86], sig_soc=0.085, sig_uc=0.0085,
    sig_soh=0.051, sig_pro=[6e-7, 6e-7, 0.0],
)

SPECS = {
    "trn": dict(
        title="terrain-referenced navigation",
        make=lambda sig: FractalTerrain(sig, **TRN_FIXED_KW),
        noise=decades(0.0, span=3.0),  # 1 .. 1e3 m
        xlabel=r"Altimeter noise s.d. $\sigma_z$ (m)",
        key=(0, 1), key_lbl="position", unit="km",
    ),
    "batt": dict(
        title="battery state estimation",
        make=lambda sig: BatteryEstimation(sigma_meas=sig, **BATT_FIXED_KW),
        noise=decades(-6.0),  # 1e-6 .. 1e-2 V
        xlabel=r"Voltage noise s.d. $\sigma_z$ (V)",
        key=(0, 2), key_lbl="SOC / SOH", unit="-",
    ),
}


# ---------------------------------------------------------------- IPLF
def _kld_factors(u1, L1, u0, L0):
    """Gaussian KLD evaluated from the two covariance square roots."""
    n = len(u0)
    d = np.linalg.solve(L0, u0 - u1)
    relative = np.linalg.solve(L0, L1)
    logdet = 2.0 * (np.sum(np.log(np.diag(L0))) - np.sum(np.log(np.diag(L1))))
    return 0.5 * (np.sum(relative ** 2) + d @ d - n + logdet)


def iplf_update_sqrt(flt, xp, Lp, z, k, gamma, max_iter):
    """IPLF iteration with the original residual model in square-root form."""
    u, L = xp.copy(), Lp.copy()
    it = 0
    for it in range(1, max_iter + 1):
        try:
            zbar, B, E = flt.moments_sqrt(u, L, k)
            moment = (zbar, B @ np.linalg.solve(L, Lp), E)
            Kk = FW.gain_from_factors(Lp, [moment], [1.0])
            innovation = z - zbar - B @ np.linalg.solve(L, xp - u)
            u_new = xp + Kk @ innovation
            L_new = FW.joseph_factor(Lp, Kk, [moment], [1.0])
            if not (np.all(np.isfinite(u_new)) and np.all(np.isfinite(L_new))):
                raise np.linalg.LinAlgError
            if np.any(np.diag(L_new) <= 0):
                raise np.linalg.LinAlgError
            d = _kld_factors(u_new, L_new, u, L)
        except np.linalg.LinAlgError:
            break
        u, L = u_new, L_new
        if d < gamma:
            break
    return u, L, it


def iplf_run_once(flt, s, rng, gamma, max_iter, *, return_factors=False):
    """Consume randomness in the same order as ``framework.run_once``.

    Every configuration therefore sees identical noise at a given seed and the
    Monte Carlo comparisons remain paired.
    """
    N, n = s.N, s.n_x
    s.sample_truth(rng)
    x = s.draw_init(rng)
    L = chol_lower(s.P0)
    errs = np.zeros((n, N + 1))
    Ps = np.zeros((N + 1, n, n))
    factors = np.zeros_like(Ps) if return_factors else None
    errs[:, 0] = x - s.x_true[:, 0]
    Ps[0] = s.P0
    if return_factors:
        factors[0] = L
    iters = 0
    for k in range(1, N + 1):
        s._k = k - 1
        x_pred, L_pred = flt.predict_sqrt(x, L)
        z = s.measure(k, rng)
        x, L, it = iplf_update_sqrt(flt, x_pred, L_pred, z, k, gamma, max_iter)
        iters += it
        errs[:, k] = x - s.x_true[:, k]
        Ps[k] = L @ L.T
        if return_factors:
            factors[k] = L
    result = (errs, Ps, iters / N)
    return result + (factors,) if return_factors else result


# ------------------------------------------------------------------ one cell
def run_configuration(flt, sysm, rng, cfg, *, return_factors=False):
    options = dict(return_factors=return_factors)
    if cfg == "F0":
        return FW.run_conventional_once(flt, sysm, rng, **options)
    if cfg == "F1":
        # metric="I" reproduces the published raw-trace back-out rule.
        return FW.run_once(flt, sysm, rng, "single", "single",
                           metric="I", **options)
    if cfg == "F2":
        # single-point update, probe-set recalibration, invariant back-out
        return FW.run_once(flt, sysm, rng, "single", "set", **options)
    if cfg == "F3":
        # probe-mean update, single-point recalibration, invariant back-out
        return FW.run_once(flt, sysm, rng, "mean", "single", **options)
    if cfg == "F4":
        # the proposed framework: probe-mean update + probe-set recalibration
        return FW.run_once(flt, sysm, rng, "mean", "set", **options)
    if cfg in ("EKF2C", "EKF3C", "GH", "CKF5"):
        # Higher-order measurement control: the richer moment map lives in flt,
        # while the surrounding framework is single/single with invariant back-out.
        return FW.run_once(flt, sysm, rng, "single", "single", **options)
    if cfg == "IPLF2":
        return iplf_run_once(flt, sysm, rng, IPLF_TOL, IPLF_MAX_ITER, **options)
    raise ValueError(cfg)


def _cell(args):
    key_name, sigma, variant, cfg, M = args
    spec = SPECS[key_name]
    K = list(spec["key"])
    t0 = time.perf_counter()
    sysm = spec["make"](sigma)
    per_run = np.empty(M)
    fires = np.empty(M)
    nees = np.empty((M, sysm.N + 1))
    for m in range(M):
        rng = np.random.default_rng(SEED0 + m)
        flt = make_filter(variant, cfg, sysm)
        errs, Ps, aux, factors = run_configuration(flt, sysm, rng, cfg,
                                                  return_factors=True)
        per_run[m] = per_run_rmse(errs, K, BURN_IN)
        nees[m] = nees_curve(errs, Ps, K, factors=factors)
        fires[m] = aux

    anees_curve = np.mean(nees, axis=0)
    invalid = ~np.isfinite(nees)
    return dict(system=key_name, sigma=sigma, variant=variant, cfg=cfg,
                **summarize_rmse(per_run),
                anees=float(np.mean(anees_curve)), anees_curve=anees_curve,
                per_run=per_run, fire=float(np.mean(fires)), runs=M,
                metrics=SUMMARY_VERSION, covariance_policy=COVARIANCE_POLICY,
                cov_bad_runs=int(np.sum(np.any(invalid, axis=1))),
                cov_bad_frac=float(np.mean(invalid)),
                secs=time.perf_counter() - t0)


# ---------------------------------------------------------------------- main
def run_system(name, M, n_jobs, *, output_dir=".", save_csv=False, save_perrun=False):
    """Run one fixed sweep and write the paper figures.

    By default only the three PNGs are retained.  ``save_csv=True`` writes
    the compact summary CSV, while ``save_perrun=True`` writes the per-run NPZ.
    """
    spec = SPECS[name]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    jobs = [(name, s, v, c, M) for s in spec["noise"]
            for v in VARIANTS for c in configs_for(v)]
    print(f"\n=== {name}: {spec['title']} ===")
    print(f"{len(jobs)} cells x {M} runs, n_jobs={n_jobs}")
    print(f"key states {spec['key']} ({spec['key_lbl']}), burn-in {BURN_IN}",
          flush=True)
    probe_system = spec["make"](spec["noise"][0])
    print(f"label schema {LABELS_VERSION}; measurement evaluations per step "
          f"(n_x = {probe_system.n_x}):")
    print(f"    {'variant':>7s} {'cfg':>6s} {'evals':>6s} {'pts/eval':>9s} "
          f"{'pts/step':>9s} {'vs F4':>7s}")
    budget = budget_table(probe_system)
    f4_budget = {variant: total for variant, cfg, _, _, total in budget
                 if cfg == "F4"}
    for variant, cfg, calls, points, total in budget:
        print(f"    {variant:>7s} {cfg:>6s} {calls:6d} {points:9d} "
              f"{total:9d} {total / f4_budget[variant]:6.0%}")
    print(flush=True)

    t0 = time.perf_counter()
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            output = list(executor.map(_cell, jobs, chunksize=1))
    else:
        output = [_cell(job) for job in jobs]
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    rows = []
    store = {} if save_perrun else None
    for result in output:
        row = {key: result[key] for key in
               ("system", "sigma", "variant", "cfg", "rmse_rms",
                "rmse_med", "rmse_p95", "anees", "fire", "metrics",
                "covariance_policy",
                "runs", "cov_bad_runs",
                "cov_bad_frac", "secs")}
        gain, recal, metric = MANIFEST[result["cfg"]]
        row.update(labels=LABELS_VERSION, gain=gain, recal=recal, metric=metric)
        rows.append(row)
        if store is not None:
            key = f"{result['sigma']:g}|{result['variant']}|{result['cfg']}"
            store[key + "|per_run"] = result["per_run"]
            store[key + "|anees_curve"] = result["anees_curve"]

    if save_csv:
        csv_path = os.path.join(output_dir, f"sweep_{name}_results.csv")
        write_sweep_csv(rows, csv_path)
    if save_perrun:
        store["covariance_policy"] = np.asarray(COVARIANCE_POLICY)
        np.savez_compressed(os.path.join(output_dir, f"sweep_{name}_perrun.npz"),
                            **store)

    print(f"\n{'sigma':>10s} {'variant':>7s} {'cfg':>6s} {'RMSErms':>11s} "
          f"{'RMSEmed':>11s} {'RMSEp95':>11s} {'ANEES':>11s} {'fire':>6s} "
          f"{'covbad':>7s} {'secs':>7s}")
    for row in rows:
        print(f"{row['sigma']:10.4g} {row['variant']:>7s} {row['cfg']:>6s} "
              f"{row['rmse_rms']:11.4g} {row['rmse_med']:11.4g} "
              f"{row['rmse_p95']:11.4g} {row['anees']:11.4g} "
              f"{row['fire']:6.3f} {row['cov_bad_frac']:7.3f} "
              f"{row['secs']:7.1f}")

    make_figures(name, spec, rows, output_dir=output_dir)
    retained = ["three PNG figures"]
    if save_csv:
        retained.append("summary CSV")
    if save_perrun:
        retained.append("per-run NPZ")
    retained = ", ".join(retained)
    print(f"\nwrote {retained} to {output_dir}")
    return rows


# --------------------------------------------------------------------- plots
PANEL = ["(a)", "(b)", "(c)", "(d)"]
# Every fixed-sweep panel contains the six principal frameworks.  Draw IPLF
# first and F4 last so the proposed curve remains visible when curves overlap.
def plot_order_for(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return ["IPLF2", "F0", "F1", "F2", "F3", "F4"]


# Canonical order of the common legend for the fixed-sweep figures.
LEGEND_ORDER = ["F0", "F1", "F2", "F3", "F4", "IPLF2"]
# Figure 3 (median / 95th percentile) doubles every line, so it shows only a
# representative subset: conventional, published, proposed, and IPLF.
TAIL_CONFIGS = ["IPLF2", "F0", "F1", "F4"]


def _rc():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
        "lines.linewidth": 1.6, "lines.markersize": 5.5,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.handlelength": 2.4,
        "legend.columnspacing": 1.4, "legend.handletextpad": 0.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def _panels(shared_y=True):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharex=True,
                             sharey=shared_y, constrained_layout=True)
    return fig, axes


def _finish(fig, axes, xlabel, ylabel, fname, ncol=6):
    """Shared x labels, one y label centred on the axes block, bottom legend."""
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel)

    # Collect unique legend entries from all four panels.
    handles_by_label = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    ordered_labels = [LABEL[c] for c in LEGEND_ORDER
                      if LABEL[c] in handles_by_label]
    h = [handles_by_label[label] for label in ordered_labels]
    l = ordered_labels
    fig.legend(h, l, loc="outside lower center", ncol=ncol)
    # Place the single y label at the vertical centre of the AXES, not of the
    # figure: the bottom legend shifts the axes up, so the figure centre sits
    # visibly below the axes centre.
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(ren).transformed(inv) for ax in axes.ravel()]
    pos = [ax.get_position() for ax in axes.ravel()]
    x0 = min(b.x0 for b in boxes)
    yc = 0.5 * (min(p.y0 for p in pos) + max(p.y1 for p in pos))
    fig.text(x0 - 0.01, yc, ylabel, rotation=90, ha="right", va="center",
             fontsize=plt.rcParams["axes.labelsize"])
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def make_figures(name, spec, rows, *, output_dir="."):
    _rc()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    noise = spec["noise"]
    unit = f" ({spec['unit']})" if spec["unit"] != "-" else ""

    def get(sig, variant, cfg, k):
        for r in rows:
            if (np.isclose(float(r["sigma"]), sig, rtol=1e-12, atol=0.0)
                    and r["variant"] == variant and r["cfg"] == cfg):
                return r[k]
        return np.nan

    def panel_label(ax, a, variant):
        # Put panel labels above the data so they cannot hide low-noise failures.
        ax.set_title(f"{PANEL[a]} {variant}", loc="left", pad=3.0)
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.10)

    # --- figure 1: RMS across the per-run average key-state RMSEs ---------
    fig, axes = _panels(True)
    for a, (ax, v) in enumerate(zip(axes.ravel(), VARIANTS)):
        for c in plot_order_for(v):
            col, mk = STYLE[c]
            ax.plot(noise, [get(s, v, c, "rmse_rms") for s in noise], "-",
                    color=col, marker=mk, markerfacecolor="none", label=LABEL[c])
        ax.set_xscale("log")
        ax.set_yscale("log")
        panel_label(ax, a, v)
    if name == "batt":
        axes[0, 0].set_ylim(top=1e1)
    _finish(fig, axes, spec["xlabel"],
            f"RMS of per-run RMSE{unit}",
            os.path.join(output_dir, f"sweep_{name}_fig1_rmse.png"))

    # --- figure 2: average key-state ANEES over all steps ------------------
    fig, axes = _panels(False)
    for a, (ax, v) in enumerate(zip(axes.ravel(), VARIANTS)):
        ax.axhline(1.0, color="0.55", lw=0.8, alpha=0.35, zorder=0)
        for c in plot_order_for(v):
            col, mk = STYLE[c]
            ax.plot(noise, [get(s, v, c, "anees") for s in noise], "-",
                    color=col, marker=mk, markerfacecolor="none", label=LABEL[c])
        ax.set_xscale("log")
        ax.set_yscale("log")
        panel_label(ax, a, v)
    if name == "batt":
        # The paper caption identifies F0 and IPLF above this display limit.
        axes[0, 0].set_ylim(top=1e10)
    _finish(fig, axes, spec["xlabel"], "Average ANEES (key states)",
            os.path.join(output_dir, f"sweep_{name}_fig2_anees.png"))

    # --- figure 3: median and p95 of the per-run average RMSE --------------
    fig, axes = _panels(True)
    for a, (ax, v) in enumerate(zip(axes.ravel(), VARIANTS)):
        for c in TAIL_CONFIGS:
            col, mk = STYLE[c]
            ax.plot(noise, [get(s, v, c, "rmse_med") for s in noise], "-",
                    color=col, marker=mk, markerfacecolor="none", label=LABEL[c])
            ax.plot(noise, [get(s, v, c, "rmse_p95") for s in noise], ":",
                    color=col, marker=mk, markerfacecolor="none", alpha=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        panel_label(ax, a, v)
    # The solid / dotted convention is stated in the caption, not the legend.
    _finish(fig, axes, spec["xlabel"],
            f"Per-run average {spec['key_lbl']} RMSE{unit}",
            os.path.join(output_dir, f"sweep_{name}_fig3_tail.png"), ncol=4)


def write_sweep_csv(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sweep_csv(name, directory="."):
    """Load current summaries, or recalculate old ones from saved per-run data."""
    path = os.path.join(directory, f"sweep_{name}_results.csv")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get("covariance_policy") != COVARIANCE_POLICY for row in rows):
        raise ValueError(
            f"{path} uses an older covariance implementation. "
            f"Rerun sweep_all.py {name}; previous trajectories cannot be reaggregated."
        )
    numeric = {
        "sigma", *METRIC_FIELDS, "fire",
        "runs", "n_requested", "cov_bad_runs", "cov_bad_frac",
        "secs",
    }
    for row in rows:
        for key in numeric.intersection(row):
            if row[key] != "":
                row[key] = float(row[key])
    if any(row.get("metrics") != SUMMARY_VERSION for row in rows):
        perrun_path = os.path.join(directory, f"sweep_{name}_perrun.npz")
        if not os.path.isfile(perrun_path):
            raise ValueError(
                f"{path} uses older RMSE summaries. Recalculation needs "
                f"{perrun_path}; otherwise rerun sweep_all.py {name}."
            )
        upgraded = []
        with np.load(perrun_path) as saved:
            stored_policy = saved.get("covariance_policy")
            if stored_policy is None or str(np.asarray(stored_policy).item()) != COVARIANCE_POLICY:
                raise ValueError(f"{perrun_path} uses an older covariance policy; rerun this sweep.")
            for row in rows:
                key = f"{row['sigma']:g}|{row['variant']}|{row['cfg']}"
                per_run = saved[key + "|per_run"]
                runs = int(row.get("runs", row.get("n_requested", len(per_run))))
                if len(per_run) != runs:
                    raise ValueError(f"Incomplete per-run data for {key}; rerun this sweep.")
                curve = saved[key + "|anees_curve"]
                # Old curves omitted invalid samples; their presence leaves
                # the all-run ANEES undefined under the current definition.
                anees = (np.nan if row.get("cov_bad_runs", 0) > 0
                         or row.get("cov_bad_frac", 0) > 0 else float(np.mean(curve)))
                current = {k: v for k, v in row.items()
                           if not k.startswith("rmse_")
                           and k not in {"anees_avg", "n_requested", "n_success", "n_bad"}}
                current.update(summarize_rmse(per_run), anees=anees,
                               runs=runs, metrics=SUMMARY_VERSION)
                upgraded.append(current)
        rows = upgraded
    validate_sweep_rows(name, rows)
    return rows


def validate_sweep_rows(name, rows):
    """Reject partial or mislabeled sweeps before plotting publication figures."""
    spec = SPECS[name]
    expected = len(spec["noise"]) * len(VARIANTS) * len(CONFIGS)
    if len(rows) != expected:
        raise ValueError(f"{name} sweep has {len(rows)} rows; expected {expected}")
    if any(row.get("labels") != LABELS_VERSION for row in rows):
        raise ValueError(f"{name} sweep does not use label schema {LABELS_VERSION}")
    if any(row.get("covariance_policy") != COVARIANCE_POLICY for row in rows):
        raise ValueError(f"{name} sweep requires the current square-root implementation")
    if any(row.get("metrics") != SUMMARY_VERSION
           or any(field not in row for field in METRIC_FIELDS) for row in rows):
        raise ValueError(f"{name} sweep requires current RMS/median/p95/ANEES summaries")
    for sigma in spec["noise"]:
        for variant in VARIANTS:
            for cfg in CONFIGS:
                count = sum(
                    np.isclose(float(row["sigma"]), sigma, rtol=1e-12, atol=0.0)
                    and row["variant"] == variant and row["cfg"] == cfg
                    for row in rows
                )
                if count != 1:
                    raise ValueError(
                        f"missing or duplicate sweep row: sigma={sigma:g}, "
                        f"variant={variant}, cfg={cfg} (count={count})"
                    )


def figures_only(name, directory=".", *, output_dir=".", save_csv=False):
    """Regenerate all three figures from a completed, validated results CSV."""
    rows = load_sweep_csv(name, directory)
    os.makedirs(output_dir, exist_ok=True)
    make_figures(name, SPECS[name], rows, output_dir=output_dir)
    if save_csv:
        write_sweep_csv(rows, os.path.join(output_dir, f"sweep_{name}_results.csv"))
    print(f"regenerated sweep_{name}_fig1..fig3 in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", nargs="?", choices=["batt", "trn", "all"],
                        default="trn")
    parser.add_argument("runs", nargs="?", type=int, default=M_DEFAULT)
    parser.add_argument("n_jobs", nargs="?", type=int, default=N_JOBS_DEFAULT)
    parser.add_argument("--figures-only", action="store_true",
                        help="validate an existing CSV and regenerate figures")
    parser.add_argument("--input-dir", default=".",
                        help="directory containing sweep_<system>_results.csv")
    parser.add_argument("--output-dir", default=".",
                        help="directory for generated figures and optional data")
    parser.add_argument("--save-csv", action="store_true",
                        help="retain sweep_<system>_results.csv")
    parser.add_argument("--save-perrun", action="store_true",
                        help="retain sweep_<system>_perrun.npz")
    parser.add_argument("--save-data", action="store_true",
                        help="retain both the summary CSV and per-run NPZ")
    args = parser.parse_args()
    names = ["batt", "trn"] if args.system == "all" else [args.system]
    for name in names:
        if args.figures_only:
            figures_only(name, args.input_dir, output_dir=args.output_dir,
                         save_csv=(args.save_csv or args.save_data))
        else:
            run_system(
                name,
                args.runs,
                args.n_jobs,
                output_dir=args.output_dir,
                save_csv=(args.save_csv or args.save_data),
                save_perrun=(args.save_perrun or args.save_data),
            )


if __name__ == "__main__":
    main()
