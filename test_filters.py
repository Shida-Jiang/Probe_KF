"""Deterministic correctness and paper-consistency checks.

Run before the publication experiments:

    python test_filters.py
"""
from __future__ import annotations

import numpy as np

import filters as FL
import framework as FW
import metrics as MT
import sweep_all as SW
from filters import CKF, EKF, EKF2, EKF3, UKF, sym
from systems import BatteryEstimation, FractalTerrain, hard_terrain_map

rng = np.random.default_rng(0)
ok = True


def check(label: str, value: float, tolerance: float) -> None:
    global ok
    passed = bool(value < tolerance)
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label:62s} "
          f"{value:.3e} < {tolerance:g}")


def finite_difference_jacobian(system, x, k, step=1e-6):
    out = np.zeros((system.n_z, system.n_x))
    for j in range(system.n_x):
        direction = np.zeros(system.n_x)
        direction[j] = step
        out[:, j] = (system.h(x + direction, k)
                     - system.h(x - direction, k)) / (2 * step)
    return out


def finite_difference_hessian(system, x, k, step=2e-5):
    out = np.zeros((system.n_z, system.n_x, system.n_x))
    for j in range(system.n_x):
        direction = np.zeros(system.n_x)
        direction[j] = step
        jp = system.h_jac(x + direction, k)
        jm = system.h_jac(x - direction, k)
        out[:, :, j] = (jp - jm) / (2 * step)
    return out


def finite_difference_third(system, x, k, step=2e-5):
    out = np.zeros((system.n_z, system.n_x, system.n_x, system.n_x))
    for j in range(system.n_x):
        direction = np.zeros(system.n_x)
        direction[j] = step
        hp = np.asarray(system.h_hess(x + direction, k), float)
        hm = np.asarray(system.h_hess(x - direction, k), float)
        out[:, :, :, j] = (hp - hm) / (2 * step)
    return out


angles, phases = hard_terrain_map()
terrain = FractalTerrain(
    1.0, n_oct=4, L0=20.0, A0=200.0, hurst=0.97,
    sig_init=0.45, sig_pro=0.0005, angles=angles, phases=phases,
)
battery = BatteryEstimation(
    sigma_meas=1e-4, rest=15, ramp=30, dt=20.0,
    amp=1.3, x0=[0.35, 0.0, 0.86], sig_soc=0.085,
    sig_uc=0.0085, sig_soh=0.051, sig_pro=[6e-7, 6e-7, 0.0],
)


print("=== 1. analytic derivatives of the two paper systems ===")
for system, x, k, label in (
    (battery, np.array([0.55, 0.01, 0.88]), 20, "battery"),
    (terrain, np.array([5.2, 4.8]), 3, "terrain"),
):
    check(f"{label} measurement Jacobian",
          np.max(np.abs(system.h_jac(x, k)
                        - finite_difference_jacobian(system, x, k))), 2e-5)
    check(f"{label} measurement Hessian",
          np.max(np.abs(np.asarray(system.h_hess(x, k))
                        - finite_difference_hessian(system, x, k))), 5e-3)
    check(f"{label} third measurement derivative",
          np.max(np.abs(np.asarray(system.h_third(x, k))
                        - finite_difference_third(system, x, k))), 5e-3)


print("=== 2. affine exactness of the wrapped moment maps ===")
class LinearSystem:
    n_x, n_z = 3, 2
    F = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]])
    C = np.array([[1.0, -0.4, 0.3], [0.2, 0.5, -0.7]])
    Q = np.diag([0.01, 0.02, 0.03])
    R = np.diag([0.04, 0.07])

    def f(self, x): return self.F @ x
    def F_jac(self, x): return self.F
    def f_hess(self, x): return None
    def f_third(self, x): return None
    def h(self, x, k): return self.C @ x
    def h_jac(self, x, k): return self.C
    def h_hess(self, x, k): return [np.zeros((3, 3)) for _ in range(2)]
    def h_third(self, x, k): return [np.zeros((3, 3, 3)) for _ in range(2)]


linear = LinearSystem()
P = np.array([[1.0, 0.2, -0.1], [0.2, 0.7, 0.05], [-0.1, 0.05, 0.5]])
xi = np.array([0.4, -0.2, 0.8])
reference = EKF(linear).moments(xi, P, 0)
for filter_class in (EKF2, EKF3, UKF, CKF):
    result = filter_class(linear).moments(xi, P, 0)
    discrepancy = max(np.max(np.abs(result[0] - reference[0])),
                      np.max(np.abs(result[1] - reference[1])),
                      np.max(np.abs(result[2] - reference[2])))
    check(f"{filter_class.name} affine measurement moments", discrepancy, 1e-8)

x_ref, P_ref = EKF(linear).predict(xi, P)
for filter_class in (EKF2, EKF3, UKF, CKF):
    x_out, P_out = filter_class(linear).predict(xi, P)
    discrepancy = max(np.max(np.abs(x_out - x_ref)),
                      np.max(np.abs(P_out - P_ref)))
    check(f"{filter_class.name} affine prediction", discrepancy, 1e-8)


print("=== 3. EKF3 cubic Gaussian moments ===")
class CubicMeasurement:
    n_x, n_z = 2, 1
    Q = np.zeros((2, 2))
    R = np.array([[0.07]])

    def f(self, x): return np.asarray(x, float)
    def F_jac(self, x): return np.eye(2)
    def f_hess(self, x): return None
    def f_third(self, x): return None

    def h(self, x, k):
        x1, x2 = x
        return np.array([0.3 + x1 - 0.4*x2 + 0.2*x1*x2
                         + 0.1*x1**2 + 0.15*x2**2
                         + 0.07*x1**3 - 0.05*x1*x2**2])

    def h_jac(self, x, k):
        x1, x2 = x
        return np.array([[1.0 + 0.2*x2 + 0.2*x1 + 0.21*x1**2 - 0.05*x2**2,
                          -0.4 + 0.2*x1 + 0.3*x2 - 0.1*x1*x2]])

    def h_hess(self, x, k):
        x1, x2 = x
        return [np.array([[0.2 + 0.42*x1, 0.2 - 0.1*x2],
                          [0.2 - 0.1*x2, 0.3 - 0.1*x1]])]

    def h_third(self, x, k):
        T = np.zeros((2, 2, 2))
        T[0, 0, 0] = 0.42
        T[0, 1, 1] = T[1, 0, 1] = T[1, 1, 0] = -0.1
        return [T]


cubic = CubicMeasurement()
xc = np.array([0.4, -0.3])
Pc = np.array([[0.8, 0.15], [0.15, 0.5]])
third = EKF3(cubic).moments(xc, Pc, 0)
reference3 = FL.GHKF(cubic, 4).moments(xc, Pc, 0)
discrepancy = max(np.max(np.abs(third[0] - reference3[0])),
                  np.max(np.abs(third[1] - reference3[1])),
                  np.max(np.abs(third[2] - reference3[2])))
check("EKF3 exactness for a cubic Gaussian transform", discrepancy, 1e-11)


print("=== 4. sigma-point, cubature, and probe geometry ===")
for filter_class in (UKF, CKF):
    rule = filter_class(linear)
    X = rule._sigma(xi, P)
    if isinstance(rule, UKF):
        mean = X @ rule.Wm
        D = X - xi[:, None]
        cov = (D * rule.Wc) @ D.T
    else:
        mean = X.mean(axis=1)
        D = X - xi[:, None]
        cov = D @ D.T / X.shape[1]
    check(f"{rule.name} points reproduce the mean", np.max(np.abs(mean - xi)), 1e-8)
    check(f"{rule.name} points reproduce P", np.max(np.abs(cov - P)), 1e-8)

for rule in (FL.GHKF(battery, 3), FL.CKF5(battery)):
    X0, W = rule.X0, rule.W
    check(f"{rule.name} points reproduce zero mean", np.max(np.abs(X0 @ W)), 1e-12)
    check(f"{rule.name} points reproduce unit covariance",
          np.max(np.abs((X0 * W) @ X0.T - np.eye(battery.n_x))), 1e-12)
    fourth = max(abs(np.sum(W * X0[j] ** 4) - 3.0)
                 for j in range(battery.n_x))
    cross_fourth = abs(np.sum(W * X0[0] ** 2 * X0[1] ** 2) - 1.0)
    check(f"{rule.name} points reproduce Gaussian fourth moments",
          max(fourth, cross_fourth), 1e-12)

points, weights = FW.probe_set(battery.x0, battery.P0, "set")
offsets = np.stack([point - battery.x0 for point in points])
probe_cov = np.einsum("i,ij,ik->jk", weights, offsets, offsets)
check("probe set has n+1 points", abs(len(points) - (battery.n_x + 1)), 0.5)
check("probe weights are equal", np.max(np.abs(weights - 1 / (battery.n_x + 1))), 1e-14)
check("probe offsets are centered", np.max(np.abs(weights @ offsets)), 1e-12)
check("probe offsets reproduce P", np.max(np.abs(probe_cov - battery.P0)), 1e-12)


print("=== 5. generalized Joseph and probe-mean identities ===")
battery._k = 30
flt = CKF(battery)
points, weights = FW.probe_set(battery.x0, battery.P0, "set")
moments = [flt.moments(point, battery.P0, 31) for point in points]
cross = np.stack([moment[1] for moment in moments])
innovation = np.stack([moment[2] for moment in moments])
cross_mean = np.einsum("i,ipq->pq", weights, cross)
innovation_mean = np.einsum("i,ipq->pq", weights, innovation)
gain_mean = cross_mean @ np.linalg.inv(innovation_mean)
mean_report = FW.generalized_joseph(
    battery.P0, cross_mean, innovation_mean, gain_mean)
reports = [FW.generalized_joseph(battery.P0, cross[i], innovation[i], gain_mean)
           for i in range(len(points))]
check("average pair equals average generalized Joseph report",
      np.max(np.abs(mean_report - np.mean(reports, axis=0))), 1e-12)

perturb_rng = np.random.default_rng(11)
base_trace = np.trace(mean_report)
minimum_gap = min(
    np.trace(FW.generalized_joseph(
        battery.P0, cross_mean, innovation_mean,
        gain_mean + 1e-3 * perturb_rng.standard_normal(gain_mean.shape)))
    - base_trace
    for _ in range(40)
)
check("probe-mean gain minimizes the average report", max(0.0, -minimum_gap), 1e-14)

for filter_class in (EKF, EKF2, EKF3, UKF, CKF):
    zhat, C, S = filter_class(battery).moments(battery.x0, battery.P0, 20)
    K = C @ np.linalg.inv(S)
    report = FW.generalized_joseph(battery.P0, C, S, K)
    minimum = sym(battery.P0 - K @ S @ K.T)
    check(f"{filter_class.name} generalized Joseph minimum",
          np.max(np.abs(report - minimum)), 1e-10)


print("=== 6. admissibility and coordinate invariance ===")
for system, center, k, label in (
    (battery, battery.x0, 20, "battery"),
    (terrain, terrain.x0, 3, "terrain"),
):
    probe_centers, _ = FW.probe_set(center, system.P0, "set")
    for filter_class in (EKF, EKF2, UKF, CKF):
        min_eig = np.inf
        for point in [center] + probe_centers:
            _, C, S = filter_class(system).moments(point, system.P0, k)
            block = np.block([[system.P0, C], [C.T, S]])
            min_eig = min(min_eig, np.linalg.eigvalsh(sym(block)).min())
        check(f"{label} {filter_class.name} joint moment block is PSD",
              max(0.0, -min_eig), 1e-9)

n, p = 3, 1
Tcoord = rng.standard_normal((n, n)) + 2.0 * np.eye(n)
Ptest = np.diag([0.1, 0.02, 0.05]) ** 2
Ctest = rng.standard_normal((n, p)) * 0.01
Stest = np.array([[0.2]])
Ktest = rng.standard_normal((n, p)) * 0.01
report = FW.generalized_joseph(Ptest, Ctest, Stest, Ktest)
Pprime = Tcoord @ Ptest @ Tcoord.T
report_prime = Tcoord @ report @ Tcoord.T
score = np.trace(np.linalg.inv(Ptest) @ report)
score_prime = np.trace(np.linalg.inv(Pprime) @ report_prime)
check("prediction-normalized trace is coordinate invariant",
      abs(score - score_prime), 1e-10)


print("=== 7. higher-order controls change measurement moments only ===")
for base_class, control in (
    (EKF, FL.ekf2_control(battery)),
    (EKF2, FL.ekf3_control(battery)),
    (UKF, FL.gh_control(battery, 3)),
    (CKF, FL.ckf5_control(battery)),
):
    x_base, P_base = base_class(battery).predict(battery.x0, battery.P0)
    x_control, P_control = control.predict(battery.x0, battery.P0)
    discrepancy = max(np.max(np.abs(x_base - x_control)),
                      np.max(np.abs(P_base - P_control)))
    check(f"{control.name} retains the lower-order prediction", discrepancy, 1e-14)


print("=== 8. center-predicted measurement and paired randomness ===")
class OneStepSystem:
    n_x = n_z = N = 1
    P0 = np.array([[1.0]])
    x0 = np.array([0.0])
    x_true = np.zeros((1, 2))

    def sample_truth(self, random): self.x_true[:] = 0.0
    def draw_init(self, random): return self.x0.copy()
    def measure(self, k, random): return np.array([0.5])


class CenterMeanRecorder:
    def __init__(self, system): self.s = system
    def predict_sqrt(self, x, L): return x.copy(), L.copy()
    def moments_sqrt(self, xi, L, k):
        return np.array([xi[0] ** 2]), np.array([[0.2]]), np.array([[np.sqrt(0.96)]])


one = OneStepSystem()
errors, _, _ = FW.run_once(
    CenterMeanRecorder(one), one, np.random.default_rng(1), "mean", "single")
check("probe-mean update retains the center predicted measurement",
      abs(errors[0, 1] - 0.1), 1e-14)

Psingular = np.diag([1.0, 0.0])
Lsingular = FL.noise_factor(Psingular)
check("process-noise factor preserves an exact zero noise mode",
      np.max(np.abs(Lsingular @ Lsingular.T - Psingular)), 1e-14)


class TraceDecisionSystem:
    """One step with a known innovation and no sampling variation."""
    n_x = n_z = 2
    N = 1

    def __init__(self, transform):
        self.P0 = transform @ np.diag([2.0, 1.0]) @ transform.T
        self.x0 = np.zeros(2)
        self.x_true = np.zeros((2, 2))

    def sample_truth(self, random): self.x_true[:] = 0.0
    def draw_init(self, random): return self.x0.copy()
    def measure(self, k, random): return np.array([1.0, -0.5])


class TraceDecisionFilter:
    """Admissible moment pairs yielding prescribed positive covariance reports."""
    def __init__(self, system, candidate, transform):
        self.s = system
        self.candidate = candidate
        self.transform = transform
        self.calls = 0

    def predict_sqrt(self, x, L): return x.copy(), L.copy()

    def moments(self, xi, P, k):
        self.calls += 1
        if self.calls == 1:
            cross = np.eye(2)
        else:
            cross = 0.5 * (np.diag([2.0, 1.0]) + np.eye(2)
                           - self.candidate)
        self.base_cross = cross
        return np.zeros(2), self.transform @ cross, np.eye(2)

    def moments_sqrt(self, xi, L, k):
        zhat, cross, innovation = self.moments(xi, L @ L.T, k)
        B = np.linalg.solve(L, cross).T
        residual = innovation - self.base_cross.T @ np.diag([0.5, 1.0]) @ self.base_cross
        return zhat, B, FL.noise_factor(residual)


trace_cases = (
    ("positive report below threshold", np.diag([1.0, 0.5]), False),
    ("positive report above threshold", np.diag([4.0, 2.0]), True),
    ("report exactly at threshold", np.diag([2.0, 1.0]), False),
)
for metric in ("I", "Pinv"):
    for label, candidate, rejected in trace_cases:
        trace_system = TraceDecisionSystem(np.eye(2))
        trace_filter = TraceDecisionFilter(trace_system, candidate, np.eye(2))
        errors, covariances, rate = FW.run_once(
            trace_filter, trace_system, np.random.default_rng(2),
            "single", "single", metric=metric)
        check(f"{metric}: {label}", abs(rate - float(rejected)), 1e-14)
        expected_state = np.zeros(2) if rejected else np.array([1.0, -0.5])
        expected_covariance = trace_system.P0 if rejected else candidate
        check(f"{metric}: decision retains the matching state and covariance",
              max(np.max(np.abs(errors[:, 1] - expected_state)),
                  np.max(np.abs(covariances[1] - expected_covariance))), 1e-14)

# Test the actual back-out decision under a nonsingular coordinate change,
# in addition to the algebraic trace identity tested above.
transform = np.array([[2.0, 0.5], [0.0, 0.25]])
for label, candidate, rejected in trace_cases[:2]:
    trace_system = TraceDecisionSystem(transform)
    trace_filter = TraceDecisionFilter(trace_system, candidate, transform)
    errors, covariances, rate = FW.run_once(
        trace_filter, trace_system, np.random.default_rng(2),
        "single", "single", metric="Pinv")
    expected_state = np.zeros(2) if rejected else transform @ np.array([1.0, -0.5])
    expected_covariance = (trace_system.P0 if rejected
                           else transform @ candidate @ transform.T)
    check(f"Pinv decision is invariant: {label}",
          max(abs(rate - float(rejected)),
              np.max(np.abs(errors[:, 1] - expected_state)),
              np.max(np.abs(covariances[1] - expected_covariance))), 1e-12)


class Recorder:
    def __init__(self, system):
        object.__setattr__(self, "_system", system)
        object.__setattr__(self, "log", {})
    def __getattr__(self, name): return getattr(self._system, name)
    def __setattr__(self, name, value): setattr(self._system, name, value)
    def sample_truth(self, random):
        self._system.sample_truth(random)
        self.log["truth"] = self._system.x_true.copy()
        self.log["measurements"] = []
    def measure(self, k, random):
        value = self._system.measure(k, random)
        self.log["measurements"].append(value.copy())
        return value


logs = []
for cfg in SW.CONFIGS:
    recorded = Recorder(BatteryEstimation(
        sigma_meas=1e-4, rest=15, ramp=30,
        x0=[0.45, 0.0, 0.90], dt=20.0))
    SW.run_configuration(SW.make_filter("CKF", cfg, recorded), recorded,
            np.random.default_rng(1005), cfg)
    logs.append(recorded.log)
check("all frameworks share the true trajectory",
      max(np.max(np.abs(logs[0]["truth"] - log["truth"])) for log in logs[1:]), 1e-15)
check("all frameworks share the measurement stream",
      max(np.max(np.abs(np.array(logs[0]["measurements"])
                        - np.array(log["measurements"]))) for log in logs[1:]), 1e-15)


print("=== 9. paper configurations, point counts, and system parameters ===")
expected_manifest = {
    "F0": ("conventional", "none", "none"),
    "F1": ("single", "single", "I"),
    "F2": ("single", "set", "Pinv"),
    "F3": ("mean", "single", "Pinv"),
    "F4": ("mean", "set", "Pinv"),
    "IPLF2": ("iplf", "none", "none"),
}
check("F0--F4/IPLF manifest matches Table 1",
      0.0 if all(SW.MANIFEST[k] == v for k, v in expected_manifest.items()) else 1.0, 0.5)

expected_calls = {"F0": 1, "F1": 2, "F2": 5, "F3": 6, "F4": 9}
for cfg, calls in expected_calls.items():
    check(f"battery {cfg} moment-call count", abs(SW.moment_calls_per_step(battery, cfg) - calls), 0.5)
check("battery UKF+F4 uses 63 measurement evaluations",
      abs(SW.moment_calls_per_step(battery, "F4") * UKF(battery).n_points - 63), 0.5)
check("terrain UKF+F4 uses 35 measurement evaluations",
      abs(SW.moment_calls_per_step(terrain, "F4") * UKF(terrain).n_points - 35), 0.5)
check("battery CKF+F4 uses 54 measurement evaluations",
      abs(SW.moment_calls_per_step(battery, "F4") * CKF(battery).n_points - 54), 0.5)
check("terrain CKF+F4 uses 28 measurement evaluations",
      abs(SW.moment_calls_per_step(terrain, "F4") * CKF(terrain).n_points - 28), 0.5)

check("fixed battery step size is 20 s", abs(battery.dt - 20.0), 1e-15)
check("fixed battery true initial state", np.max(np.abs(battery.x0 - [0.35, 0.0, 0.86])), 1e-15)
check("fixed battery P0", np.max(np.abs(np.diag(battery.P0) - np.array([0.085, 0.0085, 0.051])**2)), 1e-15)
check("fixed battery Q", np.max(np.abs(np.diag(battery.Q) - [3.6e-13, 3.6e-13, 0.0])), 1e-25)
check("fixed battery current reaches -1.3 A", abs(battery.I[-1] + 1.3), 1e-15)
check("fixed battery nominal final SOC is near 10%", abs(battery.x_true[0, -1] - 0.10), 0.015)
check("SOH process-noise variance is exactly zero", abs(battery.Q[2, 2]), 1e-30)

check("fixed terrain initial covariance", np.max(np.abs(terrain.P0 - 0.2025*np.eye(2))), 1e-15)
check("fixed terrain process covariance", np.max(np.abs(terrain.Q - 2.5e-7*np.eye(2))), 1e-18)
check("fixed terrain angles", np.max(np.abs(terrain.angles - angles)), 1e-15)
check("fixed terrain phases", np.max(np.abs(terrain.phi - phases)), 1e-15)
default_battery = BatteryEstimation(1e-4)
default_terrain = FractalTerrain(1.0)
check("BatteryEstimation defaults reproduce the fixed paper case",
      max(np.max(np.abs(default_battery.x0 - battery.x0)),
          np.max(np.abs(default_battery.P0 - battery.P0)),
          np.max(np.abs(default_battery.Q - battery.Q)),
          np.max(np.abs(default_battery.I - battery.I))), 1e-15)
check("FractalTerrain defaults reproduce the fixed paper case",
      max(np.max(np.abs(default_terrain.P0 - terrain.P0)),
          np.max(np.abs(default_terrain.Q - terrain.Q)),
          np.max(np.abs(default_terrain.angles - terrain.angles)),
          np.max(np.abs(default_terrain.phi - terrain.phi))), 1e-15)
check("terrain sweep range is 1--1000 m",
      abs(SW.SPECS["trn"]["noise"][0] - 1.0)
      + abs(SW.SPECS["trn"]["noise"][-1] - 1000.0), 1e-12)

for variant, control in SW.ORDER_CONTROL_FOR.items():
    check(f"{variant} fixed sweep excludes {SW.LABEL[control]}",
          0.0 if control not in SW.configs_for(variant) else 1.0, 0.5)
    check(f"{variant} runtime includes {SW.LABEL[control]}",
          0.0 if SW.runtime_configs_for(variant)[-1] == control else 1.0, 0.5)


print("=== 10. end-to-end F0--F4 smoke tests ===")
for cfg in ("F0", "F1", "F2", "F3", "F4"):
    errs, covariances, rate = SW.run_configuration(
        CKF(battery), battery, np.random.default_rng(42), cfg)
    check(f"{cfg} trajectory is finite",
          0.0 if np.all(np.isfinite(errs)) else 1.0, 0.5)
    check(f"{cfg} reported covariances are finite and symmetric",
          max(0.0 if np.all(np.isfinite(covariances)) else 1.0,
              np.max(np.abs(covariances - covariances.transpose(0, 2, 1)))), 1e-12)
    check(f"{cfg} auxiliary rate is finite", 0.0 if np.isfinite(rate) else 1.0, 0.5)

print("=== 11. shared experiment metrics ===")
metric_runs = [np.array([[0., 3., 3.], [0., 0., 0.]]),
               np.array([[0., 0., 0.], [0., 4., 4.]])]
run_rmse = [MT.per_run_rmse(error, (0, 1), burn_in=0.)
            for error in metric_runs]
check("per-run RMSE averages the two key-state RMSEs",
      np.max(np.abs(np.asarray(run_rmse) - [1.5, 2.])), 1e-14)
summaries = MT.summarize_rmse(run_rmse)
for field, expected in (("rmse_rms", np.sqrt(25. / 8.)),
                        ("rmse_med", 1.75), ("rmse_p95", 1.975)):
    check(f"{field} is calculated after averaging the key states",
          abs(summaries[field] - expected), 1e-14)
metric_error = np.array([[0., 1., 1.], [0., 2., 2.]])
metric_covariance = np.repeat(np.diag([1., 4.])[None], 3, axis=0)
check("ANEES includes initialization and divides by the key-state count",
      abs(MT.anees_of(metric_error, metric_covariance, (0, 1)) - 2. / 3.), 1e-14)
check("setup values use a geometric mean",
      abs(MT.geometric_mean([1., 4., 16.]) - 4.), 1e-14)

print("=== 12. numerical covariance accuracy ===")
strong_prior = np.eye(2)
strong_model = (np.zeros(1), np.array([[1e16, 0.0]]), np.ones((1, 1)))
strong_gain = FW.gain_from_factors(strong_prior, [strong_model], [1.0])
strong_factor = FW.joseph_factor(strong_prior, strong_gain, [strong_model], [1.0])
strong_covariance = strong_factor @ strong_factor.T
check("high-information update retains its analytical tiny variance",
      abs(strong_covariance[0, 0] / 1e-32 - 1.0), 1e-14)
check("high-information update retains the unobserved variance",
      abs(strong_covariance[1, 1] - 1.0), 1e-14)

shared_models = [strong_model] * 3
shared_weights = np.full(3, 1.0 / 3.0)
shared_gain = FW.gain_from_factors(strong_prior, shared_models, shared_weights)
shared_factor = FW.joseph_factor(strong_prior, shared_gain,
                                 shared_models, shared_weights)
check("identical probes retain the analytical high-information variance",
      abs((shared_factor @ shared_factor.T)[0, 0] / 1e-32 - 1.0), 1e-14)

tiny_system = SW.SPECS["batt"]["make"](1e-6)
for filter_class in (UKF, CKF):
    _, tiny_factor = filter_class(tiny_system).predict_sqrt(
        tiny_system.x0, np.eye(3) * 1e-20)
    check(f"{filter_class.name} preserves tiny SOH variance during prediction",
          abs(np.sum(tiny_factor[2] ** 2) / 1e-40 - 1.0), 1e-14)

marginal_factor = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 1e-12]])
marginal_error = np.array([[0.], [0.], [1e-12]])
marginal_nees = MT.nees_curve(
    marginal_error, np.array([marginal_factor @ marginal_factor.T]), (0, 2),
    factors=marginal_factor[None, :, :])
check("marginal NEES retains variance lost in a reconstructed matrix",
      abs(marginal_nees[0] - 0.5), 1e-14)

for cfg, seed in (("F0", 1115), ("IPLF2", 1029)):
    stable_system = SW.SPECS["batt"]["make"](1e-6)
    stable_errors, stable_reports, _, stable_factors = SW.run_configuration(
        EKF(stable_system), stable_system, np.random.default_rng(seed), cfg,
        return_factors=True)
    stable_nees = MT.nees_curve(stable_errors, stable_reports, (0, 2),
                                factors=stable_factors)
    check(f"{cfg} formerly failing run has finite NEES at every step",
          float(not np.all(np.isfinite(stable_nees))), 0.5)
    check(f"{cfg} covariance factors retain positive pivots",
          float(not np.all(np.diagonal(stable_factors, axis1=1, axis2=2) > 0)),
          0.5)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
