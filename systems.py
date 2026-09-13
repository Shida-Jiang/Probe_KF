"""Benchmark systems for the probe-set nonlinear Kalman filter study.

Each system exposes the same interface:

    n_x, n_z, N          dimensions and number of measurement updates
    x0, P0               true initial state; covariance of the initial estimate
    Q, R                 process / measurement covariances used by the filter
    x_true               (n_x, N+1) true trajectory, one column per time point
    sample_truth(rng)    redraw x_true with process noise for one Monte Carlo run
    draw_init(rng)       draw the initial estimate used in one run
    measure(k, rng)      noisy measurement of the true state at step k
    f, F_jac, f_hess              dynamics and derivatives
    h, h_jac, h_hess, h_third     measurement map and derivatives

Per-run randomness is consumed in a fixed order: sample_truth, then draw_init,
then one measurement draw per step.  Every framework follows the same order, so
runs with the same seed are driven by identical realisations and all
comparisons are paired.
"""
import numpy as np


# The map is part of the system specification, like L0 or A0.  Fixing it here
# means seed=None still gives a reproducible terrain. Pass `angles` and
# `phases` to FractalTerrain to specify another map outright; see
# its docstring for why the octave directions are the knob that decides how
# ambiguous the map is.
DEFAULT_TERRAIN_SEED = 11

# Hard but physically reasonable map used by the revised measurement-noise
# sweep.  It is based on setup 190 of the independent randomized study.  The
# directions and phases are rounded to 0.1 degree so the map can be stated
# directly in the paper and reproduced without a hidden random seed.
HARD_TERRAIN_ANGLES_DEG = (238.4, 42.1, 288.0, 112.5)
HARD_TERRAIN_PHASES_DEG = (289.4, 313.3, 92.0, 197.3)


def hard_terrain_map():
    """Directions and phases of the revised hard terrain sweep, in radians."""
    return (np.deg2rad(np.array(HARD_TERRAIN_ANGLES_DEG)),
            np.deg2rad(np.array(HARD_TERRAIN_PHASES_DEG)))


class FractalTerrain:
    """Terrain-referenced navigation over a multi-scale (fractal) map,

        h(x) = sum_j A_j sin(k_j . x + phi_j),  |k_j| = 2 pi 2^j / L0,
        A_j = A0 2^(-j H),

    so curvature is present at every length scale.  Positions are in km and
    elevations in m.
    """

    name = "fractal_terrain"
    n_x, n_z = 2, 1

    def __init__(self, sigma_meas, n_oct=4, L0=20.0, A0=200.0, hurst=0.97,
                 speed=0.25, N=60, sig_init=0.45, sig_pro=5e-4, seed=None,
                 angles=None, phases=None):
        """The map is fixed by the octave AMPLITUDES and WAVELENGTHS, which
        follow from A0, L0 and hurst, and by the octave DIRECTIONS and PHASES,
        which are the only random part.  Pass `angles` and `phases` to state
        them directly; the map is then fully specified by its arguments and
        `seed` is unused.

        The two properties that decide how hard the problem is separate
        cleanly, and only the first is affected by the directions:

          ambiguity.  If the octave directions cluster, h varies mostly along
            one projected coordinate: the map becomes a set of near-parallel
            ridges, position along a ridge is weakly observable, and a precise
            altimeter can lock onto the wrong one.  Four independent uniform
            draws cluster by chance often enough that some seeds are much
            harder than others.  Angles spread evenly over [0, pi) make every
            location distinctive in two dimensions and remove the degeneracy
            by construction; note that k and -k give the same family of
            surfaces, so [0, pi) already covers every distinct orientation.

          curvature.  |d2h| ~ A0 / lambda^2 per octave, which is what the
            probe sets exploit.  It is set by A0, L0 and hurst and is
            untouched by the directions, so relief can be raised to sharpen
            the nonlinearity without making the map more ambiguous.

        `seed=None` falls back to the default map rather than drawing fresh
        entropy, which keeps runs reproducible and paired.

        Example of an evenly spread, fully specified map:

            FractalTerrain(sig, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           phases=[0, np.pi/2, np.pi, 3*np.pi/2])
        """
        self.N = int(N)
        if angles is None and phases is None and seed is None:
            angles, phases = hard_terrain_map()
        rng = np.random.default_rng(DEFAULT_TERRAIN_SEED if seed is None
                                    else seed)
        self.K = np.zeros((n_oct, 2))
        self.A = np.zeros(n_oct)
        self.phi = (rng.uniform(0, 2 * np.pi, n_oct) if phases is None
                    else np.asarray(phases, float))
        self.angles = (None if angles is None
                       else np.asarray(angles, float))
        if self.phi.size != n_oct or (self.angles is not None
                                      and self.angles.size != n_oct):
            raise ValueError("angles and phases must have n_oct entries")
        for j in range(n_oct):
            ang = (rng.uniform(0, 2 * np.pi) if self.angles is None
                   else float(self.angles[j]))
            kmag = 2 * np.pi / (L0 * 2.0 ** (-j))       # wavelength L0 / 2^j
            self.K[j] = kmag * np.array([np.cos(ang), np.sin(ang)])
            self.A[j] = A0 * 2.0 ** (-j * hurst)
        if self.angles is None:
            self.angles = np.arctan2(self.K[:, 1], self.K[:, 0])
        self.F = np.eye(2)
        self.U_CTRL = np.array([speed, 0.6 * speed])
        self.sig_pro = np.full(2, float(sig_pro))
        self.Q = np.diag(self.sig_pro ** 2)
        self.sig_meas = np.array([float(sigma_meas)])
        self.R = np.diag(self.sig_meas ** 2)
        self.sig_init = np.full(2, float(sig_init))
        self.P0 = np.diag(self.sig_init ** 2)
        self.x0 = np.array([5.0, 5.0])
        # Nominal trajectory; sample_truth() redraws it with process noise.
        self.x_true = np.zeros((2, self.N + 1))
        self.x_true[:, 0] = self.x0
        for k in range(1, self.N + 1):
            self.x_true[:, k] = self.F @ self.x_true[:, k - 1] + self.U_CTRL

    def f(self, x):
        return self.F @ x + self.U_CTRL

    def f_increment(self, x, delta):
        """Exact transition displacement without subtracting nearby states."""
        return self.F @ delta

    def F_jac(self, x):
        return self.F

    def f_hess(self, x):
        return None

    def sample_truth(self, rng):
        """Redraw the true trajectory with process noise."""
        xt = self.x_true
        xt[:, 0] = self.x0
        for k in range(1, self.N + 1):
            w = self.sig_pro * rng.standard_normal(2)
            xt[:, k] = self.F @ xt[:, k - 1] + self.U_CTRL + w

    def measure(self, k, rng):
        return (self.h(self.x_true[:, k], k)
                + rng.standard_normal(self.n_z) * self.sig_meas)

    def h(self, x, k):
        return np.array([float(np.sum(self.A * np.sin(self.K @ x + self.phi)))])

    def draw_init(self, rng):
        return self.x0 + rng.standard_normal(self.n_x) * self.sig_init

    def h_jac(self, x, k):
        c = self.A * np.cos(self.K @ x + self.phi)
        return (c[:, None] * self.K).sum(axis=0)[None, :]

    def h_hess(self, x, k):
        s_ = self.A * np.sin(self.K @ x + self.phi)
        H = -np.einsum('j,ja,jb->ab', s_, self.K, self.K)
        return [H]

    def h_third(self, x, k):
        c_ = self.A * np.cos(self.K @ x + self.phi)
        T = -np.einsum('j,ja,jb,jc->abc', c_, self.K, self.K, self.K)
        return [T]


# --- OCV coefficients (highest order first), from the Automatica manuscript ---
_A100 = np.array([1390.38, -6961.31, 14760.31, -17230.92, 12055.71,
                  -5162.75, 1330.60, -196.37, 15.60, 2.96])
_A80 = np.array([813.94, -4229.96, 9345.49, -11415.38, 8396.15,
                 -3801.07, 1043.09, -165.29, 14.28, 2.96])
# polyder is LINEAR in the coefficients, so every derivative basis is fixed:
_D100, _D80 = np.polyder(_A100), np.polyder(_A80)
_DD100, _DD80 = np.polyder(_D100), np.polyder(_D80)
_DDD100, _DDD80 = np.polyder(_DD100), np.polyder(_DD80)
_CD = (_A100 - _A80) / 0.2              # d(coef)/d(SOH)
_DCD = np.polyder(_CD)
_DDCD = np.polyder(_DCD)


class BatteryEstimation:
    """Battery SOC / SOH estimation.

        x = [SOC, Uc, SOH]
        SOC_{k+1} = SOC_k + I dt / (3600 Q0 SOH_k)
        Uc_{k+1}  = a Uc_k + (1 - a) R2 I,        a = exp(-dt / (R2 C1))
        SOH_{k+1} = SOH_k
        z         = OCV(SOC, SOH) + Uc + R1 I

    The current profile is deterministic and known to the filter.  The true
    trajectory carries additive process noise w_k ~ N(0, Q) on the SOC and
    RC-branch voltage.  The fixed paper case uses a standard deviation of
    6e-7 per step, while the randomized study scales a 2e-6 base value,
    representing unmodelled coulomb-counting and relaxation losses, and the
    measurement carries an independent voltmeter noise v_k ~ N(0, R).  The two
    are drawn separately, so E[w_k v_k] = 0 as every filter here assumes.

    The current profile rests, then ramps linearly into a discharge, so the cell
    parameters become observable only as the current builds.  SOC and SOH are
    clipped into (0, 1) at initialisation only.
    """

    name = "battery"
    n_x, n_z = 3, 1
    _k = 0          # current step, set by the filter loop

    def __init__(self, sigma_meas=1e-4, N=60, dt=20.0, amp=1.3, Q0=1.0,
                 rest=15, ramp=30,
                 R1=0.01, R2=0.05, inv_R2C1=0.008,
                 sig_pro=(6e-7, 6e-7, 0.0),
                 sig_soc=0.085, sig_uc=0.0085, sig_soh=0.051,
                 x0=(0.35, 0.0, 0.86), lo=0.02, hi=0.98):
        self.N, self.dt, self.Q0, self.R1, self.R2 = int(N), float(dt), Q0, R1, R2
        self.a = np.exp(-self.dt * inv_R2C1)
        self.lo, self.hi = lo, hi
        self.rest, self.ramp = int(rest), int(ramp)
        self.I = np.zeros(self.N + 1)
        # I_k = 0 for k <= rest, then ramps linearly to -amp over ``ramp``
        # steps and remains at -amp.  This is the only current profile used in
        # the paper and is now the class default.
        for k in range(self.N + 1):
            if k <= rest:
                self.I[k] = 0.0
            elif k <= rest + ramp:
                self.I[k] = -amp * (k - rest) / ramp
            else:
                self.I[k] = -amp
        # Process noise on [SOC, Uc, SOH].  SOH is a constant parameter and
        # carries none.  The paper's fixed case uses 6e-7 per step on SOC and
        # the RC-branch voltage.  Randomized studies pass their sampled scale
        # explicitly.
        self.sig_pro = np.asarray(sig_pro, float)
        self.Q = np.diag(self.sig_pro ** 2)
        self.sig_meas = np.array([float(sigma_meas)])
        self.R = np.diag(self.sig_meas ** 2)
        self.sig_init = np.array([sig_soc, sig_uc, sig_soh])
        self.P0 = np.diag(self.sig_init ** 2)
        self.x0 = np.asarray(x0, float)
        # Nominal trajectory; sample_truth() redraws it with process noise.
        self.x_true = np.zeros((3, self.N + 1))
        self.x_true[:, 0] = self.x0
        for k in range(1, self.N + 1):
            self.x_true[:, k] = self._step(self.x_true[:, k - 1],
                                           self.I[k - 1])

    # ---------------------------------------------------------------- OCV
    def _coef(self, soh):
        w = (soh - 0.8) / 0.2
        return w * _A100 + (1.0 - w) * _A80

    def _ocv(self, soc, soh):
        return float(np.polyval(self._coef(soh), soc))

    # ---------------------------------------------------------------- dynamics
    def _step(self, x, I):
        soh = x[2] if abs(x[2]) > 1e-6 else 1e-6
        return np.array([x[0] + I * self.dt / (3600.0 * self.Q0 * soh),
                         self.a * x[1] + (1 - self.a) * self.R2 * I,
                         x[2]])

    def f(self, x):
        return self._step(x, self.I[min(self._k, self.N)])

    def f_increment(self, x, delta):
        """Exact transition displacement without subtracting nearby states."""
        soh0, soh1 = x[2], x[2] + delta[2]
        free0, free1 = abs(soh0) > 1e-6, abs(soh1) > 1e-6
        denominator0 = soh0 if free0 else 1e-6
        denominator1 = soh1 if free1 else 1e-6
        if free0 and free1:
            reciprocal_change = -delta[2] / (denominator0 * denominator1)
        else:
            reciprocal_change = ((denominator0 - denominator1)
                                 / (denominator0 * denominator1))
        current = self.I[min(self._k, self.N)]
        scale = current * self.dt / (3600.0 * self.Q0)
        return np.array([delta[0] + scale * reciprocal_change,
                         self.a * delta[1], delta[2]])

    def F_jac(self, x):
        I = self.I[min(self._k, self.N)]
        soh = x[2] if abs(x[2]) > 1e-6 else 1e-6
        F = np.eye(3)
        F[0, 2] = -I * self.dt / (3600.0 * self.Q0 * soh ** 2)
        F[1, 1] = self.a
        return F

    def f_hess(self, x):
        I = self.I[min(self._k, self.N)]
        soh = x[2] if abs(x[2]) > 1e-6 else 1e-6
        out = [np.zeros((3, 3)) for _ in range(3)]
        out[0][2, 2] = 2.0 * I * self.dt / (3600.0 * self.Q0 * soh ** 3)
        return out

    def sample_truth(self, rng):
        """Redraw the true trajectory with independent process noise."""
        xt = self.x_true
        xt[:, 0] = self.x0
        for k in range(1, self.N + 1):
            w = self.sig_pro * rng.standard_normal(3)
            xt[:, k] = self._step(xt[:, k - 1], self.I[k - 1]) + w

    def draw_init(self, rng):
        x = self.x0 + rng.standard_normal(3) * self.sig_init
        x[0] = np.clip(x[0], self.lo, self.hi)
        x[2] = np.clip(x[2], self.lo, self.hi)
        return x

    def measure(self, k, rng):
        """True terminal voltage plus independent voltmeter noise."""
        xt = self.x_true[:, k]
        z = self._ocv(xt[0], xt[2]) + xt[1] + self.R1 * self.I[k]
        return np.array([z]) + rng.standard_normal(1) * self.sig_meas

    # ------------------------------------------------------------- measurement
    def h(self, x, k):
        return np.array([self._ocv(x[0], x[2]) + x[1]
                         + self.R1 * self.I[min(k, self.N)]])

    def h_jac(self, x, k):
        w = (x[2] - 0.8) / 0.2
        d = w * _D100 + (1.0 - w) * _D80
        return np.array([[float(np.polyval(d, x[0])), 1.0,
                          float(np.polyval(_CD, x[0]))]])

    def h_hess(self, x, k):
        w = (x[2] - 0.8) / 0.2
        dd = w * _DD100 + (1.0 - w) * _DD80
        dss = float(np.polyval(dd, x[0]))
        dsh = float(np.polyval(_DCD, x[0]))
        H = np.array([[dss, 0.0, dsh], [0.0, 0.0, 0.0], [dsh, 0.0, 0.0]])
        return [H]

    def h_third(self, x, k):
        w = (x[2] - 0.8) / 0.2
        ddd = w * _DDD100 + (1.0 - w) * _DDD80
        dsss = float(np.polyval(ddd, x[0]))
        dssh = float(np.polyval(_DDCD, x[0]))
        T = np.zeros((3, 3, 3))
        T[0, 0, 0] = dsss
        T[0, 0, 2] = dssh
        T[0, 2, 0] = dssh
        T[2, 0, 0] = dssh
        return [T]
