"""Square-root prediction rules and measurement moment maps.

``predict_sqrt(x, L)`` propagates a covariance factor. ``moments_sqrt(xi, L, k)``
returns ``(mean, B, E)`` with cross-covariance ``L @ B.T`` and innovation
covariance ``B @ B.T + E @ E.T``. The residual factor E includes measurement
noise as separate columns. Matrix-based APIs remain available as wrappers.
"""
import numpy as np

COVARIANCE_POLICY = "square-root-joseph-v1"


def sym(M):
    return 0.5 * (M + M.T)


def chol_lower(P):
    """Strict Cholesky factorization; never alter covariance eigenvalues."""
    return np.linalg.cholesky(sym(np.asarray(P, dtype=float)))


def qr_factor(*blocks):
    """Lower factor of the sum of block Gram matrices, without forming them."""
    if not blocks:
        raise ValueError("at least one covariance-factor block is required")
    arrays = [np.asarray(block, dtype=float) for block in blocks]
    if any(block.ndim != 2 for block in arrays):
        raise ValueError("covariance-factor blocks must be matrices")
    n = arrays[0].shape[0]
    if any(block.shape[0] != n for block in arrays):
        raise ValueError("covariance-factor blocks must have matching row counts")
    joined = np.concatenate(arrays, axis=1)
    if joined.shape[1] < n:
        joined = np.column_stack((joined, np.zeros((n, n - joined.shape[1]))))
    _, upper = np.linalg.qr(joined.T, mode="reduced")
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    return upper.T * signs


def noise_factor(Q):
    """Factor a PSD noise covariance, retaining zero modes without clipping."""
    covariance = sym(np.atleast_2d(np.asarray(Q, dtype=float)))
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("noise covariance must be square")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("noise covariance must be finite")
    diagonal = np.diag(covariance)
    if np.array_equal(covariance, np.diag(diagonal)):
        if np.any(diagonal < 0.0):
            raise np.linalg.LinAlgError("noise covariance is not positive semidefinite")
        return np.diag(np.sqrt(diagonal))
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        # A singular PSD matrix can acquire negative *computed* eigenvalues.
        # Pivoted Cholesky handles exact zero modes without clipping them.
        n = covariance.shape[0]
        residual = covariance.copy()
        factor = np.zeros_like(covariance)
        remaining = list(range(n))
        for column in range(n):
            pivot = max(remaining, key=lambda index: residual[index, index])
            value = residual[pivot, pivot]
            if value < 0.0:
                raise np.linalg.LinAlgError("noise covariance is not positive semidefinite")
            if value == 0.0:
                if np.any(residual[np.ix_(remaining, remaining)] != 0.0):
                    raise np.linalg.LinAlgError("noise covariance is not positive semidefinite")
                break
            factor[remaining, column] = residual[remaining, pivot] / np.sqrt(value)
            factor[pivot, column] = np.sqrt(value)
            remaining.remove(pivot)
            if remaining:
                block = np.ix_(remaining, remaining)
                v = factor[remaining, column]
                residual[block] = sym(residual[block] - np.outer(v, v))
        return factor


def _as_derivative_array(value, shape):
    if value is None:
        return np.zeros(shape, dtype=float)
    out = np.asarray(value, dtype=float)
    if out.shape != shape:
        raise ValueError(f"derivative has shape {out.shape}, expected {shape}")
    return out


def _taylor_moments_sqrt(value, jacobian, hessians, third_derivatives, L,
                         noise_cov=None):
    """Orthogonal Gaussian polynomial components of a cubic Taylor map."""
    value = np.asarray(value, dtype=float)
    J = np.asarray(jacobian, dtype=float)
    p, n = J.shape
    rank = L.shape[1]
    H = _as_derivative_array(hessians, (p, n, n))
    whitened_h = np.einsum("aij,ip,jq->apq", H, L, L, optimize=True)
    mean = value + 0.5 * np.einsum("aii->a", whitened_h)
    B = J @ L
    residuals = [whitened_h.reshape(p, rank ** 2) / np.sqrt(2.0)]
    if third_derivatives is not None:
        T = _as_derivative_array(third_derivatives, (p, n, n, n))
        whitened_t = np.einsum("aijk,ip,jq,kr->apqr", T, L, L, L,
                               optimize=True)
        B = B + 0.5 * np.einsum("aijj->ai", whitened_t)
        residuals.append(whitened_t.reshape(p, rank ** 3) / np.sqrt(6.0))
    if noise_cov is not None:
        residuals.append(noise_factor(noise_cov))
    return mean, B, np.concatenate(residuals, axis=1)


class _MatrixMomentAPI:
    """Compatibility wrappers; filtering recursions use the factor APIs."""

    def predict(self, x, P):
        mean, factor = self.predict_sqrt(x, chol_lower(P))
        return mean, sym(factor @ factor.T)

    def moments(self, xi, P, k):
        L = chol_lower(P)
        mean, B, E = self.moments_sqrt(xi, L, k)
        return mean, L @ B.T, sym(B @ B.T + E @ E.T)


class EKF(_MatrixMomentAPI):
    name = "EKF"
    n_points = 1

    def __init__(self, sysm):
        self.s = sysm

    def predict_sqrt(self, x, L):
        s = self.s
        return s.f(x), qr_factor(s.F_jac(x) @ L, noise_factor(s.Q))

    def moments_sqrt(self, xi, L, k):
        s = self.s
        return s.h(xi, k), s.h_jac(xi, k) @ L, noise_factor(s.R)


class EKF2(EKF):
    """Second-order Gaussian Taylor prediction and measurement moments."""
    name = "EKF2"

    def predict_sqrt(self, x, L):
        s = self.s
        hessians = s.f_hess(x) if hasattr(s, "f_hess") else None
        if hessians is None:
            return super().predict_sqrt(x, L)
        mean, B, E = _taylor_moments_sqrt(
            s.f(x), s.F_jac(x), hessians, None, L, s.Q,
        )
        return mean, qr_factor(B, E)

    def moments_sqrt(self, xi, L, k):
        s = self.s
        return _taylor_moments_sqrt(
            s.h(xi, k), s.h_jac(xi, k), s.h_hess(xi, k), None, L, s.R,
        )


def gaussian_taylor3_moments(value, jacobian, hessians, third_derivatives,
                             P, noise_cov=None):
    """Exact Gaussian moments of a cubic Taylor polynomial.

    The polynomial's linear, centered-quadratic, and third-order Hermite
    components give separate Gram factors, including every cubic covariance
    term. A positive-semidefinite input covariance is supported.
    """
    L = noise_factor(P)
    mean, B, E = _taylor_moments_sqrt(
        value, jacobian, hessians, third_derivatives, L, noise_cov,
    )
    return mean, L @ B.T, sym(B @ B.T + E @ E.T)


class EKF3(EKF2):
    """Third-order Gaussian Taylor measurement moments; EKF2 prediction."""
    name = "EKF3"

    def moments_sqrt(self, xi, L, k):
        s = self.s
        return _taylor_moments_sqrt(
            s.h(xi, k), s.h_jac(xi, k),
            s.h_hess(xi, k) if hasattr(s, "h_hess") else None,
            s.h_third(xi, k) if hasattr(s, "h_third") else None,
            L, s.R,
        )


class UKF(_MatrixMomentAPI):
    """Scaled unscented transform using positive paired covariance factors."""
    name = "UKF"

    def __init__(self, sysm, alpha=1e-3, beta=2.0, kappa=0.0):
        self.s = sysm
        n = sysm.n_x
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        # Direct evaluation avoids cancellation in n + lambda.
        self.c = alpha ** 2 * (n + kappa)
        self.even_coefficient = beta + alpha ** 2 * kappa / n
        if (not np.isfinite(self.c) or self.c <= 0.0 or alpha <= 0.0
                or not np.isfinite(self.even_coefficient)
                or self.even_coefficient < 0.0):
            raise ValueError("UKF factors require alpha > 0, n+kappa > 0, "
                             "and beta + alpha**2*kappa/n >= 0")
        self.lam = self.c - n
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * self.c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = 1.0 - n / self.c
        self.Wc[0] = self.Wm[0] + 1.0 - alpha ** 2 + beta
        self.n_points = 2 * n + 1

    def _sigma(self, x, P):
        offsets = np.sqrt(self.c) * chol_lower(P)
        return np.column_stack((x, x[:, None] + offsets, x[:, None] - offsets))

    def _transform_sqrt(self, x, L, function, noise, increment=None):
        offsets = np.sqrt(self.c) * L
        center = np.asarray(function(x), dtype=float)
        if increment is None:
            plus = np.column_stack([function(x + delta) for delta in offsets.T])
            minus = np.column_stack([function(x - delta) for delta in offsets.T])
            dplus, dminus = plus - center[:, None], minus - center[:, None]
        else:
            dplus = np.column_stack([increment(x, delta) for delta in offsets.T])
            dminus = np.column_stack([increment(x, -delta) for delta in offsets.T])
        odd = 0.5 * (dplus - dminus)
        even = 0.5 * (dplus + dminus)
        even_mean = np.mean(even, axis=1)
        shift = np.sum(even, axis=1) / self.c
        B = odd / np.sqrt(self.c)
        # This is algebraically the signed UT covariance, but each component
        # is positive: beta + alpha**2*kappa/n replaces the negative center.
        E = np.column_stack((
            (even - even_mean[:, None]) / np.sqrt(self.c),
            np.sqrt(self.even_coefficient) * shift,
            noise_factor(noise),
        ))
        return center + shift, B, E

    def predict_sqrt(self, x, L):
        mean, B, E = self._transform_sqrt(
            x, L, self.s.f, self.s.Q, getattr(self.s, "f_increment", None),
        )
        return mean, qr_factor(B, E)

    def moments_sqrt(self, xi, L, k):
        return self._transform_sqrt(xi, L, lambda x: self.s.h(x, k), self.s.R)


class CKF(_MatrixMomentAPI):
    """Third-degree cubature using odd and centered-even paired factors."""
    name = "CKF"

    def __init__(self, sysm):
        self.s = sysm
        self.n_points = 2 * sysm.n_x

    def _sigma(self, x, P):
        offsets = np.sqrt(self.s.n_x) * chol_lower(P)
        return np.column_stack((x[:, None] + offsets, x[:, None] - offsets))

    def _transform_sqrt(self, x, L, function, noise, increment=None):
        n = self.s.n_x
        offsets = np.sqrt(n) * L
        if increment is None:
            plus = np.column_stack([function(x + delta) for delta in offsets.T])
            minus = np.column_stack([function(x - delta) for delta in offsets.T])
            reference = plus[:, 0]
            dplus, dminus = plus - reference[:, None], minus - reference[:, None]
        else:
            reference = np.asarray(function(x), dtype=float)
            dplus = np.column_stack([increment(x, delta) for delta in offsets.T])
            dminus = np.column_stack([increment(x, -delta) for delta in offsets.T])
        odd = 0.5 * (dplus - dminus)
        even = 0.5 * (dplus + dminus)
        shift = np.mean(even, axis=1)
        B = odd / np.sqrt(n)
        E = np.column_stack(((even - shift[:, None]) / np.sqrt(n),
                             noise_factor(noise)))
        return reference + shift, B, E

    def predict_sqrt(self, x, L):
        mean, B, E = self._transform_sqrt(
            x, L, self.s.f, self.s.Q, getattr(self.s, "f_increment", None),
        )
        return mean, qr_factor(B, E)

    def moments_sqrt(self, xi, L, k):
        return self._transform_sqrt(xi, L, lambda x: self.s.h(x, k), self.s.R)


FILTERS = {"EKF": EKF, "EKF2": EKF2, "UKF": UKF, "CKF": CKF}


class GHKF(_MatrixMomentAPI):
    """Positive-weight tensor Gauss-Hermite measurement quadrature."""
    name = "GHKF"

    def __init__(self, sysm, m=3):
        self.s = sysm
        self.m = int(m)
        n = sysm.n_x
        xi, wi = np.polynomial.hermite_e.hermegauss(self.m)
        wi = wi / wi.sum()
        grid = np.meshgrid(*([xi] * n), indexing="ij")
        wgrid = np.meshgrid(*([wi] * n), indexing="ij")
        self.X0 = np.stack([g.ravel() for g in grid])
        self.W = np.prod(np.stack([g.ravel() for g in wgrid]), axis=0)
        self.n_points = self.X0.shape[1]

    def _sigma(self, x, P):
        return x[:, None] + chol_lower(P) @ self.X0

    def moments_sqrt(self, xi, L, k):
        s = self.s
        offsets = L @ self.X0
        Z = np.column_stack([s.h(xi + delta, k) for delta in offsets.T])
        reference = Z[:, 0]
        relative = Z - reference[:, None]
        shift = relative @ self.W
        weighted_nodes = self.X0 * np.sqrt(self.W)
        weighted_outputs = (relative - shift[:, None]) * np.sqrt(self.W)
        B = weighted_outputs @ weighted_nodes.T
        # The standardized quadrature nodes have covariance I. Regress in
        # their coordinates, retaining the residuals before taking products.
        residual = weighted_outputs - B @ weighted_nodes
        E = np.column_stack((residual, noise_factor(s.R)))
        return reference + shift, B, E


class CKF5(GHKF):
    """Positive-weight fifth-degree cubature in at most four dimensions."""
    name = "CKF5"

    def __init__(self, sysm):
        self.s = sysm
        n = sysm.n_x
        if n > 4:
            raise ValueError("fifth-degree weights go negative for n > 4")
        rho = np.sqrt(n + 2.0)
        points = [np.zeros(n)]
        weights = [2.0 / (n + 2.0)]
        w1 = (4.0 - n) / (2.0 * (n + 2.0) ** 2)
        w2 = 1.0 / (n + 2.0) ** 2
        for i in range(n):
            for sign in (1.0, -1.0):
                point = np.zeros(n)
                point[i] = sign
                points.append(rho * point)
                weights.append(w1)
        for i in range(n):
            for j in range(i + 1, n):
                for si in (1.0, -1.0):
                    for sj in (1.0, -1.0):
                        point = np.zeros(n)
                        point[i], point[j] = si, sj
                        points.append(rho * point / np.sqrt(2.0))
                        weights.append(w2)
        self.X0 = np.column_stack(points)
        self.W = np.asarray(weights)
        self.n_points = self.X0.shape[1]
        assert self.n_points == 2 * n * n + 1
        assert abs(self.W.sum() - 1.0) < 1e-12


class MeasurementMomentControl(_MatrixMomentAPI):
    """Pair a lower-order predictor with a higher-order measurement map."""

    def __init__(self, predictor, measurement_rule, name=None):
        self.s = predictor.s
        self.predictor = predictor
        self.measurement_rule = measurement_rule
        self.n_points = measurement_rule.n_points
        self.name = name or f"{predictor.name}+{measurement_rule.name}"

    def predict_sqrt(self, x, L):
        return self.predictor.predict_sqrt(x, L)

    def moments_sqrt(self, xi, L, k):
        return self.measurement_rule.moments_sqrt(xi, L, k)


def ekf2_control(sysm):
    return MeasurementMomentControl(EKF(sysm), EKF2(sysm), name="EKF2")


def ekf3_control(sysm):
    return MeasurementMomentControl(EKF2(sysm), EKF3(sysm), name="EKF3")


def gh_control(sysm, order):
    return MeasurementMomentControl(UKF(sysm), GHKF(sysm, order))


def ckf5_control(sysm):
    return MeasurementMomentControl(CKF(sysm), CKF5(sysm))
