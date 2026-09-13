"""Probe-mean update and covariance-recalibration framework.

The framework varies two choices around a nonlinear Kalman filter moment map:

``gain``
    ``"single"`` evaluates one moment pair at the predicted state.
    ``"mean"`` evaluates ``n+1`` covariance-scaled simplex members and forms
    the gain from the arithmetic-mean moment pair.

``recal``
    ``"single"`` evaluates the generalized Joseph covariance once at the
    updated state.
    ``"set"`` evaluates it at ``n+1`` simplex members around the updated state
    and averages the returned moment pairs.

Back-out uses either ``metric="Pinv"`` with ``M=P_pred^{-1}``, which is
invariant to invertible linear state-coordinate changes, or ``metric="I"`` to
reproduce the raw-trace rule of the published framework. An update is backed
out exactly when ``trace(M @ P_new) > trace(M @ P_pred)``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from filters import chol_lower, qr_factor, sym

Array = np.ndarray
BACKOUT_POLICY = "trace-only-v1"


# ============================================================ probe geometry
@lru_cache(maxsize=None)
def simplex_dirs(n: int) -> Array:
    """Return the ``n+1`` unit directions of a centered regular simplex.

    The rows sum to zero and have pairwise inner product ``-1/n``. Multiplying
    them by ``sqrt(n)`` makes their arithmetic covariance equal to ``I_n``.
    The result is cached because the state dimension is fixed within a study.
    """
    if n < 1:
        raise ValueError("simplex dimension must be positive")
    if n == 1:
        vertices = np.array([[1.0], [-1.0]])
    else:
        eye = np.eye(n + 1)
        centered = eye - np.ones((n + 1, 1)) / (n + 1)
        # Any orthonormal basis of the all-ones orthogonal complement works.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:n].T
        vertices = centered @ basis
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices.setflags(write=False)
    return vertices


def probe_points(xi: Array, L: Array | None, kind: str) -> Tuple[List[Array], Array]:
    """Return probe points when a covariance factor is already available.

    For ``kind="set"``, ``L`` must satisfy ``L L.T = P``. The offsets then
    satisfy ``mean(delta_i)=0`` and ``mean(delta_i delta_i.T)=P``.
    """
    xi = np.asarray(xi, dtype=float)
    if kind == "single":
        return [xi], np.ones(1)
    if kind != "set":
        raise ValueError(f"unknown probe kind {kind!r}")
    if L is None:
        raise ValueError("a covariance factor is required for a probe set")

    n = xi.size
    directions = simplex_dirs(n)
    factor = np.asarray(L, dtype=float)
    offsets = [np.sqrt(n) * (factor @ direction) for direction in directions]
    points = [xi + offset for offset in offsets]
    return points, np.full(n + 1, 1.0 / (n + 1))


def probe_set(xi: Array, P: Array, kind: str) -> Tuple[List[Array], Array]:
    """Public convenience API that factors ``P`` before building a probe set.

    Publication drivers factor the prediction once and call :func:`probe_points`
    directly so the update and recalibration probes reuse the same factor.  This
    wrapper is retained for interactive use and deterministic geometry tests.
    """
    L = None if kind == "single" else chol_lower(P)
    return probe_points(xi, L, kind)


# =============================================================== covariance tools
def generalized_joseph(P: Array, Pxz: Array, S: Array, K: Array) -> Array:
    """Generalized Joseph covariance report for one moment pair and gain."""
    return sym(P - K @ Pxz.T - Pxz @ K.T + K @ S @ K.T)


def gain_from_factors(L, moments, weights):
    """Gain of a moment-pair average with normalized nonnegative weights."""
    reference = moments[0][1]
    B = reference + sum(w * (moment[1] - reference)
                        for w, moment in zip(weights, moments))
    blocks = [B]
    for weight, (_, local_B, E) in zip(weights, moments):
        scale = np.sqrt(weight)
        blocks.extend((scale * (local_B - B), scale * E))
    innovation_factor = qr_factor(*blocks)
    cross = L @ B.T
    return np.linalg.solve(
        innovation_factor.T,
        np.linalg.solve(innovation_factor, cross.T),
    ).T


def joseph_factor(L, K, moments, weights):
    """Square root of the average generalized Joseph covariance."""
    blocks = []
    for weight, (_, B, E) in zip(weights, moments):
        scale = np.sqrt(weight)
        blocks.extend((scale * (L - K @ B), scale * (K @ E)))
    return qr_factor(*blocks)


# ==================================================================== main loops
def run_conventional_once(flt, sysm, rng, *, return_factors=False):
    """Run F0 with an equivalent square-root Joseph covariance update."""
    s = sysm
    n_steps, n = s.N, s.n_x
    s.sample_truth(rng)
    x = s.draw_init(rng)
    L = chol_lower(s.P0)
    errors = np.zeros((n, n_steps + 1))
    covariances = np.zeros((n_steps + 1, n, n))
    factors = np.zeros_like(covariances) if return_factors else None
    errors[:, 0] = x - s.x_true[:, 0]
    covariances[0] = s.P0
    if return_factors:
        factors[0] = L

    for k in range(1, n_steps + 1):
        s._k = k - 1
        x_pred, L_pred = flt.predict_sqrt(x, L)
        z = s.measure(k, rng)
        moment = flt.moments_sqrt(x_pred, L_pred, k)
        zhat = moment[0]
        K = gain_from_factors(L_pred, [moment], [1.0])
        x = x_pred + K @ (z - zhat)
        L = joseph_factor(L_pred, K, [moment], [1.0])
        errors[:, k] = x - s.x_true[:, k]
        covariances[k] = L @ L.T
        if return_factors:
            factors[k] = L

    result = (errors, covariances, 0.0)
    return result + (factors,) if return_factors else result


def run_once(flt, sysm, rng, gain: str, recal: str, metric: str = "Pinv",
             *, return_factors=False):
    """Run one F1--F4 paired Monte Carlo trajectory.

    The prediction factor is carried directly and reused by the update and
    recalibration probes. QR updates preserve the covariance factor.
    Back-out compares only the candidate and predicted covariance traces in
    the selected metric. Equality retains the candidate update.
    """
    if gain not in {"single", "mean"}:
        raise ValueError(gain)
    if recal not in {"single", "set"}:
        raise ValueError(recal)
    if metric not in {"Pinv", "I"}:
        raise ValueError(metric)

    s = sysm
    n_steps, n = s.N, s.n_x
    s.sample_truth(rng)
    x = s.draw_init(rng)
    L = chol_lower(s.P0)
    errors = np.zeros((n, n_steps + 1))
    covariances = np.zeros((n_steps + 1, n, n))
    factors = np.zeros_like(covariances) if return_factors else None
    errors[:, 0] = x - s.x_true[:, 0]
    covariances[0] = s.P0
    if return_factors:
        factors[0] = L
    backouts = 0

    for k in range(1, n_steps + 1):
        s._k = k - 1
        x_pred, L_pred = flt.predict_sqrt(x, L)
        z = s.measure(k, rng)

        threshold = float(n) if metric == "Pinv" else float(np.sum(L_pred ** 2))

        gain_kind = "set" if gain == "mean" else "single"
        gain_points, gain_weights = probe_points(x_pred, L_pred, gain_kind)
        gain_moments = [flt.moments_sqrt(point, L_pred, k) for point in gain_points]
        if gain == "single":
            zhat = gain_moments[0][0]
        else:
            # The probes modify the covariance-based gain. The first moment is
            # the center prediction already supplied by the underlying filter.
            zhat = flt.moments_sqrt(x_pred, L_pred, k)[0]

        K = gain_from_factors(L_pred, gain_moments, gain_weights)
        x_upd = x_pred + K @ (z - zhat)

        recal_points, recal_weights = probe_points(x_upd, L_pred, recal)
        recal_moments = [flt.moments_sqrt(point, L_pred, k) for point in recal_points]
        L_new = joseph_factor(L_pred, K, recal_moments, recal_weights)

        scaled = np.linalg.solve(L_pred, L_new) if metric == "Pinv" else L_new
        score = float(np.sum(scaled ** 2))
        if score > threshold:
            backouts += 1
            x, L = x_pred, L_pred
        else:
            x, L = x_upd, L_new

        errors[:, k] = x - s.x_true[:, k]
        covariances[k] = L @ L.T
        if return_factors:
            factors[k] = L

    result = (errors, covariances, backouts / n_steps)
    return result + (factors,) if return_factors else result
