"""Accuracy and consistency summaries shared by both experiments."""
import numpy as np

from filters import qr_factor

SUMMARY_VERSION = "per-run-rms-p95-v1"
METRIC_FIELDS = ("rmse_rms", "rmse_med", "rmse_p95", "anees")


def per_run_rmse(errors, key, burn_in=0.75):
    """Average the key-state RMSEs within one run after burn-in."""
    k0 = int(round(burn_in * (errors.shape[1] - 1)))
    state_rmse = np.sqrt(np.mean(errors[list(key), k0 + 1:] ** 2, axis=1))
    return float(np.mean(state_rmse))


def summarize_rmse(per_run):
    """RMS, median, and 95th percentile across per-run RMSE values."""
    values = np.asarray(per_run, dtype=float)
    return {
        "rmse_rms": float(np.sqrt(np.mean(values ** 2))),
        "rmse_med": float(np.median(values)),
        "rmse_p95": float(np.percentile(values, 95)),
    }


def nees_curve(errors, covariances, key, factors=None):
    """Normalized key-state NEES, using covariance factors when supplied.

    A key-state marginal is factored directly from the corresponding rows of
    the full covariance factor. This preserves small variances that can be
    lost when reconstructing and then refactoring a covariance matrix.
    """
    key = list(key)
    error = errors[key, :]
    values = np.empty(error.shape[1])
    for k in range(error.shape[1]):
        try:
            if factors is None:
                block = covariances[k][np.ix_(key, key)]
                factor = np.linalg.cholesky(0.5 * (block + block.T))
            else:
                factor = qr_factor(factors[k][key, :])
            standardized = np.linalg.solve(factor, error[:, k])
            value = float(standardized @ standardized)
        except np.linalg.LinAlgError:
            value = np.nan
        values[k] = value / len(key) if np.isfinite(value) else np.nan
    return values


def anees_of(errors, covariances, key, factors=None):
    """Average normalized NEES over all steps of one run."""
    return float(np.mean(nees_curve(errors, covariances, key, factors=factors)))


def geometric_mean(values):
    """Geometric mean over all supplied setup values."""
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values))))
