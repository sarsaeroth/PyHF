import numpy as np
from scipy import special


def boys(n, x):
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    result = np.zeros_like(x)

    zero_mask = np.abs(x) < 1e-15
    result[zero_mask] = 1.0 / (2 * n + 1)

    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        x_nz = x[nonzero_mask]
        half_n = n + 0.5
        result[nonzero_mask] = (
            0.5 * special.gammainc(half_n, x_nz) *
            special.gamma(half_n) / (x_nz ** half_n)
        )

    if scalar_input:
        return float(result[0])
    return result


def boys_f0(x):
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    result = np.zeros_like(x)

    zero_mask = np.abs(x) < 1e-15
    result[zero_mask] = 1.0

    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        x_nz = x[nonzero_mask]
        sqrt_x = np.sqrt(x_nz)
        result[nonzero_mask] = 0.5 * np.sqrt(np.pi / x_nz) * special.erf(sqrt_x)

    if scalar_input:
        return float(result[0])
    return result


__all__ = ['boys', 'boys_f0']
