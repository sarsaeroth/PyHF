import numpy as np
from scipy import linalg


def symmetric_orthogonalization(S):
    s, U = linalg.eigh(S)
    if np.any(s < 1e-10):
        small_eigs = s[s < 1e-10]
        raise ValueError(
            f"Overlap matrix has {len(small_eigs)} eigenvalue(s) < 1e-10. "
            "Linear dependencies detected in basis set."
        )
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s))
    X = U @ s_inv_sqrt @ U.T
    return X


def canonical_orthogonalization(S, threshold=1e-7):
    s, U = linalg.eigh(S)
    mask = s > threshold
    s_kept = s[mask]
    U_kept = U[:, mask]
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s_kept))
    X = U_kept @ s_inv_sqrt
    n_removed = len(s) - len(s_kept)
    if n_removed > 0:
        print(f"Removed {n_removed} linear dependencies")
    return X

__all__ = ['symmetric_orthogonalization', 'canonical_orthogonalization']
