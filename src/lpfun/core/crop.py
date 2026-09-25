import numpy as np
import numba as nb
from typing import Tuple
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


@njit
def get_bucket(N: int) -> int:
    """O(log N)

    Smallest M = 2^j + 1 with M >= N, the size of the Chebyshev grid containing the
    first N Leja dyadic nodes (0 for N = 0).
    """
    if N <= 0:
        return 0
    j = 0
    while (1 << j) + 1 < N:
        j += 1
    return (1 << j) + 1


@njit
def get_omega(x: np.ndarray, N_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """O(N_max^2)

    Nodal polynomial of the unseen nodes for every cropped size N = 1..N_max.
    For N inside the bucket M = get_bucket(N) with seen nodes S = x[:N] and unseen
    nodes U = x[N:M], r = M - N,

        Omega_U(y) = prod_{p in U} 2 (y - p),

    the row for N has length M and holds

        row[:N] = Omega_U(x[:N])   (values at the seen nodes)
        row[N:] = Omega_U'(x[N:M]) (derivative at the unseen nodes).

    Rows are stored flat: row for N is omega[offsets[N] : offsets[N + 1]].
    ``x`` must contain at least get_bucket(N_max) nodes in Leja dyadic order.

    Computed by descending N inside each bucket; when p = x[N - 1] moves from S to U,
    Omega picks up the factor 2 (x_i - p) on S and U, and Omega_U'(p) = 2 Omega(p).
    """
    M_top = get_bucket(N_max)
    if x.shape[0] < M_top:
        raise ValueError("Not enough nodes for the requested cropped sizes.")
    offsets = np.zeros(N_max + 2, dtype=np.int64)
    for N in range(N_max + 1):
        offsets[N + 1] = offsets[N] + get_bucket(N)
    omega = np.empty(offsets[N_max + 1], dtype=np.float64)

    j = 0
    while (1 << j) + 1 <= M_top:
        M = (1 << j) + 1
        N_lo = ((1 << (j - 1)) + 2) if j > 0 else 1
        row = np.ones(M, dtype=np.float64)
        if M <= N_max:
            omega[offsets[M] : offsets[M + 1]] = row
        for N in range(M, N_lo, -1):
            p = x[N - 1]
            for i in range(M):
                if i != N - 1:
                    row[i] *= 2.0 * (x[i] - p)
            row[N - 1] *= 2.0
            if N - 1 <= N_max:
                omega[offsets[N - 1] : offsets[N]] = row
        j += 1
    return omega, offsets


@njit
def omega_seen(omega: np.ndarray, offsets: np.ndarray, N: int) -> np.ndarray:
    """Omega_U at the N seen nodes."""
    return omega[offsets[N] : offsets[N] + N]


@njit
def omega_unseen(omega: np.ndarray, offsets: np.ndarray, N: int) -> np.ndarray:
    """Omega_U' at the r = M - N unseen nodes."""
    return omega[offsets[N] + N : offsets[N + 1]]
