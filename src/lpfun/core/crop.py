"""Cropped solves on Leja dyadic Chebyshev nodes (fft_paper_w22, Part I).

Interpolation on the first N nodes of the dyadically nested, Leja ordered Chebyshev
sequence: the nodes are embedded in the smallest Chebyshev grid ("bucket", M = 2^j + 1
nodes) containing them, the missing values are recovered through the nodal polynomial
of the unseen nodes, and all transforms are DCT-I (fast cosine transforms).
"""

import numpy as np
import numba as nb

# rocket-fft registers numba implementations of scipy.fft; with it imported,
# scipy.fft calls inside @njit compile to its pocketfft bindings. The alias
# makes that explicit at the call sites.
import rocket_fft  # noqa: F401
import scipy.fft as rocket

from typing import Tuple
from lpfun import CACHE
from lpfun.basis.diff import chebyshev_derivative


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
def get_omega(nodes: np.ndarray, N_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """O(N_max^2)

    Nodal polynomial of the unseen nodes for every cropped size N = 1..N_max.
    For N inside the bucket M = get_bucket(N) with seen nodes S = nodes[:N] and unseen
    nodes U = nodes[N:M], r = M - N,

        Omega_U(x) = prod_{p in U} 2 (x - p).

    Going from N to N - 1 makes one more node unseen, p_N = nodes[N - 1], so Omega gains
    the single factor 2 (x - p_N); evaluating that identity and its derivative
    (product rule, using Omega^(N)(p_i) = 0 for unseen p_i) gives the recurrence
    (readable_summary.tex, Sec. 8)

        Omega^(N-1)(p_i)  = Omega^(N)(p_i)  * 2 (p_i - p_N)     i < N   (seen)
        Omega^(N-1)'(p_i) = Omega^(N)'(p_i) * 2 (p_i - p_N)     i > N   (unseen)
        Omega^(N-1)'(p_N) = 2 Omega^(N)(p_N)                            (new unseen)

    which is what the inner loop below applies. The row for N has length M and holds

        row[:N] = Omega_U(nodes[:N])   (values at the seen nodes)
        row[N:] = Omega_U'(nodes[N:M]) (derivative at the unseen nodes).

    Rows are stored flat: row for N is omega[offsets[N] : offsets[N + 1]].
    ``nodes`` must contain at least get_bucket(N_max) nodes in Leja dyadic order.

    Computed by descending N inside each bucket, starting from N = M where Omega = 1.
    """
    M_top = get_bucket(N_max)
    if nodes.shape[0] < M_top:
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
            p = nodes[N - 1]
            for i in range(M):
                if i != N - 1:
                    row[i] *= 2.0 * (nodes[i] - p)
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


@njit
def get_lobatto(lidx: np.ndarray, J: int, N: int) -> np.ndarray:
    """O(M)

    Lobatto indices of the bucket of N, from the level-J indices ``lidx`` of the whole
    Leja dyadic sequence: the i-th bucket node is cos(lob[i] pi / (M - 1)), M = get_bucket(N).
    Gather ``v_lob[lob]`` maps Lobatto order to P-order, scatter ``v_lob[lob] = v`` back.
    """
    M = get_bucket(N)
    j = 0
    while (1 << j) + 1 < M:
        j += 1
    return lidx[:M] >> (J - j)


@njit
def fct_eval(a: np.ndarray, lob: np.ndarray) -> np.ndarray:
    """O(M log M)

    V_M a: evaluate the Chebyshev series sum_k a_k T_k at the M = n + 1 bucket nodes.

    Parameters
    ----------
    a : np.ndarray
        Chebyshev coefficients a_0, ..., a_n, length M.
    lob : np.ndarray
        Lobatto indices of the bucket nodes in P-order, from get_lobatto: the i-th
        P-order node is cos(lob[i] pi / n). The DCT produces values in Lobatto order
        (position l <-> node cos(l pi / n)); gathering with ``lob`` returns them in P-order.

    Returns
    -------
    np.ndarray
        Values of the series at the M bucket nodes, in P-order.

    Notes
    -----
    Scaling. The target is V_M a, i.e. for every Lobatto index l

        p(cos(l pi / n)) = sum_{k=0}^{n} a_k cos(k l pi / n)

    with all terms weighted 1. scipy's DCT-I of a vector w returns

        c(w)_l = w_0 + (-1)^l w_n + 2 sum_{k=1}^{n-1} w_k cos(k l pi / n)

    so the interior terms k = 1, ..., n-1 come out doubled while the two endpoint
    terms do not (note (-1)^l = cos(n l pi / n)). Hence: halve the interior
    coefficients, w_k = a_k / 2 for 0 < k < n, keep w_0 = a_0 and w_n = a_n,
    transform, then gather into P-order.
    """
    n = a.shape[0] - 1
    w = a.copy()
    for k in range(1, n):
        w[k] *= 0.5
    v_lob = rocket.dct(w, type=1)
    v = np.empty_like(v_lob)
    for i in range(n + 1):
        v[i] = v_lob[lob[i]]
    return v


@njit
def fct_interp(f: np.ndarray, lob: np.ndarray) -> np.ndarray:
    """O(M log M)

    W f = V_M^{-1} f: Chebyshev coefficients interpolating values at the M = n + 1
    bucket nodes. Inverse of fct_eval.

    Parameters
    ----------
    f : np.ndarray
        Function values at the M bucket nodes, in P-order.
    lob : np.ndarray
        Lobatto indices of the bucket nodes in P-order, from get_lobatto: the i-th
        P-order node is cos(lob[i] pi / n). Scattering with ``lob`` lays the values out
        in Lobatto order (position l <-> node cos(l pi / n)), as the DCT expects.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients a_0, ..., a_n with sum_k a_k T_k(x_i) = f_i, length M.

    Notes
    -----
    Scaling. The target is a = W f. The discrete orthogonality formula for the
    Chebyshev coefficients of values f in Lobatto order is

        a_k = (2 / (n gamma_k)) * [ f_0/2 + (-1)^k f_n/2 + sum_{l=1}^{n-1} f_l cos(k l pi / n) ]

    with gamma_0 = gamma_n = 2 and gamma_k = 1 otherwise. scipy's DCT-I of f is

        c(f)_k = f_0 + (-1)^k f_n + 2 sum_{l=1}^{n-1} f_l cos(k l pi / n)

    i.e. the DCT-I already weights the endpoint values by 1 and the interior values
    by 2, which is exactly 2 * [bracket]: no halving of the values is needed. Hence

        a_k = c(f)_k / (n gamma_k).

    scipy's idct(type=1) equals c / (2n), so a_k = 2 * idct(f)_k / gamma_k: scatter f
    into Lobatto order, apply idct, double everything, then halve a_0 and a_n.
    """
    n = f.shape[0] - 1
    v = np.empty_like(f)
    for i in range(n + 1):
        v[lob[i]] = f[i]
    a = 2.0 * rocket.idct(v, type=1)
    a[0] *= 0.5
    a[n] *= 0.5
    return a


@njit
def crop_eval(c: np.ndarray, lidx: np.ndarray, J: int) -> np.ndarray:
    """O(M log M)

    Values at the first N nodes of the polynomial with the N Chebyshev coefficients c:
    zero-pad to the bucket, one forward FCT, keep the first N entries.
    """
    N = c.shape[0]
    M = get_bucket(N)
    lob = get_lobatto(lidx, J, N)
    a = np.zeros(M, dtype=np.float64)
    a[:N] = c
    return fct_eval(a, lob)[:N]


@njit
def _crop_solve_once(
    f: np.ndarray, N: int, M: int, lob: np.ndarray, om_S: np.ndarray, om_U: np.ndarray
) -> np.ndarray:
    """Algorithm 1 of the paper (three FCTs), no refinement.

    Returns all M coefficients of the final interpolation: the first N are c, the
    remaining r are zero up to rounding (a free consistency check, see crop_tail).
    """
    h = np.zeros(M, dtype=np.float64)
    h[:N] = f * om_S  # values of p * Omega_U at the bucket nodes; zero at the unseen nodes
    a = fct_interp(h, lob)  # coefficients of p * Omega_U         (FCT 1)
    b = chebyshev_derivative(a)  # coefficients of its derivative
    hp = fct_eval(b, lob)  # derivative at all bucket nodes         (FCT 2)
    h[:N] = f
    h[N:] = hp[N:] / om_U  # g = p at the unseen nodes
    return fct_interp(h, lob)  # coefficients of p, [c, ~0]          (FCT 3)


@njit
def crop_solve(
    f: np.ndarray,
    lidx: np.ndarray,
    J: int,
    omega: np.ndarray,
    offsets: np.ndarray,
    refine: bool = True,
) -> np.ndarray:
    """O(M log M)

    Cropped solve V_N c = f: Chebyshev coefficients c of the polynomial of degree < N
    interpolating values f at the first N Leja dyadic nodes (P-order), M = get_bucket(N).

    The unseen bucket nodes are filled in via the nodal polynomial Omega_U (tables from
    get_omega): p * Omega_U has bucket degree and vanishes at the unseen nodes, so one
    inverse FCT gives it exactly, and p(x_u) = (p Omega_U)'(x_u) / Omega_U'(x_u) by the
    product rule. With ``refine`` one step of iterative refinement is applied (residual
    by one forward FCT, second solve), which brings the error to machine precision.
    """
    N = f.shape[0]
    M = get_bucket(N)
    lob = get_lobatto(lidx, J, N)
    if M == N:
        return fct_interp(f, lob)
    om_S = omega_seen(omega, offsets, N)
    om_U = omega_unseen(omega, offsets, N)
    c = _crop_solve_once(f, N, M, lob, om_S, om_U)[:N]
    if refine:
        a = np.zeros(M, dtype=np.float64)
        a[:N] = c
        rho = f - fct_eval(a, lob)[:N]
        c = c + _crop_solve_once(rho, N, M, lob, om_S, om_U)[:N]
    return c


@njit
def crop_tail(
    f: np.ndarray, lidx: np.ndarray, J: int, omega: np.ndarray, offsets: np.ndarray
) -> float:
    """O(M log M)

    Diagnostic: largest discarded coefficient of the (unrefined) cropped solve for f,
    relative to the largest kept one. Exactly zero in exact arithmetic, so a value far
    above rounding indicates wrong Omega tables or a wrong permutation.
    """
    N = f.shape[0]
    M = get_bucket(N)
    if M == N:
        return 0.0
    lob = get_lobatto(lidx, J, N)
    v = _crop_solve_once(f, N, M, lob, omega_seen(omega, offsets, N), omega_unseen(omega, offsets, N))
    return np.abs(v[N:]).max() / np.abs(v[:N]).max()
