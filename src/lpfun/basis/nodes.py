import numpy as np
from lpfun.core.grid import get_leja_order, get_leja_dyadic_order
from lpfun.core.crop import get_bucket


def cheb2nd_nodes(n: int) -> np.ndarray:
    """O(n)"""
    n = int(n)
    ###
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    return np.cos(np.arange(n, dtype=np.float64) * np.pi / (n - 1))


def leja_nodes(n: int, m: int = 25_000) -> np.ndarray:
    """O(n^3)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    if n > m:
        raise ValueError(
            f"The amount of nodes {n} must be smaller or equal than the sample size {m}."
        )
    sample_nodes = cheb2nd_nodes(m)
    leja_order = get_leja_order(sample_nodes, limit=n)
    return sample_nodes[leja_order]


def leja_dyadic_nodes(n: int) -> np.ndarray:
    """O(n^2)

    The first ``n`` nodes of the dyadically nested, Leja ordered Chebyshev (Lobatto)
    sequence, see ``get_leja_dyadic_order``. Prefixes of length 2^j + 1 are full
    Chebyshev grids; the nodes are already in Leja order, so use them with
    ``Function(..., leja_ordering=False)``.
    """
    n = int(n)
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    J = (get_bucket(n) - 1).bit_length() - 1  # bucket 2^J + 1 >= n
    order = get_leja_dyadic_order(J)
    return np.cos(order[:n] * np.pi / (1 << J))
