"""Multi-dimensional Chebyshev interpolation on a downward closed set by cropped solves
(fft_paper_w22, Part II: the corrected sweep). 2D for now.

Packed layout (lpfun): the index set A is lexicographic with the last coordinate fastest,
so the tubes along the last axis ("rows" i, length T[i], data x[cs_T[i] : cs_T[i + 1]])
are contiguous, and column j (last coordinate = j) is the strided subsequence
cs_T[i] + j for the rows i with T[i] > j; its length is L_j = #{i : T[i] > j}.

Rows are polynomials in the last variable, R_i(y) = sum_j c[i, j] T_j(y); the data is
f[i, j] = sum_k R_k(p_j) T_k(p_i). The table E[j, i] = R_i(p_j) ("row polynomials at
the column nodes") links the two directions. The naive "solve all rows, then all
columns" is wrong for non-tensor A because column j also sees the rows k >= L_j, whose
values R_k(p_j) are extrapolations, not zeros. Hence, for j = 0, 1, ...:

    1. finish every row i with T[i] == j (its E[0:j, i] are known): cropped solve;
    2. extrapolate it to the later columns: E[j:, i] = R_i(p_j), ..., one forward FCT;
    3. correct column j: subtract sum_{k >= L_j} E[j, k] T_k(p_i) from f[:, j]
       (one zero-padded forward FCT; those rows are finished, by downward closedness);
    4. solve column j: E[j, :L_j] by one cropped solve of size L_j.

The forward map (eval) is the same bookkeeping with forward transforms only.
"""

import numpy as np
import numba as nb
from lpfun import CACHE, PARALLEL
from lpfun.core.crop import crop_solve, crop_eval


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


prange = nb.prange


@njit
def column_lengths(T: np.ndarray) -> np.ndarray:
    """O(|A|)

    L[j] = number of rows i with T[i] > j, for j = 0, ..., T[0] - 1 (T non-increasing
    for a downward closed set, so the rows of column j are exactly i < L[j]).
    """
    n_cols = T[0]
    L = np.zeros(n_cols, dtype=np.int64)
    for i in range(T.shape[0]):
        for j in range(T[i]):
            L[j] += 1
    return L


@njit
def interp_2d(
    f: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    lidx: np.ndarray,
    J: int,
    omega: np.ndarray,
    offsets: np.ndarray,
    refine: bool = True,
) -> np.ndarray:
    """O(|A| log |A| + box)

    Chebyshev coefficients (packed over A) of the interpolant of the values f (packed
    over the grid P_A), two dimensions. See the module docstring. ``refine`` is passed
    to every cropped solve (one step of iterative refinement, twice the solve cost).
    """
    n_rows = T.shape[0]
    n_cols = T[0]
    L = column_lengths(T)
    c = np.zeros_like(f)
    E = np.zeros((n_cols, n_rows), dtype=np.float64)  # E[j, i] = R_i(p_j)
    # NOTE serial on purpose: the column loop is sequential and a parallel region per
    # column costs ~0.2 ms to launch, which made a prange over the rows 10x slower.
    for j in range(n_cols + 1):
        # 1 + 2: finish the rows ending at column j and extrapolate them; T is
        # non-increasing, so these are exactly the rows L[j] <= i < L[j - 1]
        lo = L[j] if j < n_cols else 0
        hi = L[j - 1] if j > 0 else 0
        for i in range(lo, hi):
            rhs = np.empty(j, dtype=np.float64)
            for jj in range(j):
                rhs[jj] = E[jj, i]
            c_i = crop_solve(rhs, lidx, J, omega, offsets, refine)
            c[cs_T[i] : cs_T[i + 1]] = c_i
            if j < n_cols:
                vals = crop_eval(c_i, n_cols, lidx, J)
                for jj in range(j, n_cols):
                    E[jj, i] = vals[jj]
        if j == n_cols:
            break
        # 3: column j data, minus the contribution of the finished rows k >= L_j
        L_j = L[j]
        rhs = np.empty(L_j, dtype=np.float64)
        for i in range(L_j):
            rhs[i] = f[cs_T[i] + j]
        if L_j < n_rows:
            tail = np.zeros(n_rows, dtype=np.float64)
            tail[L_j:] = E[j, L_j:]
            rhs -= crop_eval(tail, L_j, lidx, J)
        # 4: solve column j for the row values of the unfinished rows
        E[j, :L_j] = crop_solve(rhs, lidx, J, omega, offsets, refine)
    return c


@njit(parallel=PARALLEL)
def eval_2d(
    c: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    lidx: np.ndarray,
    J: int,
) -> np.ndarray:
    """O(|A| log |A| + box)

    Values on the grid P_A (packed) of the polynomial with coefficients c (packed over A),
    two dimensions: every row polynomial at all column nodes, then every column.
    """
    n_rows = T.shape[0]
    n_cols = T[0]
    L = column_lengths(T)
    E = np.zeros((n_cols, n_rows), dtype=np.float64)
    for i in prange(n_rows):
        vals = crop_eval(c[cs_T[i] : cs_T[i + 1]], n_cols, lidx, J)
        for j in range(n_cols):
            E[j, i] = vals[j]
    f = np.zeros_like(c)
    for j in prange(n_cols):
        L_j = L[j]
        vals = crop_eval(E[j].copy(), L_j, lidx, J)
        for i in range(L_j):
            f[cs_T[i] + j] = vals[i]
    return f
