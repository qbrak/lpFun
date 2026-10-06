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


# ----------------------------------------------------------------------------- m dimensions
#
# Recursion over the last axis (readable_summary.tex Sec. 10, mdfct.py). A node of the
# recursion is a pair of downward closed sets A ⊆ K over d coordinates, both packed
# lexicographically (last coordinate fastest):
#
#   A : where the data lives (and the unknowns of this node)
#   K : the "known superset": the slice K_0 of the parent; the node's coefficient vector
#       is packed over K and is already known on K \ A (finished parent tubes).
#
# Tubes along the node's last axis are indexed by the points of K_0 (slice j = 0 of K);
# tube t has K-length RK[t] and A-length RA[t] (0 if the tube misses A). Column j is the
# (d-1)-dimensional child node (A_j, K_0) with A_j = {alpha' : (alpha', j) in A}; its
# coefficient vector is row j of this node's table D (shape (nA, |K_0|)), its data is the
# column-j gather of this node's data. At the top A = K = the index set.


class Plan:
    """Data-independent bookkeeping of the m-D recursion for one index set.

    Nodes are numbered depth first, node 0 is the top. Per node (all int64 arrays, flat
    with offsets):
        dim[q]            number of coordinates d
        n_tubes[q]        |K_0| (d >= 2), tubes along the last axis
        nA[q], nK[q]      max RA, max RK (number of columns of A and of K)
        RA, RK            tube lengths, slices node_off[q] : node_off[q] + n_tubes[q]
        csK               tube offsets in the K-packed vector (n_tubes + 1 entries)
        a2k               K_0-index of each A-tube (A_0 ⊆ K_0), slice atube_off[q] : atube_off[q+1]
        csA               tube offsets in the A-packed vector (n_A-tubes + 1 entries)
        finish            tubes sorted by RA; fo[j] .. fo[j+1] index the tubes with RA == j
        child             child node of column j, slice child_off[q] : child_off[q] + nA[q]
        LA[q], LK[q]      leaf (d == 1): lengths of A and K
    """

    def __init__(self, A: np.ndarray):
        A = np.asarray(A, dtype=np.int64)
        self.m = A.shape[1]
        self._nodes = []  # list of dicts, filled by _build
        self._build(A, A)
        self._pack()

    # -- construction (python, once) ------------------------------------------------
    @staticmethod
    def _tubes(S: np.ndarray):
        """Tubes of a lexicographically packed set S along its last axis: (S_0, lengths, offsets)."""
        if S.shape[1] == 1:
            return (
                np.zeros((1, 0), dtype=np.int64),
                np.array([len(S)], dtype=np.int64),
                np.array([0, len(S)], dtype=np.int64),
            )
        heads = S[:, :-1]
        new = np.ones(len(S), dtype=bool)
        new[1:] = np.any(heads[1:] != heads[:-1], axis=1)
        starts = np.flatnonzero(new)
        lengths = np.diff(np.append(starts, len(S)))
        return heads[starts], lengths, np.append(0, np.cumsum(lengths))

    @staticmethod
    def _encode(S: np.ndarray) -> np.ndarray:
        """One sortable key per row."""
        S = np.ascontiguousarray(S)
        return S.view(np.dtype((np.void, S.dtype.itemsize * S.shape[1]))).ravel()

    def _build(self, A: np.ndarray, K: np.ndarray) -> int:
        q = len(self._nodes)
        d = A.shape[1]
        if d == 1:
            self._nodes.append(dict(dim=1, LA=len(A), LK=len(K)))
            return q
        K0, RK, csK = self._tubes(K)
        A0, RA_A, csA = self._tubes(A)
        # map A-tubes to K_0 tubes (both lexicographic)
        kK, kA = self._encode(K0), self._encode(A0)
        order = np.argsort(kK)
        a2k = order[np.searchsorted(kK[order], kA)]
        assert np.array_equal(kK[a2k], kA), "A_0 is not a subset of K_0"
        RA = np.zeros(len(K0), dtype=np.int64)
        RA[a2k] = RA_A
        nA = int(RA.max())
        finish = np.argsort(RA, kind="stable")
        fo = np.searchsorted(RA[finish], np.arange(nA + 2))
        node = dict(
            dim=d, n_tubes=len(K0), nA=nA, nK=int(RK.max()), RA=RA, RK=RK, csK=csK,
            a2k=a2k, csA=csA, finish=finish, fo=fo, child=np.empty(nA, dtype=np.int64),
        )
        self._nodes.append(node)
        for j in range(nA):
            node["child"][j] = self._build(A0[RA_A > j], K0)
        return q

    def _pack(self):
        e = np.empty(0, dtype=np.int64)
        get = lambda key, default: [np.asarray(nd.get(key, default), dtype=np.int64) for nd in self._nodes]
        cat = lambda key: np.concatenate(get(key, e))
        offs = lambda key: np.append(0, np.cumsum([len(a) for a in get(key, e)]))
        self.n_nodes = len(self._nodes)
        self.dim = np.array([nd["dim"] for nd in self._nodes], dtype=np.int64)
        for key in ("LA", "LK", "n_tubes", "nA", "nK"):
            setattr(self, key, np.array([nd.get(key, 0) for nd in self._nodes], dtype=np.int64))
        self.RA, self.RK, self.finish = cat("RA"), cat("RK"), cat("finish")
        self.node_off = offs("RA")
        self.csK, self.csK_off = cat("csK"), offs("csK")
        self.a2k, self.atube_off = cat("a2k"), offs("a2k")
        self.csA, self.csA_off = cat("csA"), offs("csA")
        self.fo, self.fo_off = cat("fo"), offs("fo")
        self.child, self.child_off = cat("child"), offs("child")

    # -- accessors for one node (python side) -----------------------------------------
    def node(self, q: int) -> dict:
        s = slice(self.node_off[q], self.node_off[q + 1])
        return dict(
            dim=self.dim[q], LA=self.LA[q], LK=self.LK[q], n_tubes=self.n_tubes[q],
            nA=self.nA[q], nK=self.nK[q], RA=self.RA[s], RK=self.RK[s], finish=self.finish[s],
            csK=self.csK[self.csK_off[q] : self.csK_off[q + 1]],
            a2k=self.a2k[self.atube_off[q] : self.atube_off[q + 1]],
            csA=self.csA[self.csA_off[q] : self.csA_off[q + 1]],
            fo=self.fo[self.fo_off[q] : self.fo_off[q + 1]],
            child=self.child[self.child_off[q] : self.child_off[q + 1]],
        )


# -- numpy reference implementation (python recursion, jitted 1D kernels) -------------
def _column(F: np.ndarray, nd: dict, j: int) -> np.ndarray:
    """Column j of the A-packed vector F: the lexicographic packing of A_j."""
    RA_A = nd["RA"][nd["a2k"]]
    s = np.flatnonzero(RA_A > j)
    return F[nd["csA"][s] + j]


def _interp_rec_py(plan, q, F, C, lidx, J, omega, offsets, refine):
    """C (K-packed, known on K \\ A) is filled on A with the coefficients interpolating F (A-packed)."""
    nd = plan.node(q)
    if nd["dim"] == 1:
        LA, LK = nd["LA"], nd["LK"]
        rhs = F[:LA].copy()
        if LK > LA:
            tail = np.zeros(LK)
            tail[LA:] = C[LA:LK]
            rhs -= crop_eval(tail, LA, lidx, J)
        C[:LA] = crop_solve(rhs, lidx, J, omega, offsets, refine)
        return
    nA, n_tubes, RK, csK, finish, fo = nd["nA"], nd["n_tubes"], nd["RK"], nd["csK"], nd["finish"], nd["fo"]
    D = np.zeros((nA, n_tubes))
    for j in range(nA + 1):
        for t in finish[fo[j] : fo[j + 1]]:  # tubes whose A-part ends at column j
            lo, hi = csK[t], csK[t] + RK[t]
            if j > 0:
                rhs = D[:j, t].copy()
                if RK[t] > j:  # subtract the known tail of the K-tube
                    tail = np.zeros(RK[t])
                    tail[j:] = C[lo + j : hi]
                    rhs -= crop_eval(tail, j, lidx, J)
                C[lo : lo + j] = crop_solve(rhs, lidx, J, omega, offsets, refine)
            if j < nA:  # extrapolate the whole K-tube to the later columns
                D[j:, t] = crop_eval(C[lo:hi].copy(), nA, lidx, J)[j:]
        if j == nA:
            break
        _interp_rec_py(plan, nd["child"][j], _column(F, nd, j), D[j], lidx, J, omega, offsets, refine)


def _eval_rec_py(plan, q, C, F, lidx, J):
    """F (A-packed) = values on P_A of the polynomial with K-packed coefficients C."""
    nd = plan.node(q)
    if nd["dim"] == 1:
        F[: nd["LA"]] = crop_eval(C[: nd["LK"]].copy(), nd["LA"], lidx, J)
        return
    nA, n_tubes, RK, csK = nd["nA"], nd["n_tubes"], nd["RK"], nd["csK"]
    D = np.zeros((nA, n_tubes))
    for t in range(n_tubes):
        D[:, t] = crop_eval(C[csK[t] : csK[t] + RK[t]].copy(), nA, lidx, J)
    RA_A = nd["RA"][nd["a2k"]]
    for j in range(nA):
        pos = nd["csA"][np.flatnonzero(RA_A > j)] + j
        col = np.zeros(len(pos))
        _eval_rec_py(plan, nd["child"][j], D[j], col, lidx, J)
        F[pos] = col


def interp_md_py(f, plan, lidx, J, omega, offsets, refine=True):
    """Numpy reference of the m-D interpolation (python recursion over the plan)."""
    c = np.zeros_like(f)
    _interp_rec_py(plan, 0, np.asarray(f, dtype=np.float64), c, lidx, J, omega, offsets, refine)
    return c


def eval_md_py(c, plan, lidx, J):
    """Numpy reference of the m-D evaluation."""
    f = np.zeros_like(c)
    _eval_rec_py(plan, 0, np.asarray(c, dtype=np.float64), f, lidx, J)
    return f


# -- jitted implementation (recursion over the plan arrays, serial) --------------------
@njit
def _column_gather(F, j, RA, a2k, csA, n_atubes):
    """Column j of the A-packed vector F (the lexicographic packing of A_j)."""
    count = 0
    for s in range(n_atubes):
        if RA[a2k[s]] > j:
            count += 1
    col = np.empty(count, dtype=np.float64)
    k = 0
    for s in range(n_atubes):
        if RA[a2k[s]] > j:
            col[k] = F[csA[s] + j]
            k += 1
    return col


@njit
def _interp_rec(
    q, F, C, lidx, J, omega, offsets, refine,
    dim, LA, LK, n_tubes, nA, node_off, RA, RK, finish, csK, csK_off,
    a2k, atube_off, csA, csA_off, fo, fo_off, child, child_off,
):
    if dim[q] == 1:
        la, lk = LA[q], LK[q]
        rhs = F[:la].copy()
        if lk > la:
            tail = np.zeros(lk, dtype=np.float64)
            tail[la:] = C[la:lk]
            rhs -= crop_eval(tail, la, lidx, J)
        C[:la] = crop_solve(rhs, lidx, J, omega, offsets, refine)
        return
    na, nt, o = nA[q], n_tubes[q], node_off[q]
    ra = RA[o : o + nt]
    rk = RK[o : o + nt]
    fin = finish[o : o + nt]
    csk = csK[csK_off[q] : csK_off[q + 1]]
    a2k_q = a2k[atube_off[q] : atube_off[q + 1]]
    csa = csA[csA_off[q] : csA_off[q + 1]]
    fo_q = fo[fo_off[q] : fo_off[q + 1]]
    ch = child[child_off[q] : child_off[q + 1]]
    D = np.zeros((na, nt), dtype=np.float64)
    for j in range(na + 1):
        for idx in range(fo_q[j], fo_q[j + 1]):  # tubes whose A-part ends at column j
            t = fin[idx]
            lo, hi = csk[t], csk[t] + rk[t]
            if j > 0:
                rhs = np.empty(j, dtype=np.float64)
                for jj in range(j):
                    rhs[jj] = D[jj, t]
                if rk[t] > j:  # subtract the known tail of the K-tube
                    tail = np.zeros(rk[t], dtype=np.float64)
                    tail[j:] = C[lo + j : hi]
                    rhs -= crop_eval(tail, j, lidx, J)
                C[lo : lo + j] = crop_solve(rhs, lidx, J, omega, offsets, refine)
            if j < na:  # extrapolate the whole K-tube to the later columns
                vals = crop_eval(C[lo:hi].copy(), na, lidx, J)
                for jj in range(j, na):
                    D[jj, t] = vals[jj]
        if j == na:
            break
        col = _column_gather(F, j, ra, a2k_q, csa, a2k_q.shape[0])
        _interp_rec(
            ch[j], col, D[j], lidx, J, omega, offsets, refine,
            dim, LA, LK, n_tubes, nA, node_off, RA, RK, finish, csK, csK_off,
            a2k, atube_off, csA, csA_off, fo, fo_off, child, child_off,
        )


@njit
def _eval_rec(
    q, C, F, lidx, J,
    dim, LA, LK, n_tubes, nA, node_off, RA, RK, csK, csK_off,
    a2k, atube_off, csA, csA_off, child, child_off,
):
    if dim[q] == 1:
        F[: LA[q]] = crop_eval(C[: LK[q]].copy(), LA[q], lidx, J)
        return
    na, nt, o = nA[q], n_tubes[q], node_off[q]
    ra = RA[o : o + nt]
    rk = RK[o : o + nt]
    csk = csK[csK_off[q] : csK_off[q + 1]]
    a2k_q = a2k[atube_off[q] : atube_off[q + 1]]
    csa = csA[csA_off[q] : csA_off[q + 1]]
    ch = child[child_off[q] : child_off[q + 1]]
    D = np.zeros((na, nt), dtype=np.float64)
    for t in range(nt):
        vals = crop_eval(C[csk[t] : csk[t] + rk[t]].copy(), na, lidx, J)
        for j in range(na):
            D[j, t] = vals[j]
    n_atubes = a2k_q.shape[0]
    for j in range(na):
        col = _column_gather(F, j, ra, a2k_q, csa, n_atubes)  # sized like A_j; contents overwritten
        _eval_rec(
            ch[j], D[j], col, lidx, J,
            dim, LA, LK, n_tubes, nA, node_off, RA, RK, csK, csK_off,
            a2k, atube_off, csA, csA_off, child, child_off,
        )
        k = 0
        for s in range(n_atubes):
            if ra[a2k_q[s]] > j:
                F[csa[s] + j] = col[k]
                k += 1


def _plan_args(plan):
    return (
        plan.dim, plan.LA, plan.LK, plan.n_tubes, plan.nA, plan.node_off, plan.RA, plan.RK,
        plan.finish, plan.csK, plan.csK_off, plan.a2k, plan.atube_off, plan.csA, plan.csA_off,
        plan.fo, plan.fo_off, plan.child, plan.child_off,
    )


def interp_md(f, plan, lidx, J, omega, offsets, refine=True):
    """O(|A| log |A| + box)

    Chebyshev coefficients (packed over A) of the interpolant of the values f (packed
    over the grid P_A), any dimension, by the corrected recursive sweep over ``plan``.
    """
    c = np.zeros_like(f)
    _interp_rec(0, np.asarray(f, dtype=np.float64), c, lidx, J, omega, offsets, refine, *_plan_args(plan))
    return c


def eval_md(c, plan, lidx, J):
    """O(|A| log |A| + box)

    Values on the grid P_A (packed) of the polynomial with coefficients c (packed over A).
    """
    f = np.zeros_like(c)
    a = _plan_args(plan)
    _eval_rec(0, np.asarray(c, dtype=np.float64), f, lidx, J, *a[:8], *a[9:15], *a[17:])  # no finish, fo
    return f
