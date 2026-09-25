import lpfun
import pytest
import numpy as np
from itertools import product

# Parameters

ms = [1, 2, 3, 4, 5, 6]
ps = [1.0, 2.0, np.inf]
bases = ["newton", "chebyshev", "legendre"]
precomputation = [True, False]
m_p_ba_pr = list(product(ms, ps, bases, precomputation))
NS = [4, 5, 6, 7, 8]


# Tests


@pytest.mark.parametrize("m", ms)
def test_tube_absolute_degree(m: int):
    for n in NS:
        A = lpfun.core.set.lp_set(m, n, 1.0)
        tube = lpfun.core.set.lp_tube(A, m, n, 1.0)
        tube_sum = np.sum(tube)
        cardinality = lpfun.utils.binomial(n + m, m)
        assert tube_sum == cardinality


@pytest.mark.parametrize("m", ms)
def test_tube_euclidean_degree(m: int):
    for n in NS:
        A = lpfun.core.set.lp_set(m, n, 2.0)
        tube = lpfun.core.set.lp_tube(A, m, n, 2.0)
        tube_sum = np.sum(tube)
        cardinality = len(
            [
                point
                for point in product(range(n + 1), repeat=m)
                if np.linalg.norm(point) <= n
            ]
        )
        assert tube_sum == cardinality


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_interp_eval(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )
        function_values = np.random.rand(len(fun))
        reconstruction = fun.eval(fun.interp(function_values))
        eps = np.linalg.norm(reconstruction - function_values)
        assert eps < 1e-8


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_diff(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )

        def f(x):
            val = 0.0
            for j in range(m):
                val += (j + 1) * x[j] ** (n - 1)
            return val

        def df(dim, order, x):
            c = dim + 1

            if order == 0:
                return f(x)

            if order == 1:
                return c * (n - 1) * x[dim] ** (n - 2)

            elif order == 2:
                return c * (n - 1) * (n - 2) * x[dim] ** (n - 3)

            elif order == 3:
                return c * (n - 1) * (n - 2) * (n - 3) * x[dim] ** (n - 4)

        function_values = np.array([f(x) for x in fun.grid])
        coeffs = fun.interp(function_values)

        for order in [0, 1, 2, 3]:
            for dim in range(m):
                dx_function_values = np.array([df(dim, order, x) for x in fun.grid])

                dx_reconstruction = (
                    fun.eval(coeffs)
                    if order == 0
                    else fun.eval(fun.diff(coeffs, dim, order))
                )

                eps = np.linalg.norm(dx_reconstruction - dx_function_values)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_diffT(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )

        for order in [1, 2, 3]:
            for dim in range(m):
                x = np.random.randn(len(fun))
                y = np.random.randn(len(fun))

                Dx = fun.diff(x, dim, order)
                DTx = fun.diffT(y, dim, order)

                lhs = np.dot(Dx, y)
                rhs = np.dot(x, DTx)

                eps = np.abs(lhs - rhs)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_call(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n + 1,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )

        def f(x):
            return np.sum(x**3)

        function_values = np.array([f(x) for x in fun.grid])
        coeffs = fun.interp(function_values)
        points = np.random.rand(10, m)
        function_value = np.array([f(x) for x in points])
        reconstruction = fun(coeffs, points)
        eps = np.max(np.abs(reconstruction - function_value))
        assert eps < 1e-6


@pytest.mark.parametrize("J", [0, 1, 2, 3, 4, 5, 6, 8])
def test_leja_dyadic_order(J: int):
    order = lpfun.core.grid.get_leja_dyadic_order(J)
    n = 1 << J
    assert len(np.unique(order)) == n + 1
    # every dyadic prefix is the full grid Cheb_{2^j}
    for j in range(J + 1):
        prefix = order[: (1 << j) + 1] >> (J - j)
        assert sorted(prefix) == list(range((1 << j) + 1))
    # worked example: 1, -1, 0, +-sqrt2/2, +-cos(3pi/8), +-cos(pi/8)
    if J >= 3:
        x = np.cos(order[:9] * np.pi / n)
        expected = [1, -1, 0, np.sqrt(2) / 2, -np.sqrt(2) / 2]
        assert np.allclose(x[:5], expected)
        assert np.allclose(np.abs(x[5:7]), np.cos(3 * np.pi / 8))
        assert np.allclose(np.abs(x[7:9]), np.cos(np.pi / 8))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 9, 20, 33, 35])
def test_leja_dyadic_nodes(n: int):
    x = lpfun.basis.nodes.leja_dyadic_nodes(n)
    assert len(x) == n
    assert len(np.unique(x)) == n
    # prefix property
    assert np.array_equal(lpfun.basis.nodes.leja_dyadic_nodes(n + 5)[:n], x)


def _omega_naive(x, N):
    M = lpfun.core.crop.get_bucket(N)
    xs, xu = x[:N], x[N:M]
    om_S = np.array([np.prod(2 * (s - xu)) for s in xs])
    om_U = np.array(
        [2 * np.prod([2 * (xu[u] - xu[v]) for v in range(len(xu)) if v != u]) for u in range(len(xu))]
    )
    return om_S, om_U


@pytest.mark.parametrize("N_max", [1, 2, 5, 9, 20, 35])
def test_omega(N_max: int):
    M_top = lpfun.core.crop.get_bucket(N_max)
    x = lpfun.basis.nodes.leja_dyadic_nodes(M_top)
    omega, offsets = lpfun.core.crop.get_omega(x, N_max)
    assert offsets[-1] == sum(lpfun.core.crop.get_bucket(N) for N in range(N_max + 1))
    for N in range(1, N_max + 1):
        om_S = lpfun.core.crop.omega_seen(omega, offsets, N)
        om_U = lpfun.core.crop.omega_unseen(omega, offsets, N)
        om_S_ref, om_U_ref = _omega_naive(x, N)
        assert len(om_S) == N and len(om_U) == lpfun.core.crop.get_bucket(N) - N
        assert np.allclose(om_S, om_S_ref, rtol=1e-12, atol=0)
        assert np.allclose(om_U, om_U_ref, rtol=1e-12, atol=0)
    # worked example N = 7, M = 9: Omega = 2 T_2 - sqrt2, Omega' = 8 x
    if N_max >= 7:
        om_S = lpfun.core.crop.omega_seen(omega, offsets, 7)
        om_U = lpfun.core.crop.omega_unseen(omega, offsets, 7)
        assert np.allclose(om_S, 2 * (2 * x[:7] ** 2 - 1) - np.sqrt(2))
        assert np.allclose(om_U, 8 * x[7:9])


@pytest.mark.parametrize("n, pr", list(product([3, 7, 8, 10], precomputation)))
def test_function_omega(n: int, pr: bool):
    fun = lpfun.Function(
        1,
        n,
        basis="chebyshev",
        nodes=lpfun.basis.nodes.leja_dyadic_nodes,
        leja_ordering=False,
        precomputation=pr,
        precompilation=False,
        report=False,
    )
    assert np.array_equal(fun.nodes, lpfun.basis.nodes.leja_dyadic_nodes(n + 1))
    assert np.array_equal(fun.leja_order, np.arange(n + 1))
    x = lpfun.basis.nodes.leja_dyadic_nodes(lpfun.core.crop.get_bucket(n + 1))
    for N in range(1, n + 2):
        om_S, om_U = fun.omega(N)
        om_S_ref, om_U_ref = _omega_naive(x, N)
        assert np.allclose(om_S, om_S_ref, rtol=1e-12, atol=0)
        assert np.allclose(om_U, om_U_ref, rtol=1e-12, atol=0)
    # reordered nodes, or a callable that does not cover the bucket: no Omega tables
    cases = [
        dict(),
        dict(nodes=lpfun.basis.nodes.leja_dyadic_nodes, leja_ordering=True),
    ]
    if lpfun.core.crop.get_bucket(n + 1) > n + 1:  # a fixed-size callable misses the bucket
        cases.append(dict(nodes=lambda k: lpfun.basis.nodes.leja_dyadic_nodes(n + 1), leja_ordering=False))
    for kwargs in cases:
        fun = lpfun.Function(1, n, precomputation=pr, precompilation=False, report=False, **kwargs)
        with pytest.raises(ValueError):
            fun.omega(1)


def test_leja_nodes_sample_size():
    with pytest.raises(ValueError):
        lpfun.basis.nodes.leja_nodes(9, m=7)
