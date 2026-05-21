"""Unit tests for the linear-solver helper in helpers.py."""
from __future__ import annotations

import math

import pytest

from custom_components.heating_analytics.helpers import solve_gauss_jordan


def _matmul_vec(A: list[list[float]], x: list[float]) -> list[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _approx_equal(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


class TestSolveGaussJordan:
    def test_2x2_simple(self):
        # x + 2y = 5, 3x + 4y = 11  → x=1, y=2
        x = solve_gauss_jordan([[1.0, 2.0], [3.0, 4.0]], [5.0, 11.0])
        assert _approx_equal(x, [1.0, 2.0])

    def test_3x3_diagonal(self):
        A = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        x = solve_gauss_jordan(A, [4.0, 9.0, 16.0])
        assert _approx_equal(x, [2.0, 3.0, 4.0])

    def test_3x3_dense_pivoting(self):
        # Permuted rows force a pivot swap on the first step.
        A = [[0.0, 1.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.0, 3.0]]
        x_true = [1.0, 2.0, 3.0]
        b = _matmul_vec(A, x_true)
        x = solve_gauss_jordan(A, b)
        assert _approx_equal(x, x_true)

    def test_4x4_random_well_conditioned(self):
        A = [
            [4.0, 1.0, 0.0, 2.0],
            [1.0, 5.0, 1.0, 0.0],
            [0.0, 1.0, 6.0, 1.0],
            [2.0, 0.0, 1.0, 7.0],
        ]
        x_true = [1.0, -2.0, 3.0, -4.0]
        b = _matmul_vec(A, x_true)
        x = solve_gauss_jordan(A, b)
        assert _approx_equal(x, x_true)

    def test_5x5_well_conditioned(self):
        A = [[float(1 + (i == j) * 9) for j in range(5)] for i in range(5)]
        x_true = [0.5, 1.5, -1.0, 2.0, -3.0]
        b = _matmul_vec(A, x_true)
        x = solve_gauss_jordan(A, b)
        assert _approx_equal(x, x_true, tol=1e-8)

    def test_singular_returns_none(self):
        # Row 2 = 2 * Row 1 → rank-deficient.
        A = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]]
        assert solve_gauss_jordan(A, [1.0, 2.0, 3.0]) is None

    def test_zero_matrix_returns_none(self):
        A = [[0.0, 0.0], [0.0, 0.0]]
        assert solve_gauss_jordan(A, [0.0, 0.0]) is None

    def test_ridge_breaks_singularity(self):
        # Without ridge, this is singular; with a small ridge, solvable.
        A = [[1.0, 1.0], [1.0, 1.0]]
        b = [2.0, 2.0]
        assert solve_gauss_jordan(A, b) is None
        x = solve_gauss_jordan(A, b, ridge=1e-6)
        assert x is not None
        # Solution should be close to the minimum-norm LS answer [1, 1].
        assert _approx_equal(x, [1.0, 1.0], tol=1e-3)

    def test_pivot_eps_threshold(self):
        # Pivot just above default eps succeeds, just below fails.
        A = [[1e-11, 0.0], [0.0, 1.0]]
        assert solve_gauss_jordan(A, [1e-11, 1.0]) is not None
        A2 = [[1e-13, 0.0], [0.0, 1.0]]
        assert solve_gauss_jordan(A2, [1e-13, 1.0]) is None

    def test_input_not_mutated(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        b = [5.0, 11.0]
        A_copy = [row[:] for row in A]
        b_copy = b[:]
        solve_gauss_jordan(A, b)
        assert A == A_copy
        assert b == b_copy

    def test_dimension_mismatch_returns_none(self):
        assert solve_gauss_jordan([[1.0, 2.0], [3.0, 4.0]], [1.0]) is None

    def test_empty_returns_none(self):
        assert solve_gauss_jordan([], []) is None

    def test_collinear_3x3_normal_matrix(self):
        # Mimics the cold-start collinear sun case: all samples lie
        # along S-direction, so E and W columns are zero.  Normal
        # matrix is rank-1 → must return None so the caller routes to
        # the 1D collinear fallback.
        samples = [(1.0, 0.0, 0.0, 2.0), (0.5, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.6)]
        sum_s2 = sum(s[0] ** 2 for s in samples)
        sum_e2 = sum(s[1] ** 2 for s in samples)
        sum_w2 = sum(s[2] ** 2 for s in samples)
        A = [[sum_s2, 0.0, 0.0], [0.0, sum_e2, 0.0], [0.0, 0.0, sum_w2]]
        b = [sum(s[0] * s[3] for s in samples), 0.0, 0.0]
        assert solve_gauss_jordan(A, b) is None
