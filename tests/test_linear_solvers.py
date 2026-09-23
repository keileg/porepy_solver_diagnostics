"""Tests for the accuracy measures the experiment harness puts around every solve.

A failure here means the harness can no longer tell a trustworthy linear solve from an
untrustworthy one, which is the guard the whole campaign leans on: on this model's
systems a wrong answer is otherwise reported as success.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from solver_experiments.linear_solvers import (
    POREPY_BACKENDS,
    _reject_substituted_backend,
    backward_error,
    build_linear_solver,
)


@pytest.fixture
def badly_scaled_system() -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """A diagonal system whose rows span twenty orders of magnitude."""
    scales = np.array([1e10, 1.0, 1e-10])
    matrix = sps.diags(scales).tocsr()
    solution = np.array([2.0, 3.0, 4.0])
    return matrix, matrix @ solution, solution


def test_exact_solution_has_negligible_backward_error(badly_scaled_system) -> None:
    matrix, rhs, solution = badly_scaled_system
    assert backward_error(matrix, rhs, solution) < 1e-15


def test_row_scaling_does_not_change_the_backward_error(badly_scaled_system) -> None:
    """The point of the measure: a badly scaled row must not dominate the verdict."""
    matrix, rhs, solution = badly_scaled_system
    perturbed = solution + np.array([0.0, 0.0, 1e-8])
    scaling = sps.diags(1.0 / abs(matrix).max(axis=1).toarray().ravel())
    unscaled = backward_error(matrix, rhs, perturbed)
    scaled = backward_error((scaling @ matrix).tocsr(), scaling @ rhs, perturbed)
    assert unscaled == pytest.approx(scaled, rel=1e-12)


def test_a_wrong_answer_in_a_weak_row_is_detected(badly_scaled_system) -> None:
    """The weakest row is the one the plain relative residual cannot see."""
    matrix, rhs, solution = badly_scaled_system
    wrong = solution + np.array([0.0, 0.0, 1.0])
    plain = np.linalg.norm(matrix @ wrong - rhs) / np.linalg.norm(rhs)
    assert plain < 1e-19
    assert backward_error(matrix, rhs, wrong) > 1e-2


def test_empty_equations_are_ignored() -> None:
    """A row that is zero on both sides carries no information and must not count."""
    matrix = sps.csr_matrix(np.diag([1.0, 0.0]))
    rhs = np.array([1.0, 0.0])
    assert backward_error(matrix, rhs, np.array([1.0, 12.0])) == 0.0


@pytest.mark.parametrize("backend", ["superlu", "pypardiso"])
def test_porepy_backends_are_named_after_the_solver(backend) -> None:
    """The name identifies the factorisation, not the module it is reached through."""
    solver = build_linear_solver(backend, row_equilibration=False, options={})
    assert solver.delegate.backend == POREPY_BACKENDS[backend]


def test_a_silently_substituted_backend_is_rejected() -> None:
    """PorePy falls back when a backend's library is missing; a run must not inherit
    that substitution unnoticed, or its results are attributed to the wrong solver."""
    with pytest.raises(ValueError, match="fell back"):
        _reject_substituted_backend("umfpack", "scipy_sparse")


def test_an_unknown_backend_is_rejected_by_name() -> None:
    """A silent fallback would attribute a run's behaviour to the wrong solver."""
    with pytest.raises(ValueError, match="scipy_sparse"):
        build_linear_solver("scipy_sparse", row_equilibration=False, options={})
