"""Linear solvers for the experiments, and instrumentation around them.

PorePy ships `LinearSolverDirect` with a pypardiso, umfpack or scipy (SuperLU)
backend. Two additions are needed here:

* The assembled thermo-poromechanics system is very badly row-scaled (entries spanning
  tens of orders of magnitude, and almost every diagonal entry zero), and pypardiso
  silently returns a solution with a relative residual of order one on it.
  :class:`RowEquilibratedSolver` makes that measurable instead of invisible by scaling
  each row by its largest entry before the solve.
* :class:`PetscDirectSolver` reaches the MUMPS and UMFPACK factorisations built into
  the local PETSc.

:class:`InstrumentedSolver` wraps any of them and records how trustworthy every solve
was, so a backend that returns a wrong answer cannot be mistaken for a nonlinear-solver
failure. Two numbers are needed for that, because no single one suffices on this system:

* a *backward error*, the residual of each equation measured against the size of the
  terms that form it. The plain relative residual ``||Ax-b|| / ||b||`` is useless here,
  since a system whose rows span twenty orders of magnitude lets a large row's rounding
  error swamp every other equation;
* the *infinity norm of the solution*. The Jacobian has unknowns — fracture-intersection
  temperatures above all — whose columns are some twelve orders of magnitude weaker than
  its largest entries, hence numerically null. A factorisation may then return an
  enormous component along that near-null direction while leaving the residual small, so
  the backward error cannot see it but the update norm can.

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import porepy as pp
import scipy.sparse as sps

logger = logging.getLogger(__name__)


@dataclass
class SolveRecord:
    """Per-solve statistics accumulated by :class:`InstrumentedSolver`."""

    solve_times: list[float] = field(default_factory=list)
    backward_errors: list[float] = field(default_factory=list)
    update_norms: list[float] = field(default_factory=list)
    num_failures: int = 0

    def summary(self) -> dict[str, Any]:
        """Return the aggregate statistics recorded so far."""
        if not self.solve_times:
            return {"num_solves": 0}
        return {
            "num_solves": len(self.solve_times),
            "total_solve_time": float(np.sum(self.solve_times)),
            "mean_solve_time": float(np.mean(self.solve_times)),
            "max_backward_error": float(np.max(self.backward_errors)),
            "median_backward_error": float(np.median(self.backward_errors)),
            "max_update_norm": float(np.max(self.update_norms)),
            "num_failures": self.num_failures,
        }


def backward_error(
    matrix: sps.spmatrix, rhs: np.ndarray, solution: np.ndarray
) -> float:
    """Return the largest residual relative to the terms forming its own equation.

    Parameters:
        matrix: The solved system's matrix.
        rhs: The solved system's right-hand side.
        solution: The solution returned by the solver.

    Returns:
        ``max_i |r_i| / (max_j |a_ij| * ||x||_inf + |b_i|)``, which is of the order of
        machine precision for a solve that can be trusted, irrespective of how the rows
        are scaled relative to each other.

    """
    residual = np.abs(matrix @ solution - rhs)
    row_max = abs(matrix).max(axis=1).toarray().ravel()
    equation_scale = row_max * np.max(np.abs(solution)) + np.abs(rhs)
    significant = equation_scale > 0
    if not significant.any():
        return 0.0
    return float(np.max(residual[significant] / equation_scale[significant]))


class InstrumentedSolver(pp.solvers.LinearSolverBase):
    """Time every solve of a wrapped solver and check its true relative residual.

    Parameters:
        delegate: The solver doing the actual work.

    """

    def __init__(self, delegate: pp.solvers.LinearSolverBase) -> None:
        self.delegate = delegate
        self.record = SolveRecord()

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        """Forward model-dependent initialisation to the wrapped solver."""
        self.delegate.initialize_with_model(model)

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        """Solve, then record the time taken, the backward error and the update norm."""
        matrix, rhs = linear_system.matrix, linear_system.rhs
        t_0 = time.time()
        solution, status = self.delegate.solve_linear_system(linear_system)
        self.record.solve_times.append(time.time() - t_0)
        if status.is_failure():
            self.record.num_failures += 1
            self.record.backward_errors.append(np.inf)
            self.record.update_norms.append(np.inf)
        else:
            assert matrix is not None
            self.record.backward_errors.append(backward_error(matrix, rhs, solution))
            self.record.update_norms.append(float(np.max(np.abs(solution))))
        logger.info(
            "Linear solve: %.2e s, backward error %.2e, ||update||_inf %.2e",
            self.record.solve_times[-1],
            self.record.backward_errors[-1],
            self.record.update_norms[-1],
        )
        return solution, status


class RowEquilibratedSolver(pp.solvers.LinearSolverBase):
    """Scale each row of the system by its largest entry, then delegate.

    The scaling is a similarity transform of the right-hand side only, so the returned
    solution solves the original system: row `i` of both matrix and right-hand side is
    divided by `max_j |a_ij|`.

    Parameters:
        delegate: The solver doing the actual work.

    """

    def __init__(self, delegate: pp.solvers.LinearSolverBase) -> None:
        self.delegate = delegate

    def initialize_with_model(self, model: pp.PorePyModel) -> None:
        """Forward model-dependent initialisation to the wrapped solver."""
        self.delegate.initialize_with_model(model)

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        """Solve the row-equilibrated system with the wrapped solver."""
        matrix = linear_system.matrix
        assert matrix is not None
        row_max = abs(matrix).max(axis=1).toarray().ravel()
        scaling = sps.diags(1.0 / np.maximum(row_max, np.finfo(float).tiny))
        scaled = pp.solvers.LinearSystem(
            matrix=(scaling @ matrix).tocsr(),
            rhs=scaling @ linear_system.rhs,
            equation_indexer=linear_system.equation_indexer,
            variable_indexer=linear_system.variable_indexer,
        )
        return self.delegate.solve_linear_system(scaled)


class PetscDirectSolver(pp.solvers.LinearSolverBase):
    """Direct LU solve through PETSc, e.g. with MUMPS or UMFPACK.

    Parameters:
        factor_solver_type: PETSc factor solver type, such as ``"mumps"`` or
            ``"umfpack"``.

    """

    def __init__(self, factor_solver_type: str = "umfpack") -> None:
        self.factor_solver_type = factor_solver_type

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        """Factorise and solve the system with PETSc's direct solver."""
        from petsc4py import PETSc

        matrix = linear_system.matrix
        assert matrix is not None
        csr = matrix.tocsr()
        t_0 = time.time()
        petsc_matrix = PETSc.Mat().createAIJ(
            size=csr.shape, csr=(csr.indptr, csr.indices, csr.data)
        )
        petsc_matrix.assemble()
        rhs = PETSc.Vec().createWithArray(linear_system.rhs.copy())
        solution = rhs.duplicate()
        ksp = PETSc.KSP().create()
        ksp.setOperators(petsc_matrix)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType(self.factor_solver_type)
        try:
            ksp.setUp()
            ksp.solve(rhs, solution)
            values = solution.getArray().copy()
        except PETSc.Error as error:
            return np.zeros_like(linear_system.rhs), (
                pp.solvers.LinearSolverStatusFailure(reason=str(error))
            )
        finally:
            # PETSc objects outlive their Python handles, and one factorisation of this
            # system is a gigabyte or so: without this, a run accumulates one per
            # nonlinear iteration.
            for petsc_object in (ksp, petsc_matrix, rhs, solution):
                petsc_object.destroy()
        return values, pp.solvers.LinearSolverStatusSuccess(
            solve_time=time.time() - t_0
        )


POREPY_BACKENDS: dict[str, str] = {
    "superlu": "scipy_sparse",
    "pypardiso": "pypardiso",
    "umfpack": "umfpack",
}
"""Solvers reachable through PorePy's own `LinearSolverDirect`, by the name of the
factorisation rather than of the module it is reached through: `scipy_sparse` names
SciPy's binding, but the solver doing the work is SuperLU.

"""

PETSC_BACKENDS: dict[str, str] = {
    "mumps": "mumps",
    "umfpack_petsc": "umfpack",
}
"""Solvers reachable through PETSc's `KSP`, keyed the same way. UMFPACK appears twice
because it is reachable through either portal, and the two differ in cost.

"""


def _reject_substituted_backend(requested: str, actual: str) -> None:
    """Fail unless PorePy built the factorisation that was asked for.

    `LinearSolverDirect` falls back silently when a backend's library is missing:
    asking for `umfpack` without scikit-umfpack installed yields SuperLU. A run
    attributed to the wrong solver is worse than a run that does not start.

    Parameters:
        requested: The backend name given to :func:`build_linear_solver`.
        actual: The backend the constructed solver reports.

    Raises:
        ValueError: If the two disagree.

    """
    expected = POREPY_BACKENDS[requested]
    if actual != expected:
        raise ValueError(
            f"Asked for {requested!r} ({expected!r}) but PorePy fell back to "
            f"{actual!r}; install the missing library or choose another backend."
        )


def build_linear_solver(
    backend: str, row_equilibration: bool, options: dict[str, Any]
) -> InstrumentedSolver:
    """Build the linear solver named by a :class:`~solver_experiments.case.SolverSpec`.

    Parameters:
        backend: The factorisation to use — a key of :data:`POREPY_BACKENDS` or
            :data:`PETSC_BACKENDS`, ``"petsc_iterative"`` for the `pp_solvers` PETSc
            bridge with ``thm_factory``, or ``"petsc_system_amg"`` for the same bridge
            with the system-AMG configuration of :mod:`.preconditioners`.
        row_equilibration: Whether to row-equilibrate before solving.
        options: Solver options, forwarded to the iterative solver only.

    Raises:
        ValueError: If the backend is not one of the known names.

    Returns:
        The solver, wrapped in :class:`InstrumentedSolver`.

    """
    solver: pp.solvers.LinearSolverBase
    if backend in POREPY_BACKENDS:
        solver = pp.solvers.LinearSolverDirect(
            backend=POREPY_BACKENDS[backend]  # type: ignore[arg-type]
        )
        _reject_substituted_backend(backend, solver.backend)
    elif backend in PETSC_BACKENDS:
        solver = PetscDirectSolver(factor_solver_type=PETSC_BACKENDS[backend])
    elif backend == "petsc_iterative":
        import pp_solvers

        solver = pp_solvers.IterativeLinearSolver(
            configuration_factory=pp_solvers.thm_factory, solver_options=options
        )
    elif backend == "petsc_system_amg":
        from .preconditioners import system_amg_solver

        solver = system_amg_solver(options)
    else:
        known = sorted(
            [*POREPY_BACKENDS, *PETSC_BACKENDS, "petsc_iterative", "petsc_system_amg"]
        )
        raise ValueError(f"Unknown linear solver backend {backend!r}; known: {known}")
    if row_equilibration:
        solver = RowEquilibratedSolver(solver)
    return InstrumentedSolver(solver)
