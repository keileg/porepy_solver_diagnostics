"""Linear solvers for the experiments, and instrumentation around them.

PorePy ships `LinearSolverDirect` with a pypardiso, umfpack or scipy (SuperLU)
backend. Two additions are needed here:

* The assembled thermo-poromechanics system is very badly row-scaled (entries spanning
  tens of orders of magnitude, and almost every diagonal entry zero), and pypardiso
  silently returns a solution with a relative residual of order one on it.
  :class:`RowEquilibratedSolver` makes that measurable instead of invisible by scaling
  each row by its largest entry before the solve.
* :class:`PetscDirectSolver` reaches the MUMPS and UMFPACK factorisations built into
  the local PETSc, which are both faster than SuperLU on these systems.

:class:`InstrumentedSolver` wraps any of them and records the true relative residual
of every solve, so a backend that returns a wrong answer cannot be mistaken for a
nonlinear-solver failure.

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
    relative_residuals: list[float] = field(default_factory=list)
    num_failures: int = 0

    def summary(self) -> dict[str, Any]:
        """Return the aggregate statistics recorded so far."""
        if not self.solve_times:
            return {"num_solves": 0}
        return {
            "num_solves": len(self.solve_times),
            "total_solve_time": float(np.sum(self.solve_times)),
            "mean_solve_time": float(np.mean(self.solve_times)),
            "max_relative_residual": float(np.max(self.relative_residuals)),
            "median_relative_residual": float(np.median(self.relative_residuals)),
            "num_failures": self.num_failures,
        }


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
        """Solve, then record the time taken and the true relative residual."""
        matrix, rhs = linear_system.matrix, linear_system.rhs
        t_0 = time.time()
        solution, status = self.delegate.solve_linear_system(linear_system)
        self.record.solve_times.append(time.time() - t_0)
        if status.is_failure():
            self.record.num_failures += 1
            self.record.relative_residuals.append(np.inf)
        else:
            assert matrix is not None
            scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
            residual = float(np.linalg.norm(matrix @ solution - rhs)) / scale
            self.record.relative_residuals.append(residual)
        logger.info(
            "Linear solve: %.2e s, relative residual %.2e",
            self.record.solve_times[-1],
            self.record.relative_residuals[-1],
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


def build_linear_solver(
    backend: str, row_equilibration: bool, options: dict[str, Any]
) -> InstrumentedSolver:
    """Build the linear solver named by a :class:`~solver_experiments.case.SolverSpec`.

    Parameters:
        backend: ``"scipy_sparse"``, ``"pypardiso"`` or ``"umfpack"`` for PorePy's own
            direct solver, ``"petsc_mumps"`` / ``"petsc_umfpack"`` for PETSc's, or
            ``"petsc_iterative"`` for the `pp_solvers` PETSc bridge.
        row_equilibration: Whether to row-equilibrate before solving.
        options: Solver options, forwarded to the iterative solver only.

    Returns:
        The solver, wrapped in :class:`InstrumentedSolver`.

    """
    solver: pp.solvers.LinearSolverBase
    if backend.startswith("petsc_") and backend != "petsc_iterative":
        solver = PetscDirectSolver(factor_solver_type=backend.removeprefix("petsc_"))
    elif backend == "petsc_iterative":
        import pp_solvers

        solver = pp_solvers.IterativeLinearSolver(
            configuration_factory=pp_solvers.thm_factory, solver_options=options
        )
    else:
        solver = pp.solvers.LinearSolverDirect(backend=backend)  # type: ignore[arg-type]
    if row_equilibration:
        solver = RowEquilibratedSolver(solver)
    return InstrumentedSolver(solver)
