"""Wall-clock breakdown of a run.

Method comparisons trade assembly against iterations against linear solves, so a total
run time alone cannot say whether fewer iterations meant less work. The breakdown is
taken by wrapping `EquationSystem.assemble`, which is the single entry point PorePy
uses both for Jacobian assembly (once per nonlinear iteration) and for residual-only
evaluations (once per line-search sample and once per convergence check).

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AssemblyTimers:
    """Counts and wall-clock times of the two kinds of assembly."""

    jacobian_time: float = 0.0
    jacobian_count: int = 0
    residual_time: float = 0.0
    residual_count: int = 0
    prepare_time: float = 0.0
    run_time: float = 0.0
    linear_solve_time: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def other_time(self) -> float:
        """Run time not spent assembling or solving linear systems."""
        return (
            self.run_time
            - self.jacobian_time
            - self.residual_time
            - self.linear_solve_time
        )

    def summary(self) -> dict[str, Any]:
        """Return the breakdown as plain data for a metrics file."""
        return {
            "prepare_time": self.prepare_time,
            "run_time": self.run_time,
            "jacobian_assembly_time": self.jacobian_time,
            "jacobian_assembly_count": self.jacobian_count,
            "residual_evaluation_time": self.residual_time,
            "residual_evaluation_count": self.residual_count,
            "linear_solve_time": self.linear_solve_time,
            "other_time": self.other_time,
            **self.extra,
        }


def instrument_assembly(model, timers: AssemblyTimers) -> None:
    """Time every `assemble` call the model's equation system makes.

    Jacobian assemblies and residual-only evaluations are distinguished by the
    `evaluate_jacobian` argument, whose default is True.

    Parameters:
        model: The model whose equation system is instrumented, in place.
        timers: Where the times and counts are accumulated.

    """
    equation_system = model.equation_system
    original_assemble = equation_system.assemble

    def timed_assemble(*args, **kwargs):
        # `evaluate_jacobian` is the first argument, and every call site in PorePy
        # passes it by keyword; accept both spellings anyway.
        evaluate_jacobian = args[0] if args else kwargs.get("evaluate_jacobian", True)
        t_0 = time.time()
        result = original_assemble(*args, **kwargs)
        elapsed = time.time() - t_0
        if evaluate_jacobian:
            timers.jacobian_time += elapsed
            timers.jacobian_count += 1
        else:
            timers.residual_time += elapsed
            timers.residual_count += 1
        return result

    equation_system.assemble = timed_assemble  # type: ignore[method-assign]


def count_capped_dofs(nonlinear_solver, timers: AssemblyTimers) -> None:
    """Count the dofs whose Newton increment is capped to their variable bounds.

    Capping changes the Newton direction, so how often and how widely it acts is
    recorded rather than left to be inferred from the iteration counts.

    Parameters:
        nonlinear_solver: The Newton solver whose capping is instrumented, in place.
        timers: Where the counts are accumulated, under ``capped_dofs``.

    """
    counts: list[int] = []
    timers.extra["capped_dofs"] = counts
    original_cap = nonlinear_solver.cap_increment_to_bounds

    def counted_cap(model, increment):
        capped = original_cap(model, increment)
        counts.append(int(np.count_nonzero(capped != increment)))
        return capped

    nonlinear_solver.cap_increment_to_bounds = counted_cap
