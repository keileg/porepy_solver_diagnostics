"""Convergence measures that do not depend on how an equation happens to be scaled.

PorePy's shipped criterion for this case is a single absolute tolerance on the
root-mean-square of the whole residual vector. The blocks of that vector are fifteen
orders of magnitude apart, so the test is decided by the largest-scaled block alone:
measured over a restarted run of the injection interval, the contact mechanics and
interface flux equations sit below any usable absolute tolerance from the first
iteration and are never tested at all.

Dividing each residual entry by the largest magnitude in its Jacobian row states the
residual as the perturbation of the solution that would remove it. That is the same
unit for every equation, so one tolerance means the same thing in all blocks.

"""

from __future__ import annotations

import numpy as np
import porepy as pp


class RowScaledEquationMetric(pp.Metric):
    """Residual of each equation block, measured in solution units.

    The row scales are taken from the Jacobian the nonlinear solver last assembled,
    picked up by wrapping the equation system's `assemble`. They are not recomputed
    for the convergence check itself, which evaluates the residual alone.

    Parameters:
        model: A prepared model, whose equation system is wrapped on construction.

    """

    def __init__(self, model) -> None:
        self.model = model
        self._row_scale: np.ndarray | None = None
        self._blocks: dict[str, np.ndarray] = {}
        self._install()

    def _install(self) -> None:
        """Watch every Jacobian the equation system assembles."""
        equation_system = self.model.equation_system
        original = equation_system.assemble

        def assemble(*args, **kwargs):
            assembled = original(*args, **kwargs)
            # A residual-only evaluation returns the residual itself rather than a
            # linear system, and carries no row scales.
            if not isinstance(assembled, np.ndarray):
                self._observe(assembled)
            return assembled

        equation_system.assemble = assemble

    def _observe(self, linear_system) -> None:
        """Record the row scales and block structure of an assembled Jacobian."""
        matrix = abs(linear_system.matrix.tocsr())
        self._row_scale = np.maximum(matrix.max(axis=1).toarray().ravel(), 1e-300)
        blocks: dict[str, list[np.ndarray]] = {}
        for equation, indices in linear_system.equation_indexer.indices.items():
            blocks.setdefault(equation.name, []).append(np.asarray(indices))
        self._blocks = {name: np.concatenate(parts) for name, parts in blocks.items()}

    def __call__(self, values: np.ndarray) -> dict[str, float]:
        """Return the row-scaled residual of each equation block.

        Parameters:
            values: Algebraic representation of a mixed-dimensional equation.

        Returns:
            The root-mean-square of the scaled entries, per equation block.

        """
        if self._row_scale is None:
            raise RuntimeError(
                "The metric has no row scales yet: no Jacobian has been assembled."
            )
        return {
            name: float(
                np.sqrt(np.mean((values[indices] / self._row_scale[indices]) ** 2))
            )
            for name, indices in self._blocks.items()
        }
