"""Tests of the row-scaled convergence measure.

A failure here means the campaign's convergence test is measuring something other than
what it claims, which would make every run's iteration count incomparable.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sps

from solver_experiments.convergence import RowScaledEquationMetric


@dataclass(eq=False)
class _Equation:
    """Hashed by identity, as PorePy's equations are: the same equation on two
    subdomains is two entries of the indexer."""

    name: str


@dataclass
class _Indexer:
    indices: dict


@dataclass
class _LinearSystem:
    matrix: sps.spmatrix | None
    equation_indexer: _Indexer


class _EquationSystem:
    """Returns a fixed system, and records that it was wrapped."""

    def __init__(self, linear_system) -> None:
        self.linear_system = linear_system
        self.calls = 0

    def assemble(self, *args, **kwargs):
        self.calls += 1
        return self.linear_system


class _Model:
    def __init__(self, equation_system) -> None:
        self.equation_system = equation_system


def _model_with(matrix, blocks):
    system = _LinearSystem(
        matrix,
        _Indexer({_Equation(name): indices for name, indices in blocks.items()}),
    )
    return _Model(_EquationSystem(system))


def test_blocks_are_compared_in_solution_units() -> None:
    """Two equations stating the same thing in units a thousand apart must give the
    same measure, which is the whole point of the scaling."""
    matrix = sps.diags([1.0, 1000.0]).tocsr()
    model = _model_with(matrix, {"cheap": [0], "expensive": [1]})
    metric = RowScaledEquationMetric(model)
    model.equation_system.assemble()

    measures = metric(np.array([2.0, 2000.0]))
    assert measures["cheap"] == pytest.approx(2.0)
    assert measures["expensive"] == pytest.approx(2.0)


def test_a_block_spread_over_several_grids_is_one_measure() -> None:
    """An equation defined on several subdomains appears once per subdomain in the
    indexer; reporting it once per subdomain would test the same equation twice."""
    matrix = sps.identity(4, format="csr")
    system = _LinearSystem(
        matrix,
        _Indexer(
            {
                _Equation("mass_balance"): [0, 1],
                _Equation("mass_balance"): [2, 3],
            }
        ),
    )
    model = _Model(_EquationSystem(system))
    metric = RowScaledEquationMetric(model)
    model.equation_system.assemble()

    measures = metric(np.array([1.0, 1.0, 3.0, 3.0]))
    assert set(measures) == {"mass_balance"}
    assert measures["mass_balance"] == pytest.approx(np.sqrt(5.0))


def test_residual_only_assemblies_do_not_disturb_the_scales() -> None:
    """The convergence check evaluates the residual without a Jacobian, and must not
    be taken as a new set of row scales."""
    matrix = sps.diags([1.0, 1000.0]).tocsr()
    model = _model_with(matrix, {"cheap": [0], "expensive": [1]})
    metric = RowScaledEquationMetric(model)
    model.equation_system.assemble()

    model.equation_system.linear_system = np.array([1.0, 1.0])
    model.equation_system.assemble(evaluate_jacobian=False)

    assert metric(np.array([2.0, 2000.0]))["expensive"] == pytest.approx(2.0)


def test_a_metric_without_a_jacobian_refuses_to_answer() -> None:
    """Silently falling back on unscaled residuals would reinstate the very test the
    scaling replaces."""
    model = _model_with(sps.identity(2, format="csr"), {"all": [0, 1]})
    metric = RowScaledEquationMetric(model)

    with pytest.raises(RuntimeError, match="no Jacobian"):
        metric(np.ones(2))
