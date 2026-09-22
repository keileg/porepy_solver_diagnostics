"""Solver experiments on PorePy's thermo-poromechanics cases with fracture contact.

The package holds the case definitions, the run harness and the metrics collection used
to characterise where PorePy's nonlinear and linear solvers struggle on the geothermal
reservoir example.

"""

from .case import CaseSpec, SolverSpec, build_model, build_solver  # noqa: F401
from .metrics import RunMetrics, collect_metrics  # noqa: F401
from .runner import run_case  # noqa: F401
