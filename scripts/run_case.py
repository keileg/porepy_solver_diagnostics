"""Run one case, given as command-line overrides of the shipped configuration.

Example:
    python scripts/run_case.py --injection-temperature 280 --schedule-length 3

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from solver_experiments import CaseSpec, SolverSpec, run_case

RESULTS = Path(__file__).resolve().parents[1] / "results"


def parse_arguments() -> argparse.Namespace:
    """Define the case and solver overrides available from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injection-temperature", type=float)
    parser.add_argument("--injection-pressure-scale", type=float)
    parser.add_argument("--lithostatic-multipliers", type=float, nargs=3)
    parser.add_argument("--dilation-angle", type=float)
    parser.add_argument("--cell-size", type=float)
    parser.add_argument("--cell-size-fracture", type=float)
    parser.add_argument("--schedule-length", type=int)
    parser.add_argument("--end-time-fraction", type=float)
    parser.add_argument("--linear-backend")
    parser.add_argument("--row-equilibration", action="store_true")
    parser.add_argument("--nonlinear-solver")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--dt-min", type=float)
    parser.add_argument("--results", type=Path, default=RESULTS)
    return parser.parse_args()


def build_specs(arguments: argparse.Namespace) -> tuple[CaseSpec, SolverSpec]:
    """Build the case and solver specs from the given overrides."""
    given = {
        name: value
        for name, value in vars(arguments).items()
        if value not in (None, False)
    }
    case_fields = set(CaseSpec.__dataclass_fields__)
    solver_fields = set(SolverSpec.__dataclass_fields__)
    case_arguments = {
        name: tuple(value) if name == "lithostatic_multipliers" else value
        for name, value in given.items()
        if name in case_fields
    }
    solver_arguments = {
        name: value for name, value in given.items() if name in solver_fields
    }
    return CaseSpec(**case_arguments), SolverSpec(**solver_arguments)


def main() -> None:
    """Run one case and print its outcome."""
    logging.basicConfig(level=logging.INFO, force=True)
    arguments = parse_arguments()
    case, solver = build_specs(arguments)
    directory = arguments.results / f"{case.case_id}__{solver.solver_id}"
    metrics = run_case(case, solver, directory)
    print(
        f"{case.case_id} / {solver.solver_id}: "
        f"{'success' if metrics.success else 'FAILED'} "
        f"in {metrics.timing['run_time']:.0f} s "
        f"({metrics.time_steps.get('num_accepted', 0)} steps, "
        f"{metrics.iterations.get('total', 0)} iterations)"
    )
    if not metrics.success:
        print(f"  reason: {metrics.failure_reason}")


if __name__ == "__main__":
    main()
