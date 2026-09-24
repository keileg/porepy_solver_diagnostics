"""Run a one-axis-at-a-time sweep around the baseline, several points in parallel.

Every point differs from the baseline in exactly one case parameter and is run with the
same solver configuration, the baseline included, so that points run side by side are
compared under the same load. Each point runs in a fresh interpreter and in its own
working directory: the model writes its gmsh files to the working directory, and two
runs meshing different geometries into one directory would silently swap meshes.

Example:
    python scripts/run_sweep.py --name sweep_1 --workers 4

"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from solver_experiments import CaseSpec, SolverSpec, run_case

RESULTS = Path(__file__).resolve().parents[1] / "results"

REFERENCE_SOLVER = SolverSpec(
    linear_backend="petsc_iterative",
    dt_min=1.0,
    convergence_criteria="blockwise",
    cap_to_bounds=True,
)
"""The configuration all points are run with: the one that passes the baseline with no
failed attempt, so that a failure at a sweep point is not inherited from it."""

SWEEP_1: list[dict[str, Any]] = [
    {},
    # Cooler injection raises the thermal stress at the injector. Kept well above the
    # 273.16 K bound, since the discrete solution undershoots the injection temperature
    # by several kelvin and a solution outside the bounds cannot be converged to.
    {"injection_temperature": 320.0},
    {"injection_temperature": 310.0},
    {"injection_temperature": 290.0},
    {"injection_temperature": 285.0},
    {"injection_pressure_scale": 0.9},
    {"injection_pressure_scale": 1.1},
    {"injection_pressure_scale": 1.2},
    {"injection_pressure_scale": 1.3},
    # The x/z ratio resolves shear onto the fractures, which dip about the y axis.
    {"lithostatic_multipliers": (0.7, 1.2, 1.0)},
    {"lithostatic_multipliers": (0.75, 1.2, 1.0)},
    {"lithostatic_multipliers": (0.85, 1.2, 1.0)},
    {"lithostatic_multipliers": (0.9, 1.2, 1.0)},
    {"dilation_angle": 0.05},
    {"dilation_angle": 0.15},
    {"dilation_angle": 0.2},
    # Fracture mesh first, then the matrix, since refinement is expensive.
    {"cell_size_fracture": 64.0},
    {"cell_size_fracture": 48.0},
    {"cell_size": 160.0},
]
"""First sweep: small steps along each axis, in Eirik's priority order."""

SWEEPS = {"sweep_1": SWEEP_1}


def run_point(overrides: dict[str, Any], sweep_directory: Path) -> tuple[str, bool]:
    """Run one point in its own working directory, logging to a file there."""
    case = CaseSpec(**overrides)
    run_directory = sweep_directory / f"{case.case_id}__{REFERENCE_SOLVER.solver_id}"
    run_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(run_directory)
    logging.basicConfig(
        filename=run_directory / "run.log",
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    metrics = run_case(case, REFERENCE_SOLVER, run_directory)
    return case.case_id, metrics.as_dict()["success"]


def main() -> None:
    """Run every point of the named sweep and print each outcome as it arrives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(SWEEPS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()

    sweep_directory = RESULTS / arguments.name
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=arguments.workers, mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = [
            pool.submit(run_point, overrides, sweep_directory)
            for overrides in SWEEPS[arguments.name]
        ]
        for future in as_completed(futures):
            case_id, success = future.result()
            print(f"{case_id}: {'success' if success else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
