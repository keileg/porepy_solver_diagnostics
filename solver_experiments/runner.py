"""Run one (case, solver configuration) pair and write its metrics.

One run produces three files in its own directory: `metrics.json` (everything the
comparison tool reads), `telemetry.jsonl` (PorePy's own per-iteration records) and
`run.log` (the console output, kept because a crash leaves its traceback only there).

"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp

from .case import (
    CaseSpec,
    SolverSpec,
    build_case_linear_solver,
    build_model,
    build_solver,
)
from .instrumentation import AssemblyTimers, instrument_assembly
from .metrics import RunMetrics, collect_metrics

logger = logging.getLogger(__name__)

ADMISSIBILITY_BOUNDS = {
    "pressure": (0.0, 1.0e9),
    "temperature": (273.16, 647.0),
}
"""Bounds outside which an iterate is not physical for this case.

Pressure must stay positive, and the water property correlations are formulated for
liquid water, so its triple point and critical temperature bound the temperature. The
counts are diagnostic only: nothing in the solver acts on them.

"""


def run_case(
    case: CaseSpec,
    solver: SolverSpec,
    output_directory: Path,
) -> RunMetrics:
    """Run one case with one solver configuration and write its metrics.

    Parameters:
        case: The case to run.
        solver: The solver configuration to run it with.
        output_directory: Directory for this run's three output files. Created if
            missing; existing files are overwritten.

    Returns:
        The metrics of the run, also written to `metrics.json`.

    """
    output_directory.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_directory / "telemetry.jsonl"

    timers = AssemblyTimers()
    model = build_model(case, solver, folder_name=str(output_directory / "viz"))

    t_0 = time.time()
    model.prepare_simulation()
    timers.prepare_time = time.time() - t_0
    timers.extra = _model_size(model)

    instrument_assembly(model, timers)
    recorder = _build_recorder(solver, telemetry_path)
    linear_solver = build_case_linear_solver(solver)
    nonlinear_solver = build_solver(model, solver, recorder, linear_solver)
    runner = pp.ModelRunner(
        model,
        _runner_params(solver),
        time_stepper=_time_stepper(model, solver, recorder),
        nonlinear_solver=nonlinear_solver,
    )

    success, failure_reason = True, None
    t_0 = time.time()
    try:
        runner.run()
    except Exception as error:  # noqa: BLE001 - any failure is a data point, not a bug
        success, failure_reason = False, f"{type(error).__name__}: {error}"
        logger.warning("Run %s/%s failed: %s", case.case_id, solver.solver_id, error)
    timers.run_time = time.time() - t_0
    timers.linear_solve_time = sum(linear_solver.record.solve_times)
    timers.extra["simulated_time"] = model.time_manager.time
    timers.extra["end_time"] = model.time_manager.schedule.t_end

    metrics = collect_metrics(
        case_id=case.case_id,
        solver_id=solver.solver_id,
        case=_as_data(case),
        solver=_as_data(solver),
        success=success,
        failure_reason=failure_reason,
        timing=timers.summary(),
        linear_solver=linear_solver.record.summary(),
        telemetry_path=telemetry_path if recorder is not None else None,
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2, sort_keys=True, default=_json_default)
    )
    return metrics


def _build_recorder(solver: SolverSpec, telemetry_path: Path):
    """Build the telemetry recorder, or None if telemetry is switched off."""
    if solver.telemetry_verbosity == 0:
        return None
    return pp.solvers.NonlinearTelemetryRecorder(
        pp.solvers.TelemetryConfig(
            output_path=telemetry_path,
            verbosity=solver.telemetry_verbosity,
            admissibility_bounds=ADMISSIBILITY_BOUNDS,
        )
    )


def _time_stepper(model, solver: SolverSpec, recorder):
    """Build the time stepper, so that it too reports to the telemetry recorder."""
    return pp.time_stepper.TimeStepper(
        scheduler=pp.time_stepper.assemble_default_time_scheduler(
            time_manager=model.time_manager
        ),
        max_attempts=solver.max_attempts,
        telemetry=recorder,
    )


def _runner_params(solver: SolverSpec) -> dict[str, Any]:
    """Parameters for `ModelRunner` itself, which must not name a nonlinear solver."""
    from .case import build_solver_params

    return build_solver_params(solver)


def _model_size(model) -> dict[str, Any]:
    """Record the discrete size of the problem, which sets the cost of every solve."""
    mdg = model.mdg
    return {
        "num_dofs": int(model.equation_system.num_dofs()),
        "num_cells_by_dimension": {
            str(dim): int(sum(sd.num_cells for sd in mdg.subdomains(dim=dim)))
            for dim in range(mdg.dim_max(), -1, -1)
        },
        "num_mortar_cells": int(sum(intf.num_cells for intf in mdg.interfaces())),
    }


def _json_default(value: Any) -> Any:
    """Convert numpy scalars and arrays, which `json` does not handle, to plain data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable.")


def _as_data(spec: CaseSpec | SolverSpec) -> dict[str, Any]:
    """Return a spec as plain JSON-serialisable data."""
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in vars(spec).items()
    }
