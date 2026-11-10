"""Run a Coso runscript unchanged, instrumented through two hooks.

The runscripts build their model, schedule and nonlinear solver inside their own
``__main__`` block, so they cannot be driven through ``CaseSpec``. Instead, each run
executes the script itself, from a snapshot of the Coso repository at a pinned commit,
after two hooks have been installed:

- PorePy's ``initialization_pipeline``, which the script calls first;
- ``pp.ModelRunner``, which the script then uses for the main run.

Each hook instruments the nonlinear solver the script built, the same way
:func:`solver_experiments.run_case` instruments its own:
- per-solve backward error and update norm (:class:`InstrumentedSolver`);
- telemetry;
- capped-dof counts;
- assembly timing, for the main run.

A variant can also swap the linear solver, switch on the temperature and pressure cap,
stop the main run early, or stop after meshing. Nothing else about the script changes,
so a variant differs from the original by exactly the options it names.

Example:
    scripts/campaign_env.sh python cases/coso/driver.py runscript_3 results/coso/rs3

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import runpy
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp
import porepy.applications.initialization.initialization as initialization_module

from solver_experiments.case import ADMISSIBILITY_BOUNDS
from solver_experiments.instrumentation import (
    AssemblyTimers,
    count_capped_dofs,
    instrument_assembly,
)
from solver_experiments.linear_solvers import InstrumentedSolver, build_linear_solver
from solver_experiments.metrics import collect_metrics

logger = logging.getLogger(__name__)

COSO_REPOSITORY = Path("/workdir/Coso")
COSO_COMMIT = "2f95e65"
"""Head of branch ``october-attempt`` when the case was taken on (2026-11-10)."""

LINEAR_SOLVERS = ("as_given", "pypardiso", "thm_factory", "system_amg")


@dataclass
class Variant:
    """What a run changes relative to the runscript as given.

    Attributes:
        script: Name of the runscript, e.g. ``"runscript_3"``.
        linear_solver: ``"as_given"`` keeps the script's own linear solver;
            ``"pypardiso"`` replaces it by the direct solver; ``"thm_factory"`` and
            ``"system_amg"`` select one of the two PETSc preconditioners the scripts
            use, with the scripts' own PETSc options.
        cap: Whether to cap temperature and pressure to :data:`ADMISSIBILITY_BOUNDS`,
            in the initialisation and the main run alike.
        stop_after: Simulated time [s] after which the main run stops, or None to run
            the script's whole schedule.
        mesh_only: Stop after preparing the model, having recorded its size and
            well-fracture connections.

    """

    script: str
    linear_solver: str = "as_given"
    cap: bool = False
    stop_after: float | None = None
    mesh_only: bool = False


@dataclass
class RunRecord:
    """Everything a run records besides the per-phase metrics files."""

    variant: dict[str, Any]
    coso_commit: str = COSO_COMMIT
    model_size: dict[str, Any] = field(default_factory=dict)
    wells: list[dict[str, Any]] = field(default_factory=list)
    initialization_time: float | None = None
    main_status: str | None = None
    phases_done: list[str] = field(default_factory=list)
    peak_memory_gb: float | None = None


class _MeshOnlyDone(Exception):
    """Raised to leave the script once the prepared model has been inspected."""


def run(variant: Variant, directory: Path) -> RunRecord:
    """Run one variant of a runscript in its own directory.

    Parameters:
        variant: The runscript and what to change about it.
        directory: The run's directory; created, and must not exist already, so that
            no output of an earlier run is mixed into this one.

    Returns:
        The run record, also written to ``run.json``.

    """
    tree = _snapshot_coso(directory)
    record = RunRecord(variant=asdict(variant))
    _install_hooks(variant, directory, record)
    try:
        _execute(tree / f"{variant.script}.py", directory)
    except _MeshOnlyDone:
        pass
    finally:
        # ru_maxrss is in kilobytes on Linux.
        record.peak_memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        (directory / "run.json").write_text(
            json.dumps(asdict(record), indent=2, default=_json_default)
        )
    return record


def _snapshot_coso(directory: Path) -> Path:
    """Extract the pinned Coso tree into the run's directory and return its path.

    The script reads its data relative to its own directory and writes its output next
    to itself, so it runs from this private copy and leaves the repository untouched.

    """
    directory.mkdir(parents=True)
    tree = directory / "coso"
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(COSO_REPOSITORY), "archive", COSO_COMMIT],
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    return tree


def _execute(script: Path, directory: Path) -> None:
    """Execute the runscript as ``__main__``, as ``python <script>`` would.

    The script imports the Coso modules next to it and reads its data through
    ``sys.path[0]``, so its directory goes first on the path. The working directory is
    the run's, since gmsh writes its mesh files there.

    """
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)]
    os.chdir(directory)
    runpy.run_path(str(script), run_name="__main__")


def _install_hooks(variant: Variant, directory: Path, record: RunRecord) -> None:
    """Replace the initialisation pipeline and the model runner by instrumented ones."""
    original_pipeline = initialization_module.initialization_pipeline

    def instrumented_pipeline(model, *args, nonlinear_solver=None, **kwargs):
        if variant.cap:
            model.params["variable_bounds"] = dict(ADMISSIBILITY_BOUNDS)
        if variant.mesh_only:
            model.prepare_simulation()
            record.model_size = _model_size(model)
            record.wells = _well_connections(model)
            raise _MeshOnlyDone
        phase = _Phase("initialization", variant, directory, model)
        nonlinear_solver = phase.instrument(nonlinear_solver)
        t_0 = time.time()
        try:
            return original_pipeline(
                model, *args, nonlinear_solver=nonlinear_solver, **kwargs
            )
        finally:
            record.initialization_time = time.time() - t_0
            phase.write_metrics(success=True, failure_reason=None, timers=None)
            record.phases_done.append("initialization")

    class InstrumentedModelRunner(pp.ModelRunner):
        """Model runner whose nonlinear solver and time stepper are instrumented."""

        def __init__(
            self, model, params=None, time_stepper=None, nonlinear_solver=None
        ):
            self._phase = _Phase("main", variant, directory, model)
            nonlinear_solver = self._phase.instrument(nonlinear_solver)
            assert time_stepper is not None, "The runscripts pass their own stepper."
            time_stepper.telemetry = self._phase.recorder
            self._timers = AssemblyTimers()
            super().__init__(
                model,
                params=params,
                time_stepper=time_stepper,
                nonlinear_solver=nonlinear_solver,
                early_stop_criteria=_stop_criteria(variant),
            )
            record.model_size = _model_size(model)
            record.wells = _well_connections(model)
            instrument_assembly(model, self._timers)
            count_capped_dofs(self.solver, self._timers)

        def run(self):
            t_0 = time.time()
            status = super().run()
            self._timers.run_time = time.time() - t_0
            self._timers.extra["simulated_time"] = self.model.time_manager.time
            self._timers.extra["end_time"] = self.model.time_manager.schedule.t_end
            failure = getattr(status, "reason", None)
            record.main_status = type(status).__name__
            self._phase.write_metrics(
                success=failure is None, failure_reason=failure, timers=self._timers
            )
            record.phases_done.append("main")
            return status

    initialization_module.initialization_pipeline = instrumented_pipeline
    pp.ModelRunner = InstrumentedModelRunner  # type: ignore[misc]


class _Phase:
    """Instrumentation of one nonlinear solver, the initialisation's or the main one."""

    def __init__(self, name: str, variant: Variant, directory: Path, model) -> None:
        self.name = name
        self.variant = variant
        self.directory = directory
        self.telemetry_path = directory / f"telemetry_{name}.jsonl"
        self.recorder = pp.solvers.NonlinearTelemetryRecorder(
            pp.solvers.TelemetryConfig(
                output_path=self.telemetry_path,
                verbosity=1,
                admissibility_bounds=_bounds_in_model_units(model),
            )
        )
        self.linear_solver: InstrumentedSolver | None = None

    def instrument(self, nonlinear_solver):
        """Swap the linear solver if asked, and wrap it; attach the recorder."""
        delegate = _linear_solver(self.variant.linear_solver, nonlinear_solver)
        self.linear_solver = InstrumentedSolver(delegate)
        nonlinear_solver.linear_solver = self.linear_solver
        nonlinear_solver.telemetry = self.recorder
        return nonlinear_solver

    def write_metrics(
        self, success: bool, failure_reason: str | None, timers: AssemblyTimers | None
    ) -> None:
        """Write this phase's metrics, derived from its telemetry and solve record."""
        assert self.linear_solver is not None
        if timers is not None:
            timers.linear_solve_time = sum(self.linear_solver.record.solve_times)
        metrics = collect_metrics(
            case_id=self.variant.script,
            solver_id=self.directory.name,
            case={"script": self.variant.script, "coso_commit": COSO_COMMIT},
            solver=asdict(self.variant),
            success=success,
            failure_reason=failure_reason,
            timing={} if timers is None else timers.summary(),
            linear_solver=self.linear_solver.record.summary(),
            telemetry_path=self.telemetry_path,
            expensive_iterations=15,
        )
        path = self.directory / f"metrics_{self.name}.json"
        path.write_text(
            json.dumps(
                metrics.as_dict(), indent=2, sort_keys=True, default=_json_default
            )
        )


def _linear_solver(choice: str, nonlinear_solver) -> pp.solvers.LinearSolverBase:
    """The linear solver a variant asks for; the script's own for ``"as_given"``."""
    if choice == "as_given":
        return nonlinear_solver.linear_solver
    if choice == "pypardiso":
        return build_linear_solver("pypardiso", False, {}).delegate
    if choice in ("thm_factory", "system_amg"):
        # Imported from the script's own tree, so that the factory and the PETSc
        # options are the ones the script uses.
        import pp_solvers
        from nl_params import thm_factory_system_amg
        from solver_configurations import linear_solver_params

        factory = (
            thm_factory_system_amg if choice == "system_amg" else pp_solvers.thm_factory
        )
        return pp_solvers.IterativeLinearSolver(
            configuration_factory=factory, solver_options=linear_solver_params
        )
    raise ValueError(
        f"Unknown linear solver {choice!r}, expected one of {LINEAR_SOLVERS}."
    )


class _StopAfter(pp.models.model_runner.EarlyStopCriterion):
    """Stop the main run once the simulated time has passed a given time."""

    def __init__(self, stop_time: float) -> None:
        self.stop_time = stop_time

    def simulation_should_stop(self, model):
        if model.time_manager.time >= self.stop_time:
            return pp.models.model_runner.ModelRunnerStatusSuccess()
        return pp.models.model_runner.EarlyStopNotRequested()


def _stop_criteria(variant: Variant) -> list:
    return [] if variant.stop_after is None else [_StopAfter(variant.stop_after)]


def _bounds_in_model_units(model) -> dict[str, tuple[float, float]]:
    """:data:`ADMISSIBILITY_BOUNDS` in model units, from the units the script passes.

    The model is not prepared yet when the initialisation starts, so the conversion
    uses the units in its parameters rather than its variables' tags.

    """
    units = model.params["units"]
    si_units = {"pressure": "Pa", "temperature": "K"}
    return {
        name: (
            units.convert_units(lower, si_units[name]),
            units.convert_units(upper, si_units[name]),
        )
        for name, (lower, upper) in ADMISSIBILITY_BOUNDS.items()
    }


def _model_size(model) -> dict[str, Any]:
    """Cells per dimension, mortar cells and dofs of the prepared model."""
    mdg = model.mdg
    return {
        "num_dofs": int(model.equation_system.num_dofs()),
        "num_cells_by_dimension": {
            str(dim): int(sum(sd.num_cells for sd in mdg.subdomains(dim=dim)))
            for dim in range(mdg.dim_max(), -1, -1)
        },
        "num_subdomains_by_dimension": {
            str(dim): len(mdg.subdomains(dim=dim))
            for dim in range(mdg.dim_max(), -1, -1)
        },
        "num_mortar_cells": int(sum(intf.num_cells for intf in mdg.interfaces())),
    }


def _well_connections(model) -> list[dict[str, Any]]:
    """For each well grid: its depth range and the subdomains it is coupled to.

    A well that extends below its deepest coupling to a fracture has a dangling end;
    comparing ``z_min`` with the deepest coupling's depth shows whether it has one.

    """
    mdg = model.mdg
    wells = []
    for sd in mdg.subdomains(dim=1):
        if not model.is_well_grid(sd):
            continue
        couplings = []
        for intf in mdg.subdomain_to_interfaces(sd):
            primary, secondary = mdg.interface_to_subdomain_pair(intf)
            other = secondary if primary is sd else primary
            couplings.append(
                {
                    "other_dim": int(other.dim),
                    "other_is_well": bool(model.is_well_grid(other))
                    if other.dim == 1
                    else False,
                    "mortar_cells": int(intf.num_cells),
                    "z_min": float(intf.cell_centers[2].min()),
                    "z_max": float(intf.cell_centers[2].max()),
                }
            )
        wells.append(
            {
                "well_name": model.parent_well(sd).tags.get("well_name"),
                "num_cells": int(sd.num_cells),
                "z_min": float(sd.nodes[2].min()),
                "z_max": float(sd.nodes[2].max()),
                "couplings": couplings,
            }
        )
    return wells


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable.")


def _parse_arguments() -> tuple[Variant, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", choices=["runscript_1", "runscript_2", "runscript_3"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--linear-solver", choices=LINEAR_SOLVERS, default="as_given")
    parser.add_argument("--cap", action="store_true")
    parser.add_argument("--stop-after-days", type=float)
    parser.add_argument("--mesh-only", action="store_true")
    arguments = parser.parse_args()
    variant = Variant(
        script=arguments.script,
        linear_solver=arguments.linear_solver,
        cap=arguments.cap,
        stop_after=None
        if arguments.stop_after_days is None
        else arguments.stop_after_days * pp.DAY,
        mesh_only=arguments.mesh_only,
    )
    return variant, arguments.directory.resolve()


if __name__ == "__main__":
    variant, directory = _parse_arguments()
    run(variant, directory)
