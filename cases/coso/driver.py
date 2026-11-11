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
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import porepy as pp
import porepy.applications.initialization.initialization as initialization_module
from porepy.applications.initialization.model_state_storage import ModelStateStorage
from porepy.time_stepper.time_step_status import TimeStepperStatusSuccess

from solver_experiments.case import ADMISSIBILITY_BOUNDS
from solver_experiments.instrumentation import (
    AssemblyTimers,
    count_capped_dofs,
    instrument_assembly,
)
from solver_experiments.linear_solvers import (
    InstrumentedSolver,
    backward_error,
    build_linear_solver,
)
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
        steady_state: A ``steady_state.npz`` saved by an earlier run's
            initialisation, to start from instead of time stepping to a steady state.
            The pipeline still validates it against this run's model.
        stop_at_step_cut: Stop after the first accepted time step that needed more
            than one attempt, i.e. after the first cut of dt, in whichever phase it
            happens. Nothing is stored as a steady state.
        solver_options: PETSc options merged into the script's, per solver key (e.g.
            ``{"mechanics_amg": {"ksp_type": "gmres"}}``); only for
            ``"thm_factory"`` and ``"system_amg"``.
        dump_systems_min_dt: Save every linear system solved at a time step of at
            least this size [s], with its solution and index maps, under
            ``systems_{phase}/``; None saves none.
        gmres_restart: GMRES restart length of the outer solve, replacing PETSc's
            default as set by ``pp_solvers``; only for ``"thm_factory"`` and
            ``"system_amg"``. None keeps the script's options.
        replay: Solver variants to also solve each system with, where the time step is
            at least ``replay_min_dt``, before the run's own solver solves it (see
            :class:`_Replay`). Each entry has a ``"label"``, and optionally a
            ``"fixed_stress_scale"`` (a factor on ``pp_solvers``' fixed-stress term)
            and ``"solver_options"`` (merged as ``solver_options`` is). Only for
            ``"thm_factory"`` and ``"system_amg"``. None replays nothing.
        replay_min_dt: Smallest time step [s] whose systems are replayed.
        replay_systems: Number of systems to replay; the run stops after the last.

    """

    script: str
    linear_solver: str = "as_given"
    cap: bool = False
    stop_after: float | None = None
    mesh_only: bool = False
    steady_state: Path | None = None
    gmres_restart: int | None = None
    stop_at_step_cut: bool = False
    dump_systems_min_dt: float | None = None
    solver_options: dict[str, dict[str, Any]] | None = None
    replay: list[dict[str, Any]] | None = None
    replay_min_dt: float = 0.0
    replay_systems: int = 3


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


class _StoppedAtStepCut(Exception):
    """Raised to leave the script after the first time step that needed a retry."""


class _ReplayDone(Exception):
    """Raised to leave the script once the last system to replay has been solved."""


class _StopAtStepCut(pp.models.model_runner.EarlyStopCriterion):
    """Stop at the first accepted time step that needed more than one attempt.

    Only the time stepper sees the attempts, so this stands in for its telemetry
    recorder and forwards everything to the real one. Stopping raises instead of
    returning a status, since a success status would make the initialisation store
    the state as its steady state.

    Parameters:
        recorder: The time stepper's telemetry recorder.

    """

    def __init__(self, recorder) -> None:
        self.recorder = recorder
        self.last_status = None

    def __getattr__(self, name: str):
        return getattr(self.recorder, name)

    def record_time_step(self, model, status) -> None:
        self.recorder.record_time_step(model, status)
        self.last_status = status

    def simulation_should_stop(self, model):
        status = self.last_status
        if isinstance(status, TimeStepperStatusSuccess) and len(status.attempts) > 1:
            dts = ", ".join(f"{attempt.dt:.3g}" for attempt in status.attempts)
            now = model.time_manager.time
            raise _StoppedAtStepCut(f"time {now:.4g} reached with dt {dts}")
        return pp.models.model_runner.EarlyStopNotRequested()


def _watched(variant: Variant, recorder, criteria: list) -> tuple[Any, list]:
    """The time stepper's recorder and the early-stop criteria for a variant."""
    if not variant.stop_at_step_cut:
        return recorder, criteria
    watch = _StopAtStepCut(recorder)
    return watch, [*criteria, watch]


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
    except (_MeshOnlyDone, _StoppedAtStepCut, _ReplayDone):
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
    # The initialisation's phase while the pipeline runs. The pipeline builds its own
    # model runner, which must then pass through: its solver is instrumented already.
    initialization_phase: list[_Phase] = []

    def instrumented_pipeline(model, *args, nonlinear_solver=None, **kwargs):
        if variant.cap:
            model.params["variable_bounds"] = dict(ADMISSIBILITY_BOUNDS)
        if variant.steady_state is not None:
            kwargs["storage"] = _private_copy(variant.steady_state, directory)
        if variant.mesh_only:
            model.prepare_simulation()
            record.model_size = _model_size(model)
            record.wells = _well_connections(model)
            raise _MeshOnlyDone
        phase = _Phase("initialization", variant, directory, model)
        nonlinear_solver = phase.instrument(nonlinear_solver)
        timers = AssemblyTimers()
        count_capped_dofs(nonlinear_solver, timers)
        initialization_phase.append(phase)
        success, failure_reason = True, None
        t_0 = time.time()
        try:
            return original_pipeline(
                model, *args, nonlinear_solver=nonlinear_solver, **kwargs
            )
        except Exception as error:
            success, failure_reason = False, f"{type(error).__name__}: {error}"
            raise
        finally:
            initialization_phase.clear()
            timers.run_time = record.initialization_time = time.time() - t_0
            phase.write_metrics(success, failure_reason, timers)
            record.phases_done.append("initialization")

    class InstrumentedModelRunner(pp.ModelRunner):
        """Model runner whose nonlinear solver and time stepper are instrumented."""

        def __init__(
            self,
            model,
            params=None,
            time_stepper=None,
            nonlinear_solver=None,
            early_stop_criteria=None,
        ):
            self._phase: _Phase | None = None
            if initialization_phase:
                recorder, criteria = _watched(
                    variant,
                    initialization_phase[0].recorder,
                    list(early_stop_criteria or []),
                )
                super().__init__(
                    model, params, time_stepper, nonlinear_solver, criteria
                )
                self.time_stepper.telemetry = recorder
                return
            self._phase = _Phase("main", variant, directory, model)
            nonlinear_solver = self._phase.instrument(nonlinear_solver)
            assert time_stepper is not None, "The runscripts pass their own stepper."
            recorder, criteria = _watched(
                variant,
                self._phase.recorder,
                [*(early_stop_criteria or []), *_stop_criteria(variant)],
            )
            time_stepper.telemetry = recorder
            self._timers = AssemblyTimers()
            super().__init__(
                model,
                params=params,
                time_stepper=time_stepper,
                nonlinear_solver=nonlinear_solver,
                early_stop_criteria=criteria,
            )
            record.model_size = _model_size(model)
            record.wells = _well_connections(model)
            instrument_assembly(model, self._timers)
            count_capped_dofs(self.solver, self._timers)

        def run(self):
            if self._phase is None:
                return super().run()
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


def _private_copy(steady_state: Path, directory: Path) -> ModelStateStorage:
    """Storage for a copy of a saved steady state, private to this run.

    The pipeline deletes a stored state that does not match the model or fails its
    validation, so the original must not be handed to it.

    """
    if not steady_state.is_file():
        raise FileNotFoundError(f"No saved steady state at {steady_state}.")
    copy = directory / "steady_state.npz"
    shutil.copyfile(steady_state, copy)
    return ModelStateStorage(copy)


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
        delegate = _linear_solver(self.variant, nonlinear_solver)
        self.linear_solver = InstrumentedSolver(delegate)
        nonlinear_solver.linear_solver = self.linear_solver
        if self.variant.dump_systems_min_dt is not None:
            nonlinear_solver.linear_solver = _SystemDump(
                self.linear_solver,
                self.directory / f"systems_{self.name}",
                self.variant.dump_systems_min_dt,
            )
        if self.variant.replay is not None:
            nonlinear_solver.linear_solver = _Replay(
                nonlinear_solver.linear_solver,
                [
                    _ReplayVariant.build(self.variant, entry)
                    for entry in self.variant.replay
                ],
                self.directory / f"replay_{self.name}.jsonl",
                self.variant.replay_min_dt,
                self.variant.replay_systems,
            )
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


class _SystemDump(pp.solvers.LinearSolverBase):
    """Save the linear systems of a run with their solutions, for offline analysis.

    Each system goes to ``system_NNN.npz`` (matrix, right-hand side, solution, Krylov
    iterations, status, time and dt). The row and column indices of every equation and
    variable on every grid go to ``blocks.npz`` once, keyed ``row|name|dim|id`` and
    ``col|name|dim|id``.

    Parameters:
        delegate: The solver doing the actual work.
        directory: Where to save; created.
        min_dt: Only systems solved at a time step of at least this size are saved.

    """

    def __init__(self, delegate, directory: Path, min_dt: float) -> None:
        self.delegate = delegate
        self.directory = directory
        self.min_dt = min_dt
        self.model = None
        self.count = 0

    def initialize_with_model(self, model) -> None:
        self.model = model
        self.delegate.initialize_with_model(model)

    def solve_linear_system(self, linear_system):
        matrix = linear_system.matrix
        solution, status = self.delegate.solve_linear_system(linear_system)
        dt = self.model.time_manager.dt
        if dt >= self.min_dt:
            self._save(linear_system, matrix, solution, status, dt)
        return solution, status

    def _save(self, linear_system, matrix, solution, status, dt) -> None:
        if self.count == 0:
            self.directory.mkdir()
            np.savez(self.directory / "blocks.npz", **_block_indices(linear_system))
        matrix = matrix.tocsr()
        np.savez(
            self.directory / f"system_{self.count:03d}.npz",
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=np.array(matrix.shape),
            rhs=linear_system.rhs,
            solution=solution,
            krylov_iterations=getattr(status, "num_krylov_iters", -1),
            status=type(status).__name__,
            time=self.model.time_manager.time,
            dt=dt,
        )
        self.count += 1


def _block_indices(linear_system) -> dict[str, np.ndarray]:
    """Rows of each equation and columns of each variable, per grid."""
    blocks = {}
    for kind, indexer in (
        ("row", linear_system.equation_indexer),
        ("col", linear_system.variable_indexer),
    ):
        for name, per_grid in indexer.group_by_name().items():
            for grid, indices in per_grid.items():
                blocks[f"{kind}|{name}|{grid.dim}|{grid.id}"] = np.asarray(indices)
    return blocks


@dataclass
class _ReplayVariant:
    """A solver to replay systems with, and the fixed-stress scale it solves under."""

    label: str
    solver: pp.solvers.LinearSolverBase
    fixed_stress_scale: float = 1.0

    @classmethod
    def build(cls, variant: Variant, entry: dict[str, Any]) -> _ReplayVariant:
        if variant.linear_solver not in ("thm_factory", "system_amg"):
            raise ValueError("Replay needs a PETSc solver.")
        changes = _option_changes(variant)
        for key, values in entry.get("solver_options", {}).items():
            changes[key] = {**changes.get(key, {}), **values}
        return cls(
            label=entry["label"],
            solver=_petsc_solver(variant.linear_solver, changes),
            fixed_stress_scale=float(entry.get("fixed_stress_scale", 1.0)),
        )


class _Replay(pp.solvers.LinearSolverBase):
    """Solve some of a run's linear systems with other solver variants as well.

    Each system solved at a time step of at least ``min_dt`` is first solved by every
    variant, each on its own copy, and then by the run's own solver, whose solution the
    run continues with; the trajectory is that of the run's own solver. The variants
    therefore see exactly the systems the run solves, with the model in the state it
    assembled them in. Each solve's outcome is logged and appended to ``path`` as a
    JSON line, together with the size of the fixed-stress term against the pressure
    diagonal. The run stops after ``num_systems`` systems.

    """

    def __init__(
        self,
        delegate,
        variants: list[_ReplayVariant],
        path: Path,
        min_dt: float,
        num_systems: int,
    ) -> None:
        self.delegate = delegate
        self.variants = variants
        self.path = path
        self.min_dt = min_dt
        self.num_systems = num_systems
        self.model = None
        self.replayed = 0

    def initialize_with_model(self, model) -> None:
        self.model = model
        self.delegate.initialize_with_model(model)
        for variant in self.variants:
            variant.solver.initialize_with_model(model)

    def solve_linear_system(self, linear_system):
        if self.model.time_manager.dt < self.min_dt:
            return self.delegate.solve_linear_system(linear_system)
        reference = linear_system.matrix.copy()
        rhs = linear_system.rhs.copy()
        self._write(
            {"fixed_stress": _fixed_stress_against_pressure(self.model, linear_system)}
        )
        for variant in self.variants:
            copy = pp.solvers.LinearSystem(
                reference.copy(),
                rhs.copy(),
                linear_system.equation_indexer,
                linear_system.variable_indexer,
            )
            with _fixed_stress_scaled(variant.fixed_stress_scale):
                outcome = variant.solver.solve_linear_system(copy)
            self._record(variant.label, reference, rhs, *outcome)
        solution, status = self.delegate.solve_linear_system(linear_system)
        self._record("run", reference, rhs, solution, status)
        self.replayed += 1
        if self.replayed == self.num_systems:
            raise _ReplayDone()
        return solution, status

    def _record(self, label, matrix, rhs, solution, status) -> None:
        finite = bool(np.all(np.isfinite(solution)))
        residual = np.linalg.norm(rhs - matrix @ solution) if finite else np.inf
        entry = {
            "system": self.replayed,
            "label": label,
            "time": self.model.time_manager.time,
            "dt": self.model.time_manager.dt,
            "status": type(status).__name__,
            "krylov_iterations": getattr(status, "num_krylov_iters", None),
            "solve_time": getattr(status, "solve_time", None),
            "relative_residual": residual / np.linalg.norm(rhs),
            "backward_error": backward_error(matrix, rhs, solution)
            if finite
            else np.inf,
            "update_inf": float(np.max(np.abs(solution))) if finite else np.inf,
        }
        logger.info("Replay %s", json.dumps(entry, default=_json_default))
        self._write(entry)

    def _write(self, entry: dict[str, Any]) -> None:
        with self.path.open("a") as file:
            file.write(json.dumps(entry, default=_json_default) + "\n")


@contextmanager
def _fixed_stress_scaled(factor: float) -> Iterator[None]:
    """Scale ``pp_solvers``' fixed-stress term, matrix and fractures alike, by a factor.

    The term is assembled when the preconditioner is built, inside the solve, through
    the two module-level functions replaced here.

    """
    import pp_solvers.fixed_stress as fixed_stress

    names = (
        "get_fixed_stress_stabilization_porous_media",
        "get_fixed_stress_stabilization_fractures",
    )
    originals = {name: getattr(fixed_stress, name) for name in names}
    for name, original in originals.items():
        setattr(
            fixed_stress,
            name,
            lambda model, original=original: factor * original(model),
        )
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(fixed_stress, name, original)


def _fixed_stress_against_pressure(model, linear_system) -> dict[str, Any]:
    """Size of the fixed-stress term relative to the mass balance's pressure diagonal.

    Per subdomain dimension (matrix and fractures): the quantiles of the ratio, cell by
    cell, and the fraction of cells where the term vanishes or is negative.

    """
    import pp_solvers.fixed_stress as fixed_stress

    blocks = _block_indices(linear_system)
    terms = {
        model.nd: fixed_stress.get_fixed_stress_stabilization_porous_media,
        model.nd - 1: fixed_stress.get_fixed_stress_stabilization_fractures,
    }
    report = {}
    for dim, term in terms.items():
        grids = model.mdg.subdomains(dim=dim)
        rows = np.concatenate(
            [blocks[f"row|mass_balance_equation|{dim}|{sd.id}"] for sd in grids]
        )
        columns = np.concatenate(
            [blocks[f"col|pressure|{dim}|{sd.id}"] for sd in grids]
        )
        diagonal = np.abs(linear_system.matrix[rows][:, columns].diagonal())
        stabilisation = term(model).diagonal()
        ratio = np.abs(stabilisation) / np.maximum(diagonal, 1e-300)
        report[str(dim)] = {
            "ratio_quantiles_0_10_50_90_100": np.quantile(ratio, [0, 0.1, 0.5, 0.9, 1]),
            "zero_fraction": float(np.mean(stabilisation == 0)),
            "negative_fraction": float(np.mean(stabilisation < 0)),
        }
    return report


def _linear_solver(variant: Variant, nonlinear_solver) -> pp.solvers.LinearSolverBase:
    """The linear solver a variant asks for; the script's own for ``"as_given"``."""
    choice = variant.linear_solver
    changes = _option_changes(variant)
    if changes and choice not in ("thm_factory", "system_amg"):
        raise ValueError(f"PETSc options need a PETSc solver, not {choice!r}.")
    if choice == "as_given":
        return nonlinear_solver.linear_solver
    if choice == "pypardiso":
        return build_linear_solver("pypardiso", False, {}).delegate
    if choice in ("thm_factory", "system_amg"):
        return _petsc_solver(choice, changes)
    raise ValueError(
        f"Unknown linear solver {choice!r}, expected one of {LINEAR_SOLVERS}."
    )


def _petsc_solver(choice: str, changes: dict[str, dict[str, Any]]):
    """One of the scripts' two PETSc solvers, with their options changed as given."""
    # Imported from the script's own tree, so that the factory and the PETSc options
    # are the ones the script uses.
    import pp_solvers
    from nl_params import thm_factory_system_amg
    from solver_configurations import linear_solver_params

    factory = (
        thm_factory_system_amg if choice == "system_amg" else pp_solvers.thm_factory
    )
    options = dict(linear_solver_params)
    for key, values in changes.items():
        options[key] = {**options.get(key, {}), **values}
    return pp_solvers.IterativeLinearSolver(
        configuration_factory=factory, solver_options=options
    )


def _option_changes(variant: Variant) -> dict[str, dict[str, Any]]:
    """The PETSc options a variant changes, per solver key."""
    changes = {
        key: dict(values) for key, values in (variant.solver_options or {}).items()
    }
    if variant.gmres_restart is not None:
        changes.setdefault("gmres", {})["ksp_gmres_restart"] = variant.gmres_restart
    return changes


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
    parser.add_argument("--steady-state", type=Path)
    parser.add_argument("--gmres-restart", type=int)
    parser.add_argument("--stop-at-step-cut", action="store_true")
    parser.add_argument("--dump-systems-min-dt", type=float)
    parser.add_argument("--solver-options", type=json.loads)
    parser.add_argument("--replay", type=json.loads)
    parser.add_argument("--replay-min-dt", type=float, default=0.0)
    parser.add_argument("--replay-systems", type=int, default=3)
    arguments = parser.parse_args()
    variant = Variant(
        script=arguments.script,
        linear_solver=arguments.linear_solver,
        cap=arguments.cap,
        stop_after=None
        if arguments.stop_after_days is None
        else arguments.stop_after_days * pp.DAY,
        mesh_only=arguments.mesh_only,
        steady_state=None
        if arguments.steady_state is None
        else arguments.steady_state.resolve(),
        gmres_restart=arguments.gmres_restart,
        stop_at_step_cut=arguments.stop_at_step_cut,
        dump_systems_min_dt=arguments.dump_systems_min_dt,
        solver_options=arguments.solver_options,
        replay=arguments.replay,
        replay_min_dt=arguments.replay_min_dt,
        replay_systems=arguments.replay_systems,
    )
    return variant, arguments.directory.resolve()


if __name__ == "__main__":
    variant, directory = _parse_arguments()
    run(variant, directory)
