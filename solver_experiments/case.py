"""The geothermal reservoir case and the axes varied over it.

The default :class:`CaseSpec` and :class:`SolverSpec` reproduce PorePy's shipped
``porepy.examples.geothermal_reservoir`` configuration, which both specs modify rather
than restate: the shipped example is the baseline, so it must not drift from what
PorePy actually ships.

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp
from porepy.examples.geothermal_reservoir import (
    GeothermalReservoirWellBCs,
    set_model_params,
    set_solver_params,
)

from .convergence import RowScaledEquationMetric
from .linear_solvers import build_linear_solver


@dataclass(frozen=True)
class CaseSpec:
    """One point in the geothermal reservoir parameter space.

    Every field defaults to the value of the shipped example, so `CaseSpec()` is that
    example and any other instance differs from it only in the fields given.

    Parameters:
        injection_temperature: Temperature of the injected fluid [K]. The rock is
            hotter, so lowering this raises the thermal stress at the injector.
        injection_pressure_scale: Multiplies the injection-phase entries of the well
            pressure protocol (the entries after well opening), leaving the
            closed-well initialisation entries alone. Scales the overpressure driving
            fracture opening and shear.
        lithostatic_multipliers: Per-axis multipliers of the lithostatic stress, i.e.
            the in-situ stress anisotropy. The fractures dip at +/- 45 degrees about
            the y axis, so the x/z ratio is what resolves shear onto them.
        dilation_angle: Shear dilation angle [rad], the aperture-per-slip coupling
            that feeds the cubic law.
        cell_size: Matrix mesh size [m].
        cell_size_fracture: Fracture mesh size [m].
        schedule_length: Number of entries of the shipped five-point time schedule to
            keep; 3 stops one hour after the well opens, 5 runs the full 100 days.
        end_time_fraction: Fraction of the final schedule interval to actually run.
            Present because the last interval dominates the cost of a full run.
        injection_ramp_steps: Number of equal sub-intervals the time stepper must stop
            at while the well opens. 1 (the default) is the shipped behaviour, where the
            whole hour between well opening and the first pressure target is a single
            time step. Sub-intervals are equal rather than geometrically graded, since a
            graded ramp would violate `SolverSpec.dt_min` (0.1 h) at once.

    """

    injection_temperature: float = 300.0
    injection_pressure_scale: float = 1.0
    lithostatic_multipliers: tuple[float, float, float] = (0.8, 1.2, 1.0)
    dilation_angle: float = 0.1
    cell_size: float | None = None
    cell_size_fracture: float | None = None
    schedule_length: int = 5
    end_time_fraction: float = 1.0
    injection_ramp_steps: int = 1

    @property
    def case_id(self) -> str:
        """Short identifier naming only what differs from the shipped example."""
        default = CaseSpec()
        parts = [
            f"{name}={getattr(self, name)}"
            for name in self.__dataclass_fields__
            if getattr(self, name) != getattr(default, name)
        ]
        return "baseline" if not parts else "_".join(parts)


@dataclass(frozen=True)
class SolverSpec:
    """Solver configuration for one run.

    Parameters:
        linear_backend: Which factorisation to use, named after the solver rather than
            the module it is reached through; see
            :func:`~solver_experiments.linear_solvers.build_linear_solver`. The default
            is PorePy's own default, so that a run labelled ``shipped`` really is the
            shipped solver stack.
        row_equilibration: Whether to scale each row of the linear system by its
            largest entry before handing it to the backend.
        nonlinear_solver: ``"constraint_line_search"`` (the shipped choice) or
            ``"newton"`` for plain Newton without line search.
        max_iterations: Nonlinear iteration cap per time-step attempt.
        max_attempts: Time-step attempts before the run is given up.
        dt_min: Smallest admissible time step [s]. The run fails when a chop would go
            below it, so this bounds how hard the stepper is allowed to try.
        divergence_atol: Absolute residual and increment magnitude at which the
            nonlinear solver declares divergence. The example sets 1e12; PorePy's own
            default is no bound at all. Since the blocks of this model's residual span
            some forty orders of magnitude, any absolute bound is really a bound on the
            worst-scaled block, so this is the knob that decides whether the solver is
            allowed to attempt a step at all.
        convergence_criteria: ``"global"`` is the shipped choice, a single absolute
            tolerance on the root-mean-square of the whole residual and increment
            vector. ``"blockwise"`` instead measures every equation block in solution
            units against its own residual at the start of the time step, and every
            variable block against its own magnitude, so that a block is not exempted
            from the test by being small in the units it happens to be posed in. See
            :mod:`~solver_experiments.convergence`.
        convergence_rtol: Factor by which each block must shrink, under
            ``"blockwise"``.
        convergence_atol: Floor below which a block counts as converged whatever its
            reference, under ``"blockwise"``. It is what decides blocks that start out
            already satisfied, where a relative test has nothing to measure against.
        divergence_rtol: Factor by which an equation block's residual may grow beyond
            its own reference before ``"blockwise"`` declares divergence.
            ``divergence_atol`` has no role there, since an absolute bound is exactly
            what block-wise tests replace.
        telemetry_verbosity: 0 disables telemetry, 1 records the scalar fields.

    """

    linear_backend: str = "pypardiso"
    row_equilibration: bool = False
    nonlinear_solver: str = "constraint_line_search"
    max_iterations: int = 25
    max_attempts: int = 10
    dt_min: float = 0.1 * pp.HOUR
    divergence_atol: float = 1e12
    convergence_criteria: str = "global"
    convergence_rtol: float = 1e-6
    convergence_atol: float = 1e-10
    divergence_rtol: float = 1e6
    telemetry_verbosity: int = 1
    solver_options: dict[str, Any] = field(default_factory=dict)

    @property
    def solver_id(self) -> str:
        """Short identifier naming only what differs from the shipped configuration."""
        default = SolverSpec()
        parts = [
            f"{name}={getattr(self, name)}"
            for name in self.__dataclass_fields__
            if name != "solver_options"
            and getattr(self, name) != getattr(default, name)
        ]
        return "shipped" if not parts else "_".join(parts)


def build_model_params(
    case: CaseSpec,
    solver: SolverSpec,
    folder_name: str,
    restart_from: Path | None = None,
) -> dict:
    """Assemble the model parameters for one case.

    Parameters:
        case: The case to build.
        solver: Solver configuration, needed for the time-stepping bounds.
        folder_name: Where the model writes its visualisation output.
        restart_from: Visualisation directory of an earlier run to resume from, or
            `None` to start from the initial condition. The times file has to be named
            explicitly: PorePy would otherwise look for it beside the *new* run's
            output, where there is none yet.

    Returns:
        Model parameter dictionary for :class:`GeothermalReservoirWellBCs`.

    """
    params = set_model_params()
    params["folder_name"] = folder_name
    if restart_from is not None:
        params["restart_options"] = {
            "restart": True,
            "pvd_file": restart_from / "data.pvd",
            "times_file": restart_from / "times.json",
        }
    params["injection_well_temperatures"] = case.injection_temperature
    params["lithostatic_stress_multipliers"] = np.asarray(
        case.lithostatic_multipliers, dtype=float
    )
    _set_solid_value(params, "dilation_angle", case.dilation_angle)
    if case.cell_size is not None:
        params["meshing_arguments"]["cell_size"] = case.cell_size
    if case.cell_size_fracture is not None:
        params["meshing_arguments"]["cell_size_fracture"] = case.cell_size_fracture
    schedule, pressures = _schedule_and_pressures(case, params)
    params["injection_well_pressures"] = pressures
    params["time_manager"] = _time_manager(solver, params, schedule)
    return params


def _set_solid_value(params: dict, name: str, value: float) -> None:
    """Replace one solid constant, keeping every other constant as shipped."""
    solid = params["material_constants"]["solid"]
    params["material_constants"]["solid"] = replace(solid, **{name: value})


def _schedule_and_pressures(
    case: CaseSpec, params: dict
) -> tuple[np.ndarray, list[float]]:
    """Return the time schedule and the matching well pressure protocol.

    The two are built together because the protocol carries one value per schedule
    point: inserting ramp points means interpolating the protocol onto them.
    Interpolating rather than inventing values keeps the pressure the well sees at any
    given time exactly as shipped, so a ramped run differs from the baseline only in
    where the time stepper is forced to stop.

    """
    times = _shipped_times(case, params)
    pressures = _scaled_pressures(case, params)
    times, pressures = _with_ramp_points(case, times, pressures)
    if case.end_time_fraction < 1.0:
        times = times.copy()
        times[-1] = times[-2] + case.end_time_fraction * (times[-1] - times[-2])
        pressures = _interpolated(
            times, _shipped_times(case, params), _scaled_pressures(case, params)
        )
    return times, pressures


def _shipped_times(case: CaseSpec, params: dict) -> np.ndarray:
    """The shipped schedule, truncated to the requested number of points."""
    schedule = params["time_manager"].schedule.get_array()
    return np.asarray(schedule, dtype=float)[: case.schedule_length]


def _scaled_pressures(case: CaseSpec, params: dict) -> list[float]:
    """Scale the injection-phase entries of the well pressure protocol.

    The first two entries drive the closed-well initialisation and are left alone: the
    well is on a Neumann condition then, so scaling them would change nothing while
    making the protocol harder to read.

    """
    pressures = list(params["injection_well_pressures"])[: case.schedule_length]
    return [
        p * case.injection_pressure_scale if i >= 2 else p
        for i, p in enumerate(pressures)
    ]


def _with_ramp_points(
    case: CaseSpec, times: np.ndarray, pressures: list[float]
) -> tuple[np.ndarray, list[float]]:
    """Subdivide the interval in which the well opens into equal sub-intervals.

    The well is held closed until the second schedule point, so the interval that
    follows it is the one carrying the opening.

    """
    if case.injection_ramp_steps <= 1 or times.size < 3:
        return times, pressures
    interior = np.linspace(times[1], times[2], case.injection_ramp_steps + 1)[1:-1]
    ramped = np.concatenate([times[:2], interior, times[2:]])
    return ramped, _interpolated(ramped, times, pressures)


def _interpolated(
    new_times: np.ndarray, times: np.ndarray, pressures: list[float]
) -> list[float]:
    """Sample a well pressure protocol at new time points."""
    return [float(value) for value in np.interp(new_times, times, pressures)]


def _time_manager(
    solver: SolverSpec, params: dict, schedule: np.ndarray
) -> pp.TimeManager:
    """Build the time manager for a given schedule, keeping the shipped adaptivity."""
    shipped = params["time_manager"]
    return pp.TimeManager(
        schedule=schedule,
        dt_init=shipped.dt_init,
        constant_dt=False,
        dt_min_max=(solver.dt_min, max(pp.YEAR, shipped.dt_init)),
        iter_optimal_range=(6, 10),
        iter_relax_factors=(0.5, 1.8),
    )


def build_model(
    case: CaseSpec,
    solver: SolverSpec,
    folder_name: str,
    restart_from: Path | None = None,
):
    """Instantiate the model for one case, without preparing the simulation."""
    return GeothermalReservoirWellBCs(
        build_model_params(case, solver, folder_name, restart_from)
    )


def nonlinear_solver_class(solver: SolverSpec) -> type:
    """Return the nonlinear solver class named by the spec."""
    if solver.nonlinear_solver == "newton":
        return pp.solvers.NewtonSolver
    if solver.nonlinear_solver == "constraint_line_search":
        return pp.solvers.ConstraintLineSearchNonlinearSolver
    raise ValueError(f"Unknown nonlinear solver {solver.nonlinear_solver!r}.")


def build_solver_params(solver: SolverSpec, model=None) -> dict:
    """Assemble the nonlinear solver parameters, starting from the shipped ones.

    The shipped parameters name the solver class under ``"nonlinear_solver"``, which
    `ModelRunner` rejects when a solver instance is also passed to it. The class is
    dropped here and obtained from :func:`nonlinear_solver_class` instead.

    Parameters:
        solver: Solver configuration.
        model: A prepared model, required only by ``convergence_criteria="blockwise"``,
            whose metrics need the equation and variable indexers to split a vector
            into blocks.

    """
    params = set_solver_params()
    params.pop("nonlinear_solver")
    params["nl_max_iterations"] = solver.max_iterations
    params["nl_divergence_inc_atol"] = solver.divergence_atol
    params["nl_divergence_res_atol"] = solver.divergence_atol
    params["prepare_simulation"] = False
    if solver.nonlinear_solver == "newton":
        params.pop("global_line_search")
        params.pop("local_line_search")
    if solver.convergence_criteria == "blockwise":
        _set_blockwise_criteria(params, solver, model)
    elif solver.convergence_criteria != "global":
        raise ValueError(
            f"Unknown convergence criteria {solver.convergence_criteria!r}."
        )
    return params


def _set_blockwise_criteria(params: dict, solver: SolverSpec, model) -> None:
    """Replace the global tolerances by one test per equation and variable block.

    PorePy refuses a criteria dictionary alongside the individual tolerance parameters
    it would override, so those are removed here rather than left to be ignored.

    Increments are split by variable and residuals by equation, since those are the
    two blockings the respective vectors are indexed by; a single metric cannot serve
    both.

    """
    if model is None:
        raise ValueError("Block-wise convergence criteria need a prepared model.")
    for key in (
        "nl_max_iterations",
        "nl_convergence_inc_atol",
        "nl_convergence_res_atol",
        "nl_convergence_inc_rtol",
        "nl_convergence_res_rtol",
        "nl_divergence_inc_atol",
        "nl_divergence_res_atol",
    ):
        params.pop(key, None)

    increment_metric = pp.VariableBasedEuclideanMetric(model)
    residual_metric = RowScaledEquationMetric(model)
    params["nl_convergence_criteria"] = {
        "inc": pp.solvers.IncrementBasedCombinedCriterion(
            atol=solver.convergence_atol,
            rtol=solver.convergence_rtol,
            metric=increment_metric,
        ),
        "res": pp.solvers.ResidualBasedCombinedCriterion(
            atol=solver.convergence_atol,
            rtol=solver.convergence_rtol,
            metric=residual_metric,
        ),
    }
    params["nl_divergence_criteria"] = {
        "max_iter": pp.solvers.MaxIterationsCriterion(
            max_iterations=solver.max_iterations
        ),
        "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
        "res_nan": pp.solvers.ResidualBasedNanCriterion(),
        # The bound is `atol + rtol * reference`. Scaling the convergence floor by the
        # same growth factor makes divergence the mirror of convergence: a block is
        # given `divergence_rtol` times the room the convergence test allows it,
        # whether that room comes from the floor or from the block's own reference.
        # `divergence_atol` has no place here, being far above every block's own scale.
        #
        # There is deliberately no increment-based counterpart. The reference for an
        # increment is the iterate itself, and the well flux and interface fluxes are
        # zero until the well opens, so the step that opens it moves them by an
        # unbounded factor of their own magnitude - a correct jump that any relative
        # test reads as divergence. Measured: it ended every attempt of that step after
        # one iteration, down to a time step of 7 s. A runaway update is visible in the
        # residual anyway, which is measured in solution units against its own value.
        "res_growth": pp.solvers.ResidualBasedCombinedDivergenceCriterion(
            atol=solver.divergence_rtol * solver.convergence_atol,
            rtol=solver.divergence_rtol,
            metric=residual_metric,
        ),
    }


def build_solver(model, solver: SolverSpec, telemetry, linear_solver):
    """Instantiate the nonlinear solver for a prepared model.

    Parameters:
        model: A model on which `prepare_simulation` has already been called, since
            whether the problem is nonlinear is decided from the mixed-dimensional
            grid.
        solver: Solver configuration.
        telemetry: Telemetry recorder, or None.
        linear_solver: The linear solver instance to use.

    Returns:
        The nonlinear solver instance.

    """
    return nonlinear_solver_class(solver)(
        params=build_solver_params(solver, model),
        is_nonlinear_problem=model._is_nonlinear_problem(),
        linear_solver=linear_solver,
        telemetry=telemetry,
    )


def build_case_linear_solver(solver: SolverSpec):
    """Build the (possibly wrapped) linear solver for one run."""
    return build_linear_solver(
        backend=solver.linear_backend,
        row_equilibration=solver.row_equilibration,
        options=solver.solver_options,
    )
