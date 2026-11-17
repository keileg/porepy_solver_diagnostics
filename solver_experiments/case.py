"""The geothermal reservoir case and the axes varied over it.

The default :class:`CaseSpec` and :class:`SolverSpec` reproduce PorePy's shipped
``porepy.examples.geothermal_reservoir`` configuration, which both specs modify rather
than restate: the shipped example is the baseline, so it must not drift from what
PorePy actually ships.

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import porepy as pp
from porepy.applications.discretizations.flux_discretization import (
    FluxDiscretization,
)
from porepy.examples.geothermal_reservoir import (
    GeothermalReservoirWellBCs,
    set_model_params,
    set_solver_params,
)

from .convergence import RowScaledEquationMetric
from .linear_solvers import build_linear_solver


UNIT_SYSTEMS = {
    "shipped": {"m": 1.0, "kg": 1.0e5, "K": 1.0},
    "si": {},
    "pressure": {"kg": 1.0e7},
    "pressure_temperature": {"kg": 1.0e7, "K": 1.0e2},
}
"""Candidate unit systems, named by what they make of order one.

`pressure_temperature` puts the injection pressure and the reservoir temperature at
order one; `shipped` scales mass alone. Time is absent because PorePy refuses a
non-unitary time scaling, which is the one scaling that would address the twelve orders
between the three-dimensional mass and momentum balances.

"""


ADMISSIBILITY_BOUNDS = {
    "pressure": (0.0, np.inf),
    "temperature": (273.16, 647.0),
}
"""Bounds outside which an iterate is not physical for this case, in SI units.

Pressure must stay positive, and the fluid is liquid water, so its triple point and
critical temperature bound the temperature. The model's own fluid laws are defined
beyond these (density is exponential in both), so the bounds are plausibility bounds
rather than validity limits; what an unbounded temperature breaks is the density.

"""


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
            the in-situ stress anisotropy, each applied to the boundary faces normal to
            its axis. Both fractures have unit normals (+-0.5, -+0.5, +-0.707), so x
            and y weigh equally on them; lowering x raises the shear stress and lowers
            the normal stress, and is the most direct way to push them towards slip.
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
        units: Which of :data:`UNIT_SYSTEMS` the model is posed in.
        injection_duration: Time from the well opening to the end of the run [s].
            `None` keeps the shipped schedule, which ends 100 days after the opening.
            A longer duration appends one schedule point, holding the last injection
            pressure, so thermal effects have time to build up.
        well_depth: Depth of both wells' lower ends, as a fraction of the domain depth.
            The shipped wells reach 0.8 and cross their fracture at 0.6, leaving two
            dead-end cells below each crossing whose temperatures the linear system
            barely separates. 0.6 ends the wells at the crossing instead.

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
    units: str = "shipped"
    injection_duration: float | None = None
    well_depth: float = 0.8

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
        dt_max: Largest admissible time step [s]. `None` keeps the shipped bound,
            the larger of a year and the initial step (three years). A smaller bound
            also caps the initial step, and with it the initialisation's steps.
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
        divergence_bound: Magnitude at which ``"blockwise"`` declares divergence,
            applied to each equation block's residual in solution units and to each
            variable block's increment. It separates a runaway iterate, which reaches
            1e+30 here, from the nine orders a contact block legitimately moves within
            a step, which is why the bound is absolute rather than relative to
            anything.
        darcy_flux_discretization: ``"mpfa"`` (shipped) or ``"tpfa"``, the base
            discretisation of the Darcy flux.
        differentiable_darcy: Whether the Darcy flux is differentiated with respect to
            the permeability, so that Newton sees how fracture fluxes change with the
            aperture, instead of rediscretising the flux between iterations. The
            differentiation is TPFA-based: with a TPFA base Newton can in principle
            converge quadratically, with an MPFA base it cannot, and has been seen to
            behave erratically.
        tangential_contact: ``"projection"`` (shipped), the Alart-Curnier projection,
            or ``"scaled_projection"``, the formulation PorePy shipped until it changed
            its default. The two are equivalent up to a scaling, so they share their
            solutions but not their Newton paths.
        cap_to_bounds: Whether the Newton increment is capped so that pressure and
            temperature stay within :data:`ADMISSIBILITY_BOUNDS`. Without it, the
            bounds are only counted.
        telemetry_verbosity: 0 disables telemetry, 1 records the scalar fields.
        iteration_target: Number of Newton iterations the time stepper aims at, with
            dt scaled by the distance from it (``ScaledTargetNonlinearIterations``).
            `None` keeps the shipped rule: a fixed factor outside 6-10 iterations.

    """

    linear_backend: str = "pypardiso"
    row_equilibration: bool = False
    nonlinear_solver: str = "constraint_line_search"
    max_iterations: int = 25
    max_attempts: int = 10
    dt_min: float = 0.1 * pp.HOUR
    dt_max: float | None = None
    divergence_atol: float = 1e12
    convergence_criteria: str = "global"
    convergence_rtol: float = 1e-6
    convergence_atol: float = 1e-10
    divergence_bound: float = 1e6
    darcy_flux_discretization: str = "mpfa"
    differentiable_darcy: bool = False
    tangential_contact: str = "projection"
    cap_to_bounds: bool = False
    telemetry_verbosity: int = 1
    iteration_target: int | None = None
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
    params["units"] = pp.Units(**UNIT_SYSTEMS[case.units])
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
    if solver.cap_to_bounds:
        params["variable_bounds"] = dict(ADMISSIBILITY_BOUNDS)
    params["darcy_flux_discretization"] = solver.darcy_flux_discretization
    params["well_depth"] = case.well_depth
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
    if case.injection_duration is not None:
        times, pressures = _extended_injection(case, params, times, pressures)
    return times, pressures


def _extended_injection(
    case: CaseSpec, params: dict, times: np.ndarray, pressures: list[float]
) -> tuple[np.ndarray, list[float]]:
    """Append a schedule point ending the run `injection_duration` after the opening.

    The well opens at the second shipped schedule point. The appended point holds the
    last injection pressure, so the extension changes the run only by its length.

    """
    opening = _shipped_times(case, params)[1]
    end = opening + cast(float, case.injection_duration)
    if case.schedule_length != 5 or end <= times[-1]:
        raise ValueError(
            "An injection duration extends the full shipped schedule, and must end "
            "after it."
        )
    return np.append(times, end), [*pressures, pressures[-1]]


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
    dt_max = max(pp.YEAR, shipped.dt_init) if solver.dt_max is None else solver.dt_max
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=min(shipped.dt_init, dt_max),
        constant_dt=False,
        dt_min_max=(solver.dt_min, dt_max),
        iter_optimal_range=(6, 10),
        iter_relax_factors=(0.5, 1.8),
    )
    if solver.iteration_target is not None:
        # The retry factor stays the shipped one, so only the rule after convergence
        # changes.
        constraint = pp.time_stepper.ScaledTargetNonlinearIterations(
            target_iterations=solver.iteration_target, retry_factor=0.5
        )
        for interval in time_manager.schedule.intervals:
            interval.constraints = [constraint]
    return time_manager


def build_model(
    case: CaseSpec,
    solver: SolverSpec,
    folder_name: str,
    restart_from: Path | None = None,
):
    """Instantiate the model for one case, without preparing the simulation."""
    return _model_class(case, solver)(
        build_model_params(case, solver, folder_name, restart_from)
    )


class _WellDepth:
    """Wells as shipped, but ending at the depth the case asks for."""

    params: dict
    well_names: list[str]
    domain_sizes: Any

    def set_wells(self) -> None:
        dx, dy, dz = self.domain_sizes()
        depth = self.params["well_depth"] * dz
        self._wells = [
            pp.Well(
                np.array([[x * dx, x * dx], [x * dy, x * dy], [0.0, -depth]]),
                tags={"well_name": name},
            )
            for x, name in zip((0.4, 0.6), self.well_names)
        ]


def _model_class(case: CaseSpec, solver: SolverSpec) -> type:
    """The shipped model, with the wells, Darcy flux and contact formulation asked for.

    The shipped class hard-codes an MPFA Darcy flux and the well geometry; a choice of
    base, the differentiable flux, other well depths and the scaled tangential contact
    formulation are mixed in ahead of it, so they take precedence.

    """
    mixins: list[type] = []
    if solver.tangential_contact == "scaled_projection":
        mixins.append(
            pp.contact_mechanics.ScaledProjectionTangentialContactMechanicsEquation
        )
    elif solver.tangential_contact != "projection":
        raise ValueError(f"Unknown tangential contact {solver.tangential_contact!r}.")
    if case.well_depth != CaseSpec().well_depth:
        mixins.append(_WellDepth)
    if solver.differentiable_darcy:
        mixins.append(pp.constitutive_laws.DarcysLawAd)
    if solver.darcy_flux_discretization != "mpfa":
        mixins.append(FluxDiscretization)
    if not mixins:
        return GeothermalReservoirWellBCs
    return type("GeothermalReservoirVariant", (*mixins, GeothermalReservoirWellBCs), {})


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
    # Divergence is tested against a fixed magnitude, not against a block's own
    # earlier value. Growth says nothing here: a fracture cell changing contact state
    # moves `normal_fracture_deformation` nine orders within a step, and an increment
    # relative to its own variable is undefined for the well flux, which is zero until
    # the well opens. What has to be caught is the iterate running away - updates reach
    # 1e+30 - and dropping the test altogether is not an option: the run then iterates
    # into a singular Jacobian and dies inside the property correlations, on an
    # exception the nonlinear solver does not catch.
    params["nl_divergence_criteria"] = {
        "max_iter": pp.solvers.MaxIterationsCriterion(
            max_iterations=solver.max_iterations
        ),
        "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
        "res_nan": pp.solvers.ResidualBasedNanCriterion(),
        "inc_bound": pp.solvers.IncrementBasedAbsoluteDivergenceCriterion(
            tol=solver.divergence_bound, metric=increment_metric
        ),
        "res_bound": pp.solvers.ResidualBasedAbsoluteDivergenceCriterion(
            tol=solver.divergence_bound, metric=residual_metric
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
