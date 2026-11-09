"""Tests that a case spec means what it says.

The campaign's conclusions rest on every run differing from the shipped example only in
the axis under study, so the construction of model parameters is where a silent mistake
would be most expensive.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import porepy as pp
import pytest

from solver_experiments.case import (
    ADMISSIBILITY_BOUNDS,
    CaseSpec,
    SolverSpec,
    build_model_params,
)


@pytest.fixture
def shipped() -> dict:
    """Model parameters of the unmodified case."""
    return build_model_params(CaseSpec(), SolverSpec(), folder_name="unused")


def test_default_spec_reproduces_the_shipped_example(shipped):
    """`CaseSpec()` must be the shipped configuration, since it is the baseline."""
    from porepy.examples.geothermal_reservoir import set_model_params

    example = set_model_params()
    assert (
        shipped["injection_well_temperatures"] == example["injection_well_temperatures"]
    )
    assert shipped["injection_well_pressures"] == list(
        example["injection_well_pressures"]
    )
    assert np.allclose(
        shipped["lithostatic_stress_multipliers"],
        example["lithostatic_stress_multipliers"],
    )
    assert (
        shipped["material_constants"]["solid"].dilation_angle
        == example["material_constants"]["solid"].dilation_angle
    )
    assert shipped["meshing_arguments"] == example["meshing_arguments"]


def test_case_id_names_only_what_differs():
    """The identifier is what results directories are keyed by."""
    assert CaseSpec().case_id == "baseline"
    assert CaseSpec(dilation_angle=0.2).case_id == "dilation_angle=0.2"


def test_only_the_injection_phase_pressures_are_scaled(shipped):
    """The first two protocol entries drive the closed-well initialisation.

    Scaling them would change the initialisation rather than the injection, which is the
    axis actually under study.

    """
    scaled = build_model_params(
        CaseSpec(injection_pressure_scale=2.0), SolverSpec(), folder_name="unused"
    )["injection_well_pressures"]
    original = shipped["injection_well_pressures"]
    assert scaled[:2] == original[:2]
    assert scaled[2:] == [2.0 * value for value in original[2:]]


def test_shortening_the_schedule_keeps_the_leading_time_points(shipped):
    """A shortened run must be a prefix of the full one, or runs are not comparable."""
    short = build_model_params(
        CaseSpec(schedule_length=3), SolverSpec(), folder_name="unused"
    )
    full_schedule = shipped["time_manager"].schedule.get_array()
    short_schedule = short["time_manager"].schedule.get_array()
    assert np.allclose(short_schedule, full_schedule[:3])
    assert len(short["injection_well_pressures"]) == 3


def test_end_time_fraction_shortens_only_the_last_interval(shipped):
    """The knob exists to make the expensive final interval affordable."""
    partial = build_model_params(
        CaseSpec(schedule_length=3, end_time_fraction=0.5),
        SolverSpec(),
        folder_name="unused",
    )["time_manager"].schedule.get_array()
    full = shipped["time_manager"].schedule.get_array()
    assert np.allclose(partial[:-1], full[:2])
    assert partial[-1] == pytest.approx(full[1] + 0.5 * (full[2] - full[1]))


def test_solid_constants_other_than_the_swept_one_are_untouched(shipped):
    """Replacing one constant must not reset the example's other basalt values."""
    modified = build_model_params(
        CaseSpec(dilation_angle=0.05), SolverSpec(), folder_name="unused"
    )["material_constants"]["solid"]
    assert modified.dilation_angle == 0.05
    assert modified.residual_aperture == (
        shipped["material_constants"]["solid"].residual_aperture
    )
    assert modified.normal_permeability == (
        shipped["material_constants"]["solid"].normal_permeability
    )


def test_dt_min_reaches_the_time_manager():
    """The stepper's floor is a solver setting, and the runs vary it."""
    parameters = build_model_params(
        CaseSpec(), SolverSpec(dt_min=60.0), folder_name="unused"
    )
    assert parameters["time_manager"].dt_min_max[0] == pytest.approx(60.0)


def test_ramping_the_well_opening_only_adds_stopping_points(shipped):
    """A ramped run must see the same pressure-versus-time as the shipped one.

    The ramp exists to force the stepper to stop more often while the well opens. If
    it also changed the protocol, a difference in solver behaviour could not be
    attributed to the time stepping.

    """
    ramped = build_model_params(
        CaseSpec(schedule_length=3, injection_ramp_steps=4),
        SolverSpec(),
        folder_name="unused",
    )
    times = ramped["time_manager"].schedule.get_array()
    original = shipped["time_manager"].schedule.get_array()[:3]
    assert len(times) == 6
    assert np.allclose(np.asarray(times)[[0, 1, -1]], original)
    # Equal sub-intervals over the opening hour, each above the default dt_min.
    opening = np.diff(np.asarray(times)[1:])
    assert np.allclose(opening, opening[0])
    assert opening[0] > SolverSpec().dt_min

    pressures = ramped["injection_well_pressures"]
    expected = np.interp(times, original, shipped["injection_well_pressures"][:3])
    assert np.allclose(pressures, expected)


def test_no_ramp_leaves_the_schedule_alone(shipped):
    """`injection_ramp_steps=1` must be exactly the shipped schedule."""
    unramped = build_model_params(
        CaseSpec(injection_ramp_steps=1), SolverSpec(), folder_name="unused"
    )
    assert np.allclose(
        unramped["time_manager"].schedule.get_array(),
        shipped["time_manager"].schedule.get_array(),
    )


def test_restart_names_the_previous_run_s_times_file() -> None:
    """PorePy defaults the times file to the new run's own output folder, where a
    restarted run has none yet, so it has to be named explicitly or the restart
    fails."""
    previous = Path("results/previous/viz")
    params = build_model_params(
        CaseSpec(), SolverSpec(), folder_name="results/new/viz", restart_from=previous
    )
    assert params["restart_options"] == {
        "restart": True,
        "pvd_file": previous / "data.pvd",
        "times_file": previous / "times.json",
    }


def test_no_restart_options_without_a_previous_run() -> None:
    """An unrequested restart would silently resume someone else's state."""
    params = build_model_params(CaseSpec(), SolverSpec(), folder_name="results/new/viz")
    assert "restart_options" not in params


def test_global_criteria_keep_the_shipped_tolerances() -> None:
    """The default spec must leave the convergence test exactly as shipped."""
    from porepy.examples.geothermal_reservoir import set_solver_params

    from solver_experiments.case import build_solver_params

    params = build_solver_params(SolverSpec())
    shipped = set_solver_params()
    assert "nl_convergence_criteria" not in params
    for key in ("nl_convergence_inc_atol", "nl_convergence_res_atol"):
        assert params[key] == shipped[key]


class _StubModel:
    """Enough of a model for the block-wise metrics, which only wrap and index."""

    class equation_system:  # noqa: N801 - mirrors the attribute it stands in for
        @staticmethod
        def assemble(*args, **kwargs):
            raise AssertionError("the stub model does not assemble")


def test_blockwise_criteria_replace_the_global_tolerances() -> None:
    """PorePy asserts that a criteria dictionary is not accompanied by the individual
    tolerances it would override, so leaving one behind aborts the run."""
    from solver_experiments.case import build_solver_params

    params = build_solver_params(
        SolverSpec(convergence_criteria="blockwise"), model=_StubModel()
    )
    assert set(params["nl_convergence_criteria"]) == {"inc", "res"}
    assert set(params["nl_divergence_criteria"]) == {
        "max_iter",
        "inc_nan",
        "res_nan",
        "inc_bound",
        "res_bound",
    }
    for key in params:
        assert not key.startswith(
            (
                "nl_convergence_inc",
                "nl_convergence_res",
                "nl_divergence_inc",
                "nl_divergence_res",
            )
        )
    assert "nl_max_iterations" not in params


def test_blockwise_criteria_split_increments_and_residuals_differently() -> None:
    """Increments are indexed by variable and residuals by equation; one metric used
    for both would group entries by the wrong blocks."""
    import porepy as pp

    from solver_experiments.case import build_solver_params
    from solver_experiments.convergence import RowScaledEquationMetric

    criteria = build_solver_params(
        SolverSpec(convergence_criteria="blockwise"), model=_StubModel()
    )["nl_convergence_criteria"]
    assert isinstance(criteria["inc"].metric, pp.VariableBasedEuclideanMetric)
    assert isinstance(criteria["res"].metric, RowScaledEquationMetric)


def test_blockwise_divergence_is_not_a_test_of_growth() -> None:
    """A fracture cell changing contact state moves its block nine orders within a
    step, so a bound relative to the block's own earlier value ends the attempt on a
    state change rather than on a runaway iterate."""
    from solver_experiments.case import build_solver_params

    spec = SolverSpec(convergence_criteria="blockwise")
    criteria = build_solver_params(spec, model=_StubModel())["nl_divergence_criteria"]
    for name in ("inc_bound", "res_bound"):
        assert criteria[name].tol == pytest.approx(spec.divergence_bound)
        assert not hasattr(criteria[name], "reference_value")


def test_the_default_unit_system_is_the_shipped_one(shipped) -> None:
    """The baseline must stay the shipped example, units included."""
    from porepy.examples.geothermal_reservoir import set_model_params

    example = set_model_params()["units"]
    assert shipped["units"].kg == example.kg
    assert shipped["units"].m == example.m
    assert shipped["units"].K == example.K


def test_a_named_unit_system_reaches_the_model() -> None:
    """A unit system that silently failed to apply would make every scaling
    measurement a measurement of the shipped one."""
    params = build_model_params(
        CaseSpec(units="pressure_temperature"), SolverSpec(), folder_name="unused"
    )
    assert params["units"].kg == 1.0e7
    assert params["units"].K == 1.0e2


def test_bounds_are_only_counted_by_default(shipped) -> None:
    """The baseline must not cap the Newton increment, or it is no longer shipped."""
    assert "variable_bounds" not in shipped


def test_capping_hands_the_admissibility_bounds_to_the_model() -> None:
    """Capping acts on the same bounds the telemetry counts against, in SI units,
    which the model converts itself."""
    params = build_model_params(
        CaseSpec(), SolverSpec(cap_to_bounds=True), folder_name="unused"
    )
    assert params["variable_bounds"] == ADMISSIBILITY_BOUNDS


def test_injection_duration_appends_one_point_holding_the_last_pressure(
    shipped,
) -> None:
    """Extending the run must change only its length, not the pressure history."""
    params = build_model_params(
        CaseSpec(injection_duration=30 * pp.YEAR), SolverSpec(), folder_name="unused"
    )
    shipped_times = shipped["time_manager"].schedule.get_array()
    times = params["time_manager"].schedule.get_array()
    assert np.array_equal(times[:-1], shipped_times)
    assert np.isclose(times[-1], shipped_times[1] + 30 * pp.YEAR)
    assert params["injection_well_pressures"] == [
        *shipped["injection_well_pressures"],
        shipped["injection_well_pressures"][-1],
    ]


def test_injection_duration_must_extend_the_shipped_schedule() -> None:
    with pytest.raises(ValueError):
        build_model_params(
            CaseSpec(injection_duration=pp.DAY), SolverSpec(), folder_name="unused"
        )


def test_dt_max_caps_the_initial_step_too() -> None:
    """The shipped initial step is three years, so a one-year bound must reach it, or
    the time manager would start above its own maximum."""
    params = build_model_params(
        CaseSpec(), SolverSpec(dt_max=pp.YEAR), folder_name="unused"
    )
    time_manager = params["time_manager"]
    assert time_manager.dt_min_max[1] == pp.YEAR
    assert time_manager.dt_init == pp.YEAR


@pytest.mark.parametrize(
    ("solver", "flux_owner", "discretization_owner"),
    [
        (SolverSpec(), "DarcysLaw", "DarcysLaw"),
        (
            SolverSpec(darcy_flux_discretization="tpfa"),
            "DarcysLaw",
            "FluxDiscretization",
        ),
        (
            SolverSpec(darcy_flux_discretization="tpfa", differentiable_darcy=True),
            "DarcysLawAd",
            "FluxDiscretization",
        ),
    ],
)
def test_darcy_flux_variants_take_precedence(
    solver, flux_owner, discretization_owner
) -> None:
    """A variant must supply the Darcy flux methods ahead of the shipped DarcysLaw, or
    the run silently uses the shipped flux and the comparison measures nothing."""
    from solver_experiments.case import _model_class

    model_class = _model_class(CaseSpec(), solver)

    def owner(method: str) -> str:
        return next(c for c in model_class.__mro__ if method in c.__dict__).__name__

    assert owner("darcy_flux") == flux_owner
    assert owner("darcy_flux_discretization") == discretization_owner


@pytest.mark.parametrize(
    ("tangential_contact", "owner_name"),
    [
        ("projection", "TangentialContactMechanicsEquation"),
        ("scaled_projection", "ScaledProjectionTangentialContactMechanicsEquation"),
    ],
)
def test_tangential_contact_formulation_takes_precedence(
    tangential_contact: str, owner_name: str
) -> None:
    """The scaled formulation must override the shipped one, or a comparison of the
    two formulations silently runs the shipped one twice."""
    from solver_experiments.case import _model_class

    solver = SolverSpec(tangential_contact=tangential_contact)
    model_class = _model_class(CaseSpec(), solver)
    owner = next(
        c
        for c in model_class.__mro__
        if "tangential_fracture_deformation_equation" in c.__dict__
    )
    assert owner.__name__ == owner_name


@pytest.mark.parametrize("depth", [0.8, 0.6])
def test_well_depth_sets_where_the_wells_end(depth: float) -> None:
    """The mixin must reproduce the shipped wells at the shipped depth, so that the
    geometry changes with the depth alone."""
    from porepy.applications.md_grids.model_geometries import TwoWells3d

    from solver_experiments.case import _WellDepth

    class Shipped(TwoWells3d):
        params = {"domain_sizes": np.array([1e3, 1e3, 1e3])}
        units = pp.Units()

    class Deepened(_WellDepth, Shipped):
        params = {**Shipped.params, "well_depth": depth}

    shipped, deepened = Shipped(), Deepened()
    shipped.set_wells()
    deepened.set_wells()
    for a, b in zip(shipped._wells, deepened._wells):
        assert np.allclose(b.pts[:, 0], a.pts[:, 0])
        assert np.allclose(b.pts[:2, 1], a.pts[:2, 1])
        assert np.isclose(b.pts[2, 1], -depth * 1e3)
        assert b.tags == a.tags
