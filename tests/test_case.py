"""Tests that a case spec means what it says.

The campaign's conclusions rest on every run differing from the shipped example only in
the axis under study, so the construction of model parameters is where a silent mistake
would be most expensive.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solver_experiments.case import CaseSpec, SolverSpec, build_model_params


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
