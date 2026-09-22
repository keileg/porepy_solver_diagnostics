"""Tests that a case spec means what it says.

The campaign's conclusions rest on every run differing from the shipped example only in
the axis under study, so the construction of model parameters is where a silent mistake
would be most expensive.

"""

from __future__ import annotations

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
