"""Tests of the metrics derived from telemetry records.

Every number reported about a run comes out of these functions, so a failure here means
the campaign's conclusions are wrong rather than merely its formatting.

"""

from __future__ import annotations

import pytest

from solver_experiments.metrics import (
    diagnosis,
    group_by_attempt,
    iteration_statistics,
    time_step_statistics,
)


def iteration(
    index: int,
    residual: float = 1.0,
    *,
    contact: dict | None = None,
    cycle: dict | None = None,
    alpha: float | None = None,
) -> dict:
    """Build one `"iteration"` record with only the fields under test."""
    return {
        "record_type": "iteration",
        "iteration_index": index,
        "blocks": {
            "mass_balance_equation": {"residual_inf_norm": residual},
            "momentum_balance_equation": {"residual_inf_norm": residual / 10},
        },
        "contact_state": contact,
        "cycle": cycle,
        "line_search": None if alpha is None else {"alpha": alpha},
        "admissibility": {"temperature": {"num_below": 1, "num_above": 0}},
    }


def time_step(status: str, dt: float = 1.0) -> dict:
    """Build one `"time_step"` record."""
    return {"record_type": "time_step", "status": status, "dt": dt}


@pytest.fixture
def records() -> list[dict]:
    """A run of two attempts: one rejected after two iterations, one accepted."""
    return [
        iteration(1),
        iteration(2),
        time_step("in_progress", dt=100.0),
        iteration(1),
        iteration(2),
        iteration(3),
        time_step("successful", dt=50.0),
    ]


def test_attempts_are_delimited_by_time_step_records(records):
    """Iterations must be attributed to the attempt they preceded, not the next one."""
    attempts = group_by_attempt(records)
    assert [len(attempt["iterations"]) for attempt in attempts] == [2, 3]
    assert attempts[1]["time_step"]["status"] == "successful"


def test_iterations_of_rejected_attempts_count_as_wasted(records):
    """The wasted fraction is the whole point of per-attempt grouping."""
    statistics = iteration_statistics(records)
    assert statistics["total"] == 5
    assert statistics["wasted"] == 2
    assert statistics["wasted_fraction"] == pytest.approx(0.4)
    assert statistics["per_successful_step"]["max"] == 3


def test_only_successful_attempts_count_as_accepted_steps(records):
    """`"in_progress"` is a rejected attempt, not an accepted step."""
    statistics = time_step_statistics(records)
    assert statistics["num_accepted"] == 1
    assert statistics["num_failed_attempts"] == 1
    assert statistics["mean_accepted_dt"] == pytest.approx(50.0)


def test_trailing_iterations_without_a_time_step_record_are_kept():
    """A crash mid-attempt leaves iterations with no time-step record.

    They still count as wasted work.

    """
    attempts = group_by_attempt([iteration(1), iteration(2)])
    assert len(attempts) == 1
    assert attempts[0]["time_step"] is None
    assert iteration_statistics([iteration(1), iteration(2)])["wasted"] == 2


def test_dominant_block_is_the_one_with_the_largest_residual(records):
    """Which block dominates is how the failure mode is classified."""
    counts = diagnosis(records)["dominant_residual_blocks"]
    assert next(iter(counts)) == "mass_balance_equation"


def test_settled_active_set_is_not_reported_as_cycling():
    """`repeat_length` 1 means the active set stopped changing, i.e. healthy."""
    settled = [
        iteration(
            i,
            cycle={"repeat_length": 1, "num_flipped_cells": 0, "chattering_cells": []},
        )
        for i in (1, 2, 3)
    ]
    assert diagnosis(settled)["cycling"]["num_iterations_cycling"] == 0

    cycling = settled + [
        iteration(
            4,
            cycle={
                "repeat_length": 3,
                "num_flipped_cells": 5,
                "chattering_cells": [1, 2],
            },
        )
    ]
    summary = diagnosis(cycling)["cycling"]
    assert summary["num_iterations_cycling"] == 1
    assert summary["num_iterations_with_chattering"] == 1


def test_contact_and_line_search_fields_are_absent_when_not_measured():
    """On a model without the contact indicators the fields are null, not zero."""
    result = diagnosis([iteration(1)])
    assert result["contact_state"] is None
    assert result["cycling"] is None
    assert result["line_search"] is None


def test_line_search_summary_reports_the_deepest_backtrack():
    """A collapsing alpha is the field that separates mode B from mode A."""
    records = [iteration(1, alpha=1.0), iteration(2, alpha=0.02)]
    assert diagnosis(records)["line_search"]["min_alpha"] == pytest.approx(0.02)
