"""Metrics derived from one run: cost, robustness, and what the solver struggled with.

The primary metrics are time-to-solution and whether the run reached its end time; the
explanatory metrics say why. Most of them come from the solver telemetry PorePy writes
during the run, so the derivation here is grouping and summarising, never recomputation.

"""

from __future__ import annotations

import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import porepy as pp

_ACCEPTED = "successful"
"""`TimeStepperStatusSuccess.serialize()`: the status of an accepted attempt.

The other two values a per-attempt record can carry are `"in_progress"` (the attempt
failed and the stepper will retry with a smaller step) and `"failed"` (the stepper gave
up), so anything but this one is a rejected attempt whose iterations were wasted.

"""


@dataclass
class RunMetrics:
    """Everything recorded about one run.

    Parameters:
        case_id: Identifier of the case, from
            :attr:`~solver_experiments.case.CaseSpec.case_id`.
        solver_id: Identifier of the solver configuration.
        success: Whether the run reached its scheduled end time.
        failure_reason: Why it stopped, if it did not.
        provenance: Code versions, machine and tolerances, so the numbers stay
            interpretable once any of those move.

    """

    case_id: str
    solver_id: str
    success: bool
    failure_reason: str | None
    case: dict[str, Any]
    solver: dict[str, Any]
    timing: dict[str, Any]
    linear_solver: dict[str, Any]
    time_steps: dict[str, Any]
    iterations: dict[str, Any]
    diagnosis: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the metrics as plain data, ready to be written as JSON."""
        return asdict(self)


def collect_metrics(
    case_id: str,
    solver_id: str,
    case: dict[str, Any],
    solver: dict[str, Any],
    success: bool,
    failure_reason: str | None,
    timing: dict[str, Any],
    linear_solver: dict[str, Any],
    telemetry_path: Path | None,
) -> RunMetrics:
    """Assemble the metrics for one finished run."""
    records = _read_records(telemetry_path)
    return RunMetrics(
        case_id=case_id,
        solver_id=solver_id,
        success=success,
        failure_reason=failure_reason,
        case=case,
        solver=solver,
        timing=timing,
        linear_solver=linear_solver,
        time_steps=time_step_statistics(records),
        iterations=iteration_statistics(records),
        diagnosis=diagnosis(records),
        provenance=provenance(),
    )


def _read_records(telemetry_path: Path | None) -> list[dict[str, Any]]:
    """Read the telemetry records of a run, tolerating a run without telemetry."""
    if telemetry_path is None or not telemetry_path.exists():
        return []
    return pp.solvers.read_jsonl_records(telemetry_path)


def group_by_attempt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group iteration records into the time-step attempt they belong to.

    Telemetry is one append-only stream, and a `"time_step"` record is written when an
    attempt concludes. So the iteration records preceding one belong to that attempt,
    which is what makes wasted-iteration accounting possible at all.

    Returns:
        One entry per attempt, with its iteration records and its time-step record.

    """
    attempts: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for record in records:
        if record["record_type"] == "iteration":
            pending.append(record)
        elif record["record_type"] == "time_step":
            attempts.append({"iterations": pending, "time_step": record})
            pending = []
    if pending:
        attempts.append({"iterations": pending, "time_step": None})
    return attempts


def time_step_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the time-step history: accepted steps, chops, and step sizes."""
    attempts = group_by_attempt(records)
    if not attempts:
        return {"num_attempts": 0}
    statuses = [
        (attempt["time_step"] or {}).get("status", "unknown") for attempt in attempts
    ]
    accepted_dts = [
        attempt["time_step"]["dt"]
        for attempt, status in zip(attempts, statuses, strict=True)
        if attempt["time_step"] is not None and status == _ACCEPTED
    ]
    return {
        "num_attempts": len(attempts),
        "num_accepted": sum(status == _ACCEPTED for status in statuses),
        "num_failed_attempts": sum(status != _ACCEPTED for status in statuses),
        "mean_accepted_dt": float(np.mean(accepted_dts)) if accepted_dts else None,
        "min_accepted_dt": float(np.min(accepted_dts)) if accepted_dts else None,
        "max_accepted_dt": float(np.max(accepted_dts)) if accepted_dts else None,
        "status_counts": {status: statuses.count(status) for status in set(statuses)},
    }


def iteration_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise nonlinear iteration counts, including the wasted fraction.

    Spread matters as much as the centre: a configuration that is cheap on average but
    occasionally needs the iteration cap is not robust, and the mean hides that.

    """
    attempts = group_by_attempt(records)
    if not attempts:
        return {"total": 0}
    counts = np.array([len(attempt["iterations"]) for attempt in attempts])
    successful = np.array(
        [
            len(attempt["iterations"])
            for attempt in attempts
            if (attempt["time_step"] or {}).get("status") == _ACCEPTED
        ]
    )
    wasted = int(counts.sum() - successful.sum())
    return {
        "total": int(counts.sum()),
        "wasted": wasted,
        "wasted_fraction": float(wasted / counts.sum()) if counts.sum() else 0.0,
        "per_successful_step": {
            "mean": float(successful.mean()) if successful.size else None,
            "median": float(np.median(successful)) if successful.size else None,
            "max": int(successful.max()) if successful.size else None,
            "iqr": float(np.subtract(*np.percentile(successful, [75, 25])))
            if successful.size
            else None,
        },
    }


def diagnosis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise what the telemetry says the solver struggled with.

    The fields follow the nonlinear failure-mode taxonomy: which equation block holds
    the largest residual (scaling/coupling), how the contact state evolves and whether
    it cycles (active-set combinatorics), admissibility violations (unphysical
    excursion), and how far the line search had to back off.

    """
    iterations = [r for r in records if r["record_type"] == "iteration"]
    if not iterations:
        return {"num_iterations": 0}
    return {
        "num_iterations": len(iterations),
        "dominant_residual_blocks": _dominant_block_counts(iterations),
        "contact_state": _contact_state_summary(iterations),
        "cycling": _cycling_summary(iterations),
        "admissibility": _admissibility_summary(iterations),
        "line_search": _line_search_summary(iterations),
    }


def _dominant_block_counts(iterations: list[dict[str, Any]]) -> dict[str, int]:
    """Count how often each equation block holds the largest residual norm."""
    counts: dict[str, int] = {}
    for record in iterations:
        blocks = record.get("blocks") or {}
        if not blocks:
            continue
        worst = max(blocks, key=lambda name: blocks[name]["residual_inf_norm"])
        counts[worst] = counts.get(worst, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _contact_state_summary(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarise the open/stick/slip split over the run, or None if not measured."""
    states = [r["contact_state"] for r in iterations if r.get("contact_state")]
    if not states:
        return None
    return {
        "num_total": states[-1]["num_total"],
        "max_open": max(s["num_open"] for s in states),
        "max_slip": max(s["num_slip"] for s in states),
        "min_stick": min(s["num_stick"] for s in states),
        "max_near_opening_transition": max(
            s["num_near_opening_transition"] for s in states
        ),
        "max_near_sliding_transition": max(
            s["num_near_sliding_transition"] for s in states
        ),
    }


def _cycling_summary(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarise active-set cycling.

    A `repeat_length` of 1 means the active set stopped changing, which is what a
    converging solve looks like; only 2 or more means the solver left a state and came
    back, so only those are counted here.

    """
    cycles = [r["cycle"] for r in iterations if r.get("cycle")]
    if not cycles:
        return None
    repeats = [c["repeat_length"] for c in cycles if c["repeat_length"] is not None]
    return {
        "num_iterations_cycling": sum(1 for r in repeats if r >= 2),
        "max_repeat_length": max(repeats) if repeats else None,
        "max_flipped_cells": max(c["num_flipped_cells"] for c in cycles),
        "num_iterations_with_chattering": sum(
            1 for c in cycles if c["chattering_cells"]
        ),
    }


def _admissibility_summary(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Count iterates outside the configured variable bounds, per variable."""
    summary: dict[str, Any] = {}
    for record in iterations:
        for name, counts in (record.get("admissibility") or {}).items():
            entry = summary.setdefault(name, {"max_below": 0, "max_above": 0})
            entry["max_below"] = max(entry["max_below"], counts["num_below"])
            entry["max_above"] = max(entry["max_above"], counts["num_above"])
    return summary or None


def _line_search_summary(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarise how far the line search had to back off."""
    alphas = [
        r["line_search"]["alpha"]
        for r in iterations
        if r.get("line_search") and r["line_search"].get("alpha") is not None
    ]
    if not alphas:
        return None
    return {
        "min_alpha": float(np.min(alphas)),
        "median_alpha": float(np.median(alphas)),
        "num_iterations_below_0.5": int(np.sum(np.array(alphas) < 0.5)),
    }


def provenance() -> dict[str, Any]:
    """Record what the numbers depend on besides the case and solver configuration."""
    return {
        "porepy_commit": _git_description(Path(pp.__file__).parents[2]),
        "pp_solvers_commit": _git_description(Path("/workdir/pp_solvers")),
        "telemetry_schema_version": pp.solvers.SCHEMA_VERSION,
        "acceptance_criterion": "porepy-defaults-unfrozen",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _git_description(repository: Path) -> str | None:
    """Return `<commit>[-dirty]` for a repository, or None if it is not one."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return f"{commit}-dirty" if dirty else commit
