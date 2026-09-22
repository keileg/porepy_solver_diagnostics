"""Turn a set of metrics files into a comparison table.

The table is keyed by case and solver configuration, and shows the primary metrics
(did the run finish, and what did it cost) next to the explanatory ones that say why.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COLUMNS: list[tuple[str, str]] = [
    ("case", "{case_id}"),
    ("solver", "{solver_id}"),
    ("ok", "{success}"),
    ("run_s", "{run_time:.0f}"),
    ("steps", "{num_accepted}"),
    ("att", "{num_attempts}"),
    ("iters", "{total_iterations}"),
    ("waste", "{wasted_fraction:.2f}"),
    ("it/step", "{iterations_per_step}"),
    ("lin_s", "{linear_solve_time:.0f}"),
    ("max_bwd_err", "{max_backward_error:.1e}"),
    ("max_|dx|", "{max_update_norm:.1e}"),
    ("slip", "{max_slip}"),
    ("open", "{max_open}"),
    ("cycl", "{num_iterations_cycling}"),
    ("dominant block", "{dominant_block}"),
]
"""Column headers and their format strings, applied to :func:`flatten`'s output."""


def load_metrics(results_directory: Path) -> list[dict[str, Any]]:
    """Read every `metrics.json` under a results directory, newest path order last."""
    return [
        json.loads(path.read_text())
        for path in sorted(results_directory.rglob("metrics.json"))
    ]


def flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    """Pick the comparison-relevant fields out of one metrics file."""
    contact = metrics["diagnosis"].get("contact_state") or {}
    cycling = metrics["diagnosis"].get("cycling") or {}
    blocks = metrics["diagnosis"].get("dominant_residual_blocks") or {}
    per_step = metrics["iterations"].get("per_successful_step") or {}
    return {
        "case_id": metrics["case_id"],
        "solver_id": metrics["solver_id"],
        "success": "yes" if metrics["success"] else "NO",
        "run_time": metrics["timing"]["run_time"],
        "num_accepted": metrics["time_steps"].get("num_accepted", 0),
        "num_attempts": metrics["time_steps"].get("num_attempts", 0),
        "total_iterations": metrics["iterations"].get("total", 0),
        "wasted_fraction": metrics["iterations"].get("wasted_fraction", 0.0),
        "iterations_per_step": _format_spread(per_step),
        "linear_solve_time": metrics["timing"]["linear_solve_time"],
        "max_backward_error": metrics["linear_solver"].get(
            "max_backward_error", float("nan")
        ),
        "max_update_norm": metrics["linear_solver"].get(
            "max_update_norm", float("nan")
        ),
        "max_slip": contact.get("max_slip", "-"),
        "max_open": contact.get("max_open", "-"),
        "num_iterations_cycling": cycling.get("num_iterations_cycling", "-"),
        "dominant_block": next(iter(blocks), "-").removesuffix("_equation"),
    }


def _format_spread(per_step: dict[str, Any]) -> str:
    """Render iterations per successful step as `median (max)`."""
    if per_step.get("median") is None:
        return "-"
    return f"{per_step['median']:.0f} ({per_step['max']})"


def format_table(metrics_files: list[dict[str, Any]]) -> str:
    """Render the comparison table."""
    rows = [flatten(metrics) for metrics in metrics_files]
    cells = [[header for header, _ in COLUMNS]]
    for row in rows:
        cells.append([template.format(**row) for _, template in COLUMNS])
    widths = [max(len(row[i]) for row in cells) for i in range(len(COLUMNS))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in cells
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)
