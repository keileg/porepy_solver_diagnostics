"""Reading the Coso runs' telemetry by attempt, and naming the well grids in it.

Shared by the analysis scripts in this directory. Only NumPy and the standard library
are needed, so the scripts run without PorePy.

The telemetry is PorePy's solver telemetry, one JSON record per line: an
``"iteration"`` record per Newton iteration, carrying a ``"wells"`` section when the
model has wells, and a ``"time_step"`` record when an attempt concludes. All values are
in model units, which for Coso are ``pp.Units(kg=1e9)``: pressure in GPa, density in
1e9 kg/m^3, temperature in K. Times are in seconds.

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DAY = 86400.0

PRODUCERS = ("16A-20", "16B-20")
"""Well names of the two producers of ``runscript_3``."""

TEMPERATURE_BOUNDS = (273.16, 647.0)
"""The cap's temperature bounds [K] in the Coso runs."""


@dataclass
class Attempt:
    """One Newton solve of one time step, with the iteration records it produced."""

    index: int
    """Position of the attempt in the telemetry file, counting from 0."""
    start: float
    """Simulation time at the start of the attempted step [s]."""
    successful: bool
    dt: float | None
    """Size of the attempted step [s]; not recorded for a failed attempt."""
    iterations: list[dict[str, Any]] = field(repr=False)


def read_attempts(telemetry: Path | str) -> list[Attempt]:
    """Group a telemetry file into attempts, with the start time of each.

    A ``"time_step"`` record closes the attempt whose iteration records precede it.
    A successful one carries the step's start time and size. A failed one carries
    neither, but its retry starts at the same time, so it takes the start time of the
    next successful attempt. Attempts after the last successful one are dropped, since
    their start is unknown.

    Parameters:
        telemetry: Path of a ``telemetry_main.jsonl`` written by the Coso driver.

    Returns:
        The attempts in the order they were made.

    """
    attempts: list[Attempt] = []
    iterations: list[dict[str, Any]] = []
    failed: list[list[dict[str, Any]]] = []
    with open(telemetry) as stream:
        for line in stream:
            record = json.loads(line)
            if record["record_type"] == "iteration":
                iterations.append(record)
                continue
            if record["status"] != "successful":
                failed.append(iterations)
            else:
                start = record["time"]
                for failed_iterations in failed:
                    attempts.append(
                        Attempt(len(attempts), start, False, None, failed_iterations)
                    )
                attempts.append(
                    Attempt(len(attempts), start, True, record["dt"], iterations)
                )
                failed = []
            iterations = []
    return attempts


def attempts_starting_in(
    attempts: list[Attempt], first_day: float, last_day: float
) -> list[Attempt]:
    """The attempts whose step starts in ``[first_day, last_day)``, times in days."""
    return [a for a in attempts if first_day * DAY <= a.start < last_day * DAY]


def grid_labels(iteration: dict[str, Any]) -> dict[int, str]:
    """Readable names of the well and junction grids, by subdomain id.

    Wells keep their well name. A junction (the 0-d grid where a well crosses a
    fracture) is named after the well it belongs to, found from the codimension-1
    interface between the two, and numbered from the top when a well has several:
    ``"J 68-20RD 1"``, ``"J 68-20RD 2"``, ``"J 16A-20"``.

    Parameters:
        iteration: Any iteration record with a ``"wells"`` section.

    """
    wells = iteration["wells"]
    labels = {well["domain_id"]: well["well_name"] for well in wells["wells"]}
    depth = {junction["domain_id"]: junction["z"] for junction in wells["junctions"]}
    junctions_of: dict[str, list[int]] = {}
    for interface in wells["interfaces"]:
        if interface["codim"] != 1:
            continue
        well_id, junction_id = interface["subdomain_ids"]
        if junction_id in depth:
            junctions_of.setdefault(labels[well_id], []).append(junction_id)
    for well_name, junction_ids in junctions_of.items():
        top_down = sorted(junction_ids, key=lambda j: -depth[j])
        for number, junction_id in enumerate(top_down, start=1):
            suffix = f" {number}" if len(top_down) > 1 else ""
            labels[junction_id] = f"J {well_name}{suffix}"
    return labels


def label(labels: dict[int, str], domain_id: int) -> str:
    """The readable name of a grid, or ``g<id>`` for one that is not a well grid."""
    return labels.get(domain_id, f"g{domain_id}")


def well(iteration: dict[str, Any], name: str) -> dict[str, Any]:
    """The well section of one iteration record, by well name."""
    return next(w for w in iteration["wells"]["wells"] if w["well_name"] == name)


def junction_of(iteration: dict[str, Any], well_name: str) -> dict[str, Any]:
    """The junction section of a well with a single junction."""
    labels = grid_labels(iteration)
    return next(
        j
        for j in iteration["wells"]["junctions"]
        if labels[j["domain_id"]] == f"J {well_name}"
    )


def junction_fracture_interface(
    iteration: dict[str, Any], junction_id: int
) -> dict[str, Any]:
    """The codimension-2 interface between a junction and the fracture holding it."""
    return next(
        i
        for i in iteration["wells"]["interfaces"]
        if i["codim"] == 2 and junction_id in i["subdomain_ids"]
    )


def grid_temperatures(iteration: dict[str, Any]) -> dict[int, list[float]]:
    """Cell temperatures [K] of every well and junction grid, by subdomain id."""
    wells = iteration["wells"]
    return {
        grid["domain_id"]: grid["variables"]["temperature"]["value"]
        for grid in wells["wells"] + wells["junctions"]
    }
