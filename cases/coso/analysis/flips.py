"""Changes of flow direction in the wells during the Newton iterations of attempts.

    python cases/coso/analysis/flips.py results/coso/<run>/rs3/telemetry_main.jsonl \
        17 18 19

For each listed attempt (indices as printed by ``attempts.py``), prints every
well-related flux that changes sign during the iterations, with its sign pattern, one
character per iteration (``+``, ``-``, or ``0`` for a flux below ``REL_FLOOR`` times its
largest magnitude in the attempt). The fluxes are:
- the Darcy flux on each face of each well, face 0 being the wellhead;
- the flux variable on each interface: ``interface_darcy_flux`` between a well and its
  junction (codimension 1), ``well_flux`` between a junction and its fracture
  (codimension 2).

The recorded fluxes are evaluated after the state update, which is also when PorePy
stores the fluxes its upwind discretisation uses. Each recorded sign is therefore the
upwind direction of the *next* iteration.

"""

from __future__ import annotations

import argparse

import numpy as np
from well_records import grid_labels, label, read_attempts

REL_FLOOR = 1e-8
"""A flux counts as zero below this fraction of its largest magnitude in the attempt."""


def well_fluxes(iteration: dict) -> dict[str, float]:
    """All directed well-related fluxes of one iteration, by readable label."""
    labels = grid_labels(iteration)
    out = {}
    for w in iteration["wells"]["wells"]:
        for face, (z, value) in enumerate(zip(w["face_z"], w["faces"]["darcy_flux"])):
            out[f"{w['well_name']} face{face} z={z:.0f}"] = value
    for interface in iteration["wells"]["interfaces"]:
        a, b = interface["subdomain_ids"]
        name = f"intf{interface['interface_id']} {label(labels, a)}/{label(labels, b)}"
        key = "interface_darcy_flux" if interface["codim"] == 1 else "well_flux"
        for k, value in enumerate(interface["variables"][key]["value"]):
            out[f"{name} [{k}]"] = value
    return out


def sign_pattern(series: list[float]) -> str:
    values = np.array(series)
    floor = REL_FLOOR * np.abs(values).max(initial=0.0)
    signs = np.where(np.abs(values) <= floor, 0, np.sign(values)).astype(int)
    return "".join({1: "+", -1: "-", 0: "0"}[s] for s in signs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry")
    parser.add_argument("attempts", type=int, nargs="+", help="Attempt indices")
    arguments = parser.parse_args()

    attempts = read_attempts(arguments.telemetry)
    for index in arguments.attempts:
        attempt = attempts[index]
        series: dict[str, list[float]] = {}
        for iteration in attempt.iterations:
            for name, value in well_fluxes(iteration).items():
                series.setdefault(name, []).append(value)
        patterns = {name: sign_pattern(values) for name, values in series.items()}
        changing = {
            name: p
            for name, p in patterns.items()
            if any(a != b for a, b in zip(p, p[1:]))
        }
        outcome = "successful" if attempt.successful else "failed"
        print(
            f"[{index}] {outcome}, {len(attempt.iterations)} iterations; "
            f"{len(changing)} of {len(series)} fluxes change sign"
        )
        for name, pattern in sorted(changing.items()):
            print(f"    {name:36s} {pattern}")


if __name__ == "__main__":
    main()
