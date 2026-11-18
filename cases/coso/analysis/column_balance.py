"""Pressure balance of each producer column at the accepted steps of opening 1.

    python cases/coso/analysis/column_balance.py \
        results/coso/<run>/rs3/telemetry_main.jsonl [--last-day 1.9346]

With a column as permeable as a Coso well, the sign of the net flow follows

    excess = p_fracture - p_wellhead - weight,

where p_fracture is the pressure of the fracture cell holding the producer's junction,
p_wellhead the imposed wellhead pressure, and weight the column's hydrostatic load
from its cell densities. For each accepted step from the start of the opening-1 ramp,
prints the excess, the step-to-step change of each of its three terms, the wellhead
Darcy flux and the column's top/middle/bottom temperatures. The changes show which
term drives a change of direction or rate.

Each fracture cell holding a producer junction sits 11-15 m above it, which puts a
constant 100-135 kPa offset into the excess; read the changes, not the absolute value.

The wellhead pressure is not in the telemetry. It is reconstructed from the ramp of
``SmoothWellTransitions`` in the Coso scripts, linear over ``RAMP`` seconds from the
wellhead pressure cached at the ramp start to atmospheric pressure. The cached values
(``SHUT_IN_WELLHEAD``) were read from the model of the reference run at 165,600 s; they
hold for opening 1 of ``runscript_3`` only.

"""

from __future__ import annotations

import argparse

from well_records import (
    DAY,
    PRODUCERS,
    attempts_starting_in,
    junction_fracture_interface,
    junction_of,
    read_attempts,
    well,
)

GRAVITY = 9.80665
OPENING_1 = 165600.0
"""Start of the opening-1 ramp [s]."""
RAMP = 7200.0
"""Ramp length [s]."""
ATMOSPHERIC = 101325e-9
"""Target wellhead pressure of the producers [GPa]."""
SHUT_IN_WELLHEAD = {"16A-20": 1.0142385194562213e-4, "16B-20": 1.0596413993185652e-4}
"""Wellhead pressure [GPa] at the start of the opening-1 ramp, per producer."""


def balance(iteration: dict, name: str, time: float) -> dict[str, float]:
    """The terms of the column balance [GPa], and the column's temperatures [K]."""
    producer = well(iteration, name)
    density = producer["fluid"]["density"]
    faces_top_down = sorted(producer["face_z"], reverse=True)
    lengths = [a - b for a, b in zip(faces_top_down[:-1], faces_top_down[1:])]
    cells_top_down = sorted(
        range(len(producer["cell_z"])), key=lambda i: -producer["cell_z"][i]
    )
    weight = sum(density[i] * GRAVITY * h for i, h in zip(cells_top_down, lengths))
    progress = min(max((time - OPENING_1) / RAMP, 0.0), 1.0)
    wellhead = (1 - progress) * SHUT_IN_WELLHEAD[name] + progress * ATMOSPHERIC
    junction_id = junction_of(iteration, name)["domain_id"]
    interface = junction_fracture_interface(iteration, junction_id)
    fracture = interface["primary_cells"]["variables"]["pressure"]["value"][0]
    temperature = producer["variables"]["temperature"]["value"]
    return {
        "fracture": fracture,
        "wellhead": wellhead,
        "weight": weight,
        "wellhead_flux": producer["faces"]["darcy_flux"][0],
        "temperatures": [temperature[i] for i in cells_top_down],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry")
    parser.add_argument(
        "--last-day", type=float, default=(OPENING_1 + RAMP) / DAY, help="[d]"
    )
    arguments = parser.parse_args()

    window = attempts_starting_in(
        read_attempts(arguments.telemetry), OPENING_1 / DAY, arguments.last_day
    )
    print(
        "attempt  end-open[s]  well      excess[Pa]  d(p_frac)[Pa]  d(p_head)[Pa]  "
        "-d(weight)[Pa]  wellhead flux  T top/mid/bottom [K]"
    )
    previous: dict[str, dict[str, float]] = {}
    for attempt in window:
        if not attempt.successful:
            continue
        end = attempt.start + attempt.dt
        for name in PRODUCERS:
            terms = balance(attempt.iterations[-1], name, end)
            excess = terms["fracture"] - terms["wellhead"] - terms["weight"]
            changes = " " * 44
            if name in previous:
                before = previous[name]
                changes = (
                    f"{(terms['fracture'] - before['fracture']) * 1e9:+13.1f}  "
                    f"{(terms['wellhead'] - before['wellhead']) * 1e9:+13.1f}  "
                    f"{-(terms['weight'] - before['weight']) * 1e9:+14.1f}"
                )
            previous[name] = terms
            temperatures = terms["temperatures"]
            print(
                f"[{attempt.index:4d}] {end - OPENING_1:9.1f}  {name}  "
                f"{excess * 1e9:+12.1f}  {changes}  {terms['wellhead_flux']:+.2e}  "
                f"{temperatures[0]:.0f}/{temperatures[len(temperatures) // 2]:.0f}/"
                f"{temperatures[-1]:.0f}"
            )


if __name__ == "__main__":
    main()
