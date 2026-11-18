"""Classify the failed attempts of a Coso run by where their temperature updates sit.

    python cases/coso/analysis/failures.py results/coso/<run>/rs3/telemetry_main.jsonl

For each failed attempt:
- where the largest temperature increment sits, counted over its iterations (the
  grid names of :func:`well_records.grid_labels`; ``g<id>`` is a matrix or fracture
  grid);
- which well and junction grids reach a temperature bound of the cap (within 0.1 K);
- the range of temperatures in the producers' top cells and junctions;
- the residual norm at the first iteration, its minimum and at the last iteration.

This is what separates the failure types: type A (residual met, junction temperatures
swinging, no cap reached) from type B (residual growing, top cells driven to the cap),
and failures that live in the fractures (largest increment in a ``g<id>`` grid).

"""

from __future__ import annotations

import argparse
import collections

from well_records import (
    DAY,
    PRODUCERS,
    TEMPERATURE_BOUNDS,
    grid_labels,
    grid_temperatures,
    junction_of,
    label,
    read_attempts,
    well,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry")
    arguments = parser.parse_args()

    for attempt in read_attempts(arguments.telemetry):
        if attempt.successful:
            continue
        iterations = attempt.iterations
        labels = grid_labels(iterations[0])
        locus = collections.Counter(
            label(
                labels, it["increments"]["temperature"]["argmax_location"]["domain_id"]
            )
            for it in iterations
        )
        at_bound = sorted(
            {
                label(labels, grid)
                for it in iterations
                for grid, values in grid_temperatures(it).items()
                if _at_bound(values)
            }
        )
        residuals = [it["convergence_info"]["res_abs"]["norm"] for it in iterations]
        print(
            f"[{attempt.index}] from {attempt.start / DAY:.4f} d, {len(iterations)} "
            f"iterations; largest T increment in {dict(locus.most_common(4))}; "
            f"at a bound: {at_bound or '-'}"
        )
        print(
            f"      producer top cell / junction T range {_producer_spans(iterations)};"
            f" residual first/min/last {residuals[0]:.2g}/{min(residuals):.2g}/"
            f"{residuals[-1]:.2g}"
        )


def _at_bound(values: list[float]) -> bool:
    lower, upper = TEMPERATURE_BOUNDS
    return any(v <= lower + 0.1 or v >= upper - 0.1 for v in values)


def _producer_spans(iterations: list[dict]) -> dict[str, tuple[int, int]]:
    """Minimum and maximum temperature [K] over the iterations, per producer."""

    def top_cell(it: dict, name: str) -> float:
        producer = well(it, name)
        top = max(range(len(producer["cell_z"])), key=lambda i: producer["cell_z"][i])
        return producer["variables"]["temperature"]["value"][top]

    def junction(it: dict, name: str) -> float:
        return junction_of(it, name)["variables"]["temperature"]["value"][0]

    spans = {}
    for name in PRODUCERS:
        for kind, value_of in (("top", top_cell), ("J", junction)):
            values = [value_of(it, name) for it in iterations]
            spans[f"{name} {kind}"] = (round(min(values)), round(max(values)))
    return spans


if __name__ == "__main__":
    main()
