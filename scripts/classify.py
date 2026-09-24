"""Classify the time-step attempts of finished runs from their telemetry.

Reclassifies without rerunning, e.g. after a threshold changes, and prints every
costly attempt with its labels.

Example:
    python scripts/classify.py results/long_temperature --expensive-iterations 10

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import porepy as pp

from solver_experiments.metrics import classification_summary


def main() -> None:
    """Print the classification of every run under a results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expensive-iterations", type=int, default=10)
    arguments = parser.parse_args()
    for telemetry in sorted(arguments.directory.rglob("telemetry.jsonl")):
        summary = classification_summary(
            pp.solvers.read_jsonl_records(telemetry), arguments.expensive_iterations
        )
        print(f"== {telemetry.parent.name}")
        print(f"   costs {summary['cost_counts']}, labels {summary['label_counts']}")
        for attempt in summary["costly_attempts"]:
            labels = json.dumps(attempt["labels"], default=str)
            print(
                f"   attempt {attempt['attempt']} {attempt['cost']}, "
                f"{attempt['iterations']} iterations, dt {attempt['dt']}: {labels}"
            )
        for stretch in summary["step_size_insensitive"]:
            print(f"   step-size insensitive: {stretch}")


if __name__ == "__main__":
    main()
