"""List the attempts of a Coso run in a time window, with totals.

Used to compare runs over the same window, e.g. the opening-1 restarts::

    python cases/coso/analysis/attempts.py results/coso/<run>/rs3/telemetry_main.jsonl \
        1.9166 1.931

Each line gives the attempt's index, start time, outcome, step size (successful
attempts only) and Newton iterations. The last line counts accepted and failed
attempts and their iterations. An attempt belongs to the window if its step starts in
it, so runs stopped at the same time are compared over the same attempts.

"""

from __future__ import annotations

import argparse

from well_records import DAY, attempts_starting_in, read_attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry")
    parser.add_argument("first_day", type=float, help="Window start [d]")
    parser.add_argument("last_day", type=float, help="Window end [d], exclusive")
    arguments = parser.parse_args()

    window = attempts_starting_in(
        read_attempts(arguments.telemetry), arguments.first_day, arguments.last_day
    )
    for attempt in window:
        outcome = f"ok  dt {attempt.dt:9.2f} s" if attempt.successful else "FAILED"
        print(
            f"[{attempt.index:4d}] from {attempt.start / DAY:.5f} d  {outcome:18s}"
            f"  {len(attempt.iterations)} iterations"
        )
    accepted = sum(attempt.successful for attempt in window)
    iterations = sum(len(attempt.iterations) for attempt in window)
    print(
        f"accepted {accepted}, failed {len(window) - accepted}, iterations {iterations}"
    )


if __name__ == "__main__":
    main()
