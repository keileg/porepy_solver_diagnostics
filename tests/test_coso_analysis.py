"""Tests of the telemetry reading shared by the Coso analysis scripts.

Every count of attempts and iterations compared between Coso runs comes from
`read_attempts`. A failure here most likely means attempts are assigned to the wrong
time, so that runs compared over "the same" window are not.

"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).parents[1] / "cases" / "coso" / "analysis"


@pytest.fixture
def well_records(monkeypatch):
    monkeypatch.syspath_prepend(str(ANALYSIS))
    import well_records

    return well_records


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _iterations(count: int) -> list[dict]:
    return [{"record_type": "iteration"} for _ in range(count)]


def _success(time: float, dt: float) -> dict:
    return {"record_type": "time_step", "status": "successful", "time": time, "dt": dt}


FAILURE = {"record_type": "time_step", "status": "in_progress"}


def test_failed_attempts_start_with_their_retry(well_records, tmp_path):
    """A failed attempt starts where its successful retry starts, not where the
    preceding step started; and an unresolved trailing failure is dropped."""
    records = (
        _iterations(3)
        + [_success(0.0, 10.0)]
        + _iterations(20)
        + [FAILURE]
        + _iterations(20)
        + [FAILURE]
        + _iterations(4)
        + [_success(10.0, 2.0)]
        + _iterations(20)
        + [FAILURE]
    )
    attempts = well_records.read_attempts(_write(tmp_path / "t.jsonl", records))

    assert [(a.start, a.successful, len(a.iterations)) for a in attempts] == [
        (0.0, True, 3),
        (10.0, False, 20),
        (10.0, False, 20),
        (10.0, True, 4),
    ]
    assert [a.index for a in attempts] == [0, 1, 2, 3]
    window = well_records.attempts_starting_in(attempts, 5.0 / 86400, 1.0)
    assert [a.index for a in window] == [1, 2, 3]


def test_junctions_are_named_after_their_well(well_records):
    """Junctions take their well's name, numbered from the top if there are several."""
    iteration = {
        "wells": {
            "wells": [
                {"domain_id": 1, "well_name": "inj"},
                {"domain_id": 2, "well_name": "prod"},
            ],
            "junctions": [
                {"domain_id": 5, "z": -2000.0},
                {"domain_id": 6, "z": -1000.0},
                {"domain_id": 7, "z": -3000.0},
            ],
            "interfaces": [
                {"codim": 1, "subdomain_ids": [1, 5]},
                {"codim": 1, "subdomain_ids": [1, 6]},
                {"codim": 1, "subdomain_ids": [2, 7]},
                {"codim": 2, "subdomain_ids": [9, 7]},
            ],
        }
    }
    labels = well_records.grid_labels(iteration)

    assert labels == {1: "inj", 2: "prod", 6: "J inj 1", 5: "J inj 2", 7: "J prod"}
    assert well_records.label(labels, 9) == "g9"
