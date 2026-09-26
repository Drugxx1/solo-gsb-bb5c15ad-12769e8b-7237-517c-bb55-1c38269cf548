"""End-to-end tests for bounded extrema downsampling on /api/timeline.

The old fixed-stride sampling returned more points than ``max_points`` and
could drop an isolated spike, flattening anomalies.  These tests pin the new
bounded extrema behaviour and make sure count/shown metadata plus the other
timeline layers (faults, maintenance, suppression, annotations) keep using
the original, unsampled data.
"""

import json
import os
import tempfile

import pytest

from app import db, services


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def _import_values(values, device="d", metric="m"):
    rows = [
        {
            "device": device,
            "metric": metric,
            "ts": f"2026-09-23T00:{m:02d}:00Z",
            "value": v,
        }
        for m, v in enumerate(values)
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows))


def _minute(ts: float) -> int:
    """Minute of the timestamp relative to the 00:00 base of the test series."""
    from app.engine import to_epoch

    return int(round((ts - to_epoch("2026-09-23T00:00:00Z")) / 60))


def test_timeline_bounded_extrema_keeps_spike_and_respects_budget(client):
    # 20 minute-spaced samples; an isolated spike at minute 7.
    values = [99 if m == 7 else 10 for m in range(20)]
    _import_values(values)

    body = client.get("/api/timeline", params={"max_points": 4}).json()
    ch = body["channels"][0]
    assert len(ch["values"]) <= 4
    timestamps_min = sorted(_minute(p["ts"]) for p in ch["values"])
    # first/last samples in the queried range and the isolated spike survive
    assert timestamps_min[0] == 0
    assert timestamps_min[-1] == 19
    assert 7 in timestamps_min
    spike = next(p for p in ch["values"] if _minute(p["ts"]) == 7)
    assert spike["v"] == 99
    # returned points stay in chronological (ts, event-id) order
    keys = [(p["ts"], p["id"]) for p in ch["values"]]
    assert keys == sorted(keys)


def test_timeline_under_budget_returns_all_samples(client):
    values = [10, 11, 12, 13]
    _import_values(values)
    body = client.get("/api/timeline", params={"max_points": 400}).json()
    ch = body["channels"][0]
    assert len(ch["values"]) == 4
    assert [p["v"] for p in ch["values"]] == values


def test_timeline_count_and_shown_use_original_data(client):
    values = [99 if m == 7 else 10 for m in range(20)]
    _import_values(values)
    body = client.get("/api/timeline", params={"max_points": 4}).json()
    ch = body["channels"][0]
    assert ch["count"] == 20  # all stored channel events
    assert ch["shown"] == 20  # all samples in range, before sampling
    assert len(ch["values"]) <= 4  # only the drawn series is sampled


def test_timeline_faults_and_maintenance_layers_unsampled(client):
    values = [90 if 6 <= m <= 10 else 10 for m in range(20)]
    _import_values(values)
    db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("尖峰规则", "AND", 0, 0.0),
    )
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (1, 'd', 'm', '>=', 80, 0)"
    )
    services.evaluate_all_rules()
    # Maintenance covering part of the true segment -> one suppressed piece.
    from app.engine import to_epoch

    db.execute(
        "INSERT INTO maintenance_windows (device, name, start_ts, end_ts, note, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (
            "d",
            "巡检",
            to_epoch("2026-09-23T00:08:00Z"),
            to_epoch("2026-09-23T00:09:00Z"),
            None,
            0.0,
        ),
    )
    services.evaluate_all_rules()

    body = client.get("/api/timeline", params={"max_points": 4}).json()
    # sampled series
    assert len(body["channels"][0]["values"]) <= 4
    # other layers remain full-precision original data; maintenance subtracts
    # [8,9) from the true segment [6,11), splitting it into two fault intervals.
    assert len(body["faults"]) == 2
    assert [(_minute(f["start"]), _minute(f["end"])) for f in body["faults"]] == [
        (6, 8),
        (9, 11),
    ]
    assert len(body["maintenance"]) == 1
    assert body["maintenance"][0]["name"] == "巡检"
    assert len(body["suppressed"]) == 1
    supp = body["suppressed"][0]
    assert (_minute(supp["start"]), _minute(supp["end"])) == (8, 9)
