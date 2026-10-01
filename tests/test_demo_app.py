"""Demo server end to end: investigate -> report -> audit -> feedback -> monitoring."""

import json

import pytest
from fastapi.testclient import TestClient

import demo.app as demo_app
from copilot.audit import AuditLog


@pytest.fixture(scope="module")
def client():
    demo_app.STATE["audit"] = AuditLog(":memory:")
    with TestClient(demo_app.app) as c:
        yield c
    demo_app.STATE["audit"] = None


def investigate(client, alert, seed=3001):
    body = client.get("/api/diagnose", params={"seed": seed, "start": alert["start"],
                                               "end": alert["start"] + 120,
                                               "channel": alert["channel"], "provider": "scripted"}).text
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    return events[-1]


def test_investigation_produces_report_and_audit_record(client):
    run = client.get("/api/run", params={"seed": 3001}).json()
    lamp = next(a for a in run["detectors"]["residual_z"]["alerts"] if a["channel"] == "lamp_power_pct")
    done = investigate(client, lamp)
    assert done["type"] == "done" and done["status"] == "ok"
    assert done["report"]["severity"]["level"] == "plan" and done["audit_id"]
    assert set(done["versions"]) == {"model", "prompt_version", "kb_version", "detector_version"}

    assert client.post("/api/feedback", json={"diagnosis_id": done["audit_id"],
                                              "verdict": "correct"}).status_code == 200
    mon = client.get("/api/monitoring").json()
    assert mon["chain"]["ok"] and mon["metrics"]["n"] >= 1
    assert mon["metrics"]["engineer_confirmed_accuracy"] == 1.0
    assert mon["recent"][0]["id"] == done["audit_id"]
    exported = [json.loads(x) for x in client.get("/api/audit/export").text.splitlines()]
    assert exported[-1]["label"] == "lamp_degradation"


def test_feedback_rejects_bad_input(client):
    assert client.post("/api/feedback", json={"diagnosis_id": "nope", "verdict": "correct"}).status_code == 404
    assert client.post("/api/feedback", json={"diagnosis_id": "x", "verdict": "meh"}).status_code == 422
    assert client.post("/api/feedback", json={"diagnosis_id": "x", "verdict": "incorrect",
                                              "actual_cause": "gremlins"}).status_code == 422
