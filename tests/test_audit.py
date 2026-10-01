"""Audit trail: append-only, hash-chained, feedback linked to diagnoses."""

import json

import pytest

from copilot.audit import AuditLog


def record(**kw):
    base = {"provider": "scripted", "model": "scripted", "prompt_version": "p1", "kb_version": "k1",
            "detector_version": "d1", "alert_channel": "lamp_power_pct", "status": "ok",
            "likely_cause": "lamp_degradation", "confidence": "medium", "latency_s": 0.1,
            "input_tokens": 100, "output_tokens": 20, "injection_seen": False, "violations": [],
            "rule_baseline": "lamp_degradation", "ground_truth": "lamp_degradation"}
    return {**base, **kw}


@pytest.fixture
def log():
    return AuditLog(":memory:")


def test_metrics_and_feedback(log):
    a = log.log_diagnosis(record())
    log.log_diagnosis(record(status="needs_human_review", likely_cause="unknown", latency_s=0.3))
    log.log_feedback(a, "incorrect", "humidity_seal_leak", "door was open")
    log.log_feedback(a, "correct")                       # latest feedback wins
    m = log.metrics()
    assert m["n"] == 2 and m["passed_first_try"] == 0.5 and m["needs_human_review"] == 0.5
    assert m["feedback_n"] == 1 and m["engineer_confirmed_accuracy"] == 1.0
    assert m["simulation_accuracy"] == 0.5 and m["tokens_total"] == 240
    assert [r["label"] for r in log.export_labelled()] == ["lamp_degradation"]


def test_feedback_is_validated(log):
    a = log.log_diagnosis(record())
    with pytest.raises(KeyError):
        log.log_feedback("nope", "correct")
    with pytest.raises(ValueError):
        log.log_feedback(a, "maybe")
    with pytest.raises(ValueError):
        log.log_feedback(a, "incorrect", "aliens")


def test_chain_detects_edited_record(log):
    ids = [log.log_diagnosis(record()) for _ in range(3)]
    assert log.verify() == {"ok": True, "n": 3, "broken_at_seq": None}
    # Quietly "improve" a past diagnosis.
    row = log._db.execute("SELECT payload FROM events WHERE id = ?", (ids[1],)).fetchone()
    payload = {**json.loads(row[0]), "status": "ok_after_retry"}
    log._db.execute("UPDATE events SET payload = ? WHERE id = ?",
                    (json.dumps(payload, sort_keys=True), ids[1]))
    assert log.verify() == {"ok": False, "n": 3, "broken_at_seq": 2}


def test_chain_detects_deleted_record(log):
    ids = [log.log_diagnosis(record()) for _ in range(3)]
    log._db.execute("DELETE FROM events WHERE id = ?", (ids[1],))
    assert log.verify()["ok"] is False


def test_empty_log(log):
    assert log.metrics() == {"n": 0} and log.verify()["ok"]
