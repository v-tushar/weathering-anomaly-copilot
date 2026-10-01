"""Engineer report: the measured half must be right regardless of what the AI says."""

import pytest

from copilot.agent import diagnose
from copilot.baseline import scripted_policy
from copilot.kb import KnowledgeBase, safety_notes, section_steps
from copilot.llm import ScriptedLLM
from copilot.report import build_report, exposure_check, lamp_headroom
from copilot.tools import ToolSession
from weathering_ad.detectors import to_alerts
from weathering_ad.simulator import FAULT_TYPES, simulate_run


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.fixture(scope="module")
def run3001(fitted):
    df, truth = simulate_run(3001, FAULT_TYPES)
    return df, truth, to_alerts(fitted["residual_z"].score(df))


def first(alerts, channel):
    return next(a for a in alerts if a["channel"] == channel)


def paged(alert, hours=12):
    """The evidence an engineer paged in real time would have."""
    return {**alert, "end": min(alert["end"], alert["start"] + hours * 10)}


def report_for(fitted, df, alert, kb):
    ctx = ToolSession(df, fitted["residual_z"], alert, kb).get_alert_context()
    r = diagnose(ScriptedLLM(scripted_policy), df, fitted["residual_z"], alert, kb=kb)
    return build_report(df, alert, ctx, r.diagnosis, r.status, kb), r


def test_manual_steps_and_safety_notes_are_quoted_verbatim(kb):
    steps = section_steps(kb.by_id["LAMP-02"].text)
    assert len(steps) == 5 and steps[2].startswith("Inspect and clean the optical filters")
    assert any("qualified technician" in s for s in safety_notes(kb.by_id["LAMP-02"].text))
    assert section_steps(kb.by_id["LAMP-01"].text) == []  # explanation only, no procedure


def test_lamp_alert_is_early_warning_not_act_now(fitted, run3001, kb):
    df, truth, alerts = run3001
    rep, _ = report_for(fitted, df, paged(first(alerts, "lamp_power_pct")), kb)
    assert rep["severity"]["level"] == "plan"          # caught before the test is affected
    assert rep["measured"]["exposure_ok"]
    assert rep["ai"]["verified"] and rep["ai"]["cause"] == "lamp_degradation"
    assert any("optical filters" in a for a in rep["ai"]["actions"])
    # The steps' source section is cited, so its safety warning reaches the engineer.
    assert "LAMP-02" in rep["ai"]["citations"]
    assert any("high voltage" in s for m in rep["manual"] for s in m["safety"])


def test_lamp_headroom_forecast_matches_simulation(fitted, run3001, kb):
    """Days until drive hits 100 %, forecast 12 h after the alert, vs. what really happened."""
    df, _, alerts = run3001
    a = paged(first(alerts, "lamp_power_pct"))
    ctx = ToolSession(df, fitted["residual_z"], a, kb).get_alert_context()
    head = lamp_headroom(df, a, ctx)
    light = (df.phase == "light") & (df.t_in_phase_min >= 30)
    hit = df.index[(df.lamp_power_pct >= 99.9) & light][0]
    actual_days = (df.timestamp[hit] - df.timestamp[a["end"]]).total_seconds() / 86400
    assert abs(head["days_to_max"] - actual_days) < 1.5


def test_lamp_at_max_is_act_now(fitted, run3001, kb):
    """In hindsight (whole alert), the lamp has run out and irradiance is below setpoint."""
    df, _, alerts = run3001
    rep, _ = report_for(fitted, df, first(alerts, "lamp_power_pct"), kb)
    assert rep["severity"]["level"] == "act"
    irr = next(r for r in rep["measured"]["exposure"] if r["channel"] == "irradiance")
    assert not irr["ok"] and irr["hours_out"] > 10


def test_sensor_glitch_is_not_an_exposure_deviation(fitted, run3001, kb):
    df, truth, alerts = run3001
    spike = next(f for f in truth if f.type == "sensor_spike")
    a = next(x for x in alerts if x["channel"] == "irradiance" and spike.start <= x["start"] <= spike.end + 10)
    rep, _ = report_for(fitted, df, a, kb)
    irr = next(r for r in rep["measured"]["exposure"] if r["channel"] == "irradiance")
    assert irr.get("sensor_suspect") and rep["measured"]["exposure_ok"]
    assert rep["severity"]["level"] == "monitor"


def test_unverified_ai_answer_is_flagged_and_measured_facts_remain(fitted, run3001, kb):
    df, _, alerts = run3001
    a = paged(first(alerts, "lamp_power_pct"))
    ctx = ToolSession(df, fitted["residual_z"], a, kb).get_alert_context()
    bogus = {"likely_cause": "humidity_seal_leak", "confidence": "high", "summary": "x",
             "evidence": [], "recommended_actions": [], "citations": [], "test_validity_impact": ""}
    rep = build_report(df, a, ctx, bogus, "needs_human_review", kb)
    assert rep["severity"]["level"] == "review" and not rep["ai"]["verified"]
    assert rep["title"].startswith("Cause unclear")      # no confident headline from an unverified answer
    assert rep["measured"]["facts"] and rep["measured"]["series"]


def test_report_chart_stops_at_the_evidence_cutoff(fitted, run3001, kb):
    df, _, alerts = run3001
    a = paged(first(alerts, "lamp_power_pct"), hours=6)
    rep, _ = report_for(fitted, df, a, kb)
    for s in rep["measured"]["series"]:
        assert max(s["t_h"]) <= s["alert_end_h"] + 1e-9   # no hindsight data in the chart


def test_exposure_check_flags_seal_leak_humidity(fitted, run3001):
    df, truth, alerts = run3001
    rows = {r["channel"]: r for r in exposure_check(df, first(alerts, "rh_pct"))}
    assert not rows["rh_pct"]["ok"] and rows["chamber_air_c"]["ok"]
