"""Rule-based diagnoser, and the scripted offline policy built on it.

The rules are the bar the LLM has to clear. If a lookup table gets the cause
right just as often, the LLM's value has to come from elsewhere: grounded
explanations, next steps pulled from the manual, and handling cases the rules
don't cover.
"""

from __future__ import annotations

import json
import re

from .kb import section_steps
from .llm import ToolCall, Turn
from .tools import CAUSES

CAUSE_QUERY = {
    "lamp_degradation": "lamp drive rising faster than normal aging closed-loop irradiance",
    "irradiance_sensor_fault": "irradiance short spike dropout sensor fault",
    "temperature_control_fault": "chamber air temperature oscillation overshoot control",
    "humidity_seal_leak": "relative humidity above setpoint light phase door seal",
    "no_fault_or_false_alarm": "alert triage false alarm no deviation",
    "unknown": "alert triage principles",
}


def rule_diagnosis(ctx: dict) -> str:
    z = {c: (v.get("peak_abs_z_during_alert") or 0.0) for c, v in ctx["channels"].items()}
    trend = ctx.get("lamp_drive_trend_pct_per_day_last_72h") or 0.0
    normal = ctx.get("normal_lamp_aging_pct_per_day") or 0.05
    lamp_max = ctx["channels"]["lamp_power_pct"].get("max_during_alert") or 0.0
    if max(z.values()) < 5:
        return "no_fault_or_false_alarm"
    if trend > 10 * normal or z["lamp_power_pct"] >= 5:
        return "lamp_degradation"
    if z["irradiance"] >= 5:
        return "lamp_degradation" if lamp_max >= 98 else "irradiance_sensor_fault"
    if max(z["chamber_air_c"], z["black_panel_c"]) >= 5:
        return "temperature_control_fault"
    if z["rh_pct"] >= 5:
        return "humidity_seal_leak"
    return "unknown"


def _last_tool_json(transcript: list[dict], name: str) -> dict | None:
    for m in reversed(transcript):
        if m["role"] == "tool" and m["name"] == name:
            body = re.search(r"\n(.*)\n</tool_result>", m["content"], re.S)
            return json.loads(body.group(1)) if body else None
    return None


def scripted_policy(transcript: list[dict]) -> Turn:
    """Offline stand-in: alert context -> manual search -> rule-based diagnosis."""
    n_tool = sum(1 for m in transcript if m["role"] == "tool")
    if n_tool == 0:
        return Turn(None, [ToolCall("c1", "get_alert_context", {})])
    ctx = _last_tool_json(transcript, "get_alert_context")
    cause = rule_diagnosis(ctx)
    if n_tool == 1:
        return Turn(None, [ToolCall("c2", "search_manual", {"query": CAUSE_QUERY[cause]})])
    hits = _last_tool_json(transcript, "search_manual")["results"]
    ch = ctx["alert_channel"]
    info = ctx["channels"][ch]
    # Actions: the procedure steps of the first retrieved section that has any, verbatim.
    # That section is always cited, so its safety notes reach the engineer with the steps.
    src = next((h for h in hits if section_steps(h["text"])), None)
    steps = section_steps(src["text"]) if src else []
    cited = list(dict.fromkeys([h["id"] for h in hits[:1]] + ([src["id"]] if src else [])))
    diag = {
        "likely_cause": cause,
        "confidence": "medium",
        "summary": (f"The {ch} alert matches the pattern for {cause.replace('_', ' ')} "
                    f"(rule-based classification, not a language model). Peak deviation on "
                    f"{ch} was {info['peak_abs_z_during_alert']} standard deviations."),
        "evidence": [f"{ch} peak |z| during alert was {info['peak_abs_z_during_alert']}",
                     f"{ch} median during alert {info['median_during_alert']} vs baseline "
                     f"{info['baseline_median_prior_24h']}"],
        "recommended_actions": steps or ([f"Follow manual section {hits[0]['id']}."] if hits else []),
        "citations": cited,
        "test_validity_impact": "Check whether exposure conditions left tolerance (GEN-03).",
    }
    return Turn(None, [ToolCall(f"c{n_tool + 1}", "submit_diagnosis", diag)])


assert set(CAUSE_QUERY) == set(CAUSES)
