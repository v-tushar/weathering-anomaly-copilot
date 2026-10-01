"""Engineer-facing incident report for one investigated alert.

The report keeps two kinds of content apart, and the UI labels them:

measured   Computed here, by code, from the telemetry: whether test conditions
           stayed in tolerance, how long they were out, lamp headroom, the chart.
           Deterministic. No model involved, so it is trustworthy even when the
           AI's answer is not.
ai         The copilot's explanation and recommended actions, shown together with
           its guardrail status. If the guardrails did not pass, the UI says so.

Manual steps and safety notes are quoted verbatim from the cited sections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weathering_ad.detectors import Tolerance

from .guardrails import CAUSE_CHANNELS
from .kb import KnowledgeBase, safety_notes, section_steps
from .tools import SAMPLES_PER_HOUR, UNITS, _r

TOL = Tolerance()
LIGHT_SETPOINTS = {"irradiance": 0.55, "black_panel_c": 70.0, "chamber_air_c": 47.0, "rh_pct": 50.0}
LABELS = {"irradiance": "Irradiance", "lamp_power_pct": "Lamp drive",
          "black_panel_c": "Black panel temperature", "chamber_air_c": "Chamber air temperature",
          "rh_pct": "Relative humidity"}

TITLES = {
    "lamp_degradation": "Xenon lamp (or its filters) is degrading faster than normal",
    "irradiance_sensor_fault": "Irradiance sensor glitch: the lamp itself looks fine",
    "temperature_control_fault": "Chamber temperature control is unstable",
    "humidity_seal_leak": "Humidity is too high: likely a door seal or humidifier problem",
    "no_fault_or_false_alarm": "No fault found: likely a false alarm",
    "unknown": "Cause unclear: needs an engineer's review",
}
WHY = {
    "lamp_degradation": ("The controller is turning the lamp up to hide its aging, so irradiance "
                         "still looks perfect and a setpoint alarm stays silent. Once the lamp runs "
                         "out of headroom, specimens get less light than the test method requires."),
    "irradiance_sensor_fault": ("A reading this short cannot come from the lamp under closed-loop "
                                "control. It points to the sensor, its cable, or data acquisition."),
    "temperature_control_fault": ("Specimen temperature drives degradation rate. Unstable control "
                                  "can make results differ from the test method and from other runs."),
    "humidity_seal_leak": ("Moisture changes how specimens weather. Humidity well above the light-"
                           "phase setpoint can invalidate exposure, and a leak usually gets worse."),
    "no_fault_or_false_alarm": "No channel moved far from a healthy chamber around the alert time.",
    "unknown": "The evidence does not clearly match a known fault pattern.",
}
SEVERITY = {  # level -> (label shown to the engineer, colour key)
    "act": ("Act now", "bad"),
    "plan": ("Plan maintenance", "warn"),
    "investigate": ("Investigate today", "warn"),
    "monitor": ("Check, then monitor", "info"),
    "none": ("No action needed", "ok"),
    "review": ("Engineer review needed", "bad"),
}


def _steady(w: pd.DataFrame) -> pd.Series:
    return (w.phase == "light") & (w.t_in_phase_min >= TOL.settle_min)


def exposure_check(df: pd.DataFrame, alert: dict) -> list[dict]:
    """Did the controlled test conditions stay in tolerance during the alert?"""
    w = df.iloc[alert["start"]: alert["end"] + 1]
    w = w[_steady(w)]
    rows = []
    for c, sp in LIGHT_SETPOINTS.items():
        tol = getattr(TOL, c)
        dev = (w[c] - sp).abs()
        out = dev > tol
        rows.append({
            "channel": c, "label": LABELS[c], "unit": UNITS[c], "setpoint": sp, "tolerance": tol,
            "worst": _r(w[c].iloc[int(np.argmax(dev.to_numpy()))]) if len(w) else None,
            "hours_out": _r(out.sum() / SAMPLES_PER_HOUR),
            "ok": not bool(out.any()),
        })
    return rows


def lamp_headroom(df: pd.DataFrame, alert: dict, ctx: dict) -> dict | None:
    """Lamp drive left before the controller runs out, and days until then at the current rate.

    The rate is fitted on readings since the alert fired (at most the last 24 h), not
    on a longer window: a window that reaches back before the fault mixes healthy and
    faulty aging and badly overestimates the time left. On the simulated runs this
    estimate is within about a day of when drive actually reaches 100 %, from 6 h on.
    """
    e = alert["end"]
    lo = max(alert["start"], e - 24 * SAMPLES_PER_HOUR)
    w = df.iloc[lo: e + 1]
    w = w[_steady(w)]
    if len(w) < 2 * SAMPLES_PER_HOUR:
        return None
    days_x = (w.timestamp - w.timestamp.iloc[0]).dt.total_seconds() / 86400
    rate = _r(np.polyfit(days_x, w.lamp_power_pct, 1)[0])
    drive = _r(w.lamp_power_pct.tail(SAMPLES_PER_HOUR).median())
    normal = ctx.get("normal_lamp_aging_pct_per_day")
    days = _r((100 - drive) / rate) if rate > 0 and drive < 100 else None
    return {"drive_pct": drive, "headroom_pct": _r(max(0.0, 100 - drive)),
            "rate_pct_per_day": rate,
            "rate_window_h": _r((w.timestamp.iloc[-1] - w.timestamp.iloc[0]).total_seconds() / 3600),
            "normal_pct_per_day": normal,
            "times_normal": _r(rate / normal) if rate > 0 and normal else None,
            "days_to_max": days, "as_of": str(df.timestamp.iloc[e])}


def _facts(ctx: dict, channels: list[str], head: dict | None) -> list[str]:
    out = []
    for c in channels:
        i = ctx["channels"][c]
        if i["median_during_alert"] is None:
            continue
        out.append(f"{LABELS[c]}: {i['baseline_median_prior_24h']} before the alert, median "
                   f"{i['median_during_alert']} during it (range {i['min_during_alert']} to "
                   f"{i['max_during_alert']} {i['unit']}). Peak deviation from a healthy chamber: "
                   f"{i['peak_abs_z_during_alert']} standard deviations.")
    if head:
        out.append(f"Lamp drive has been rising {head['rate_pct_per_day']} %/day since the alert "
                   f"fired (fitted over {head['rate_window_h']} h). Normal aging for this lamp: "
                   f"{head['normal_pct_per_day']} %/day"
                   + (f", so it is aging {head['times_normal']}x faster than normal." if head["times_normal"] else "."))
    return out


def _series(df: pd.DataFrame, alert: dict, channels: list[str]) -> list[dict]:
    """Hourly light-phase medians from 48 h before the alert to the end of the evidence.
    Nothing after the evidence cutoff: an engineer paged in real time would not have it."""
    s, e = alert["start"], alert["end"]
    lo = max(0, s - 48 * SAMPLES_PER_HOUR)
    hi = min(len(df), e + 1)
    w = df.iloc[lo:hi]
    w = w[_steady(w)]
    t0 = df.timestamp.iloc[s]
    out = []
    for c in channels:
        hourly = w.set_index("timestamp")[c].resample("1h").median().dropna()
        out.append({"channel": c, "label": LABELS[c], "unit": UNITS[c],
                    "t_h": [_r((t - t0).total_seconds() / 3600) for t in hourly.index],
                    "v": [_r(v) for v in hourly.to_numpy()],
                    "setpoint": LIGHT_SETPOINTS.get(c),
                    "tolerance": getattr(TOL, c, None) if c in LIGHT_SETPOINTS else None,
                    "alert_end_h": _r((df.timestamp.iloc[e] - t0).total_seconds() / 3600)})
    return out


def _severity(cause: str, status: str, exposure_ok: bool, head: dict | None) -> tuple[str, str]:
    if status in ("needs_human_review", "error") or cause in (None, "unknown"):
        return "review", "The AI's answer did not pass verification. An engineer should review the evidence below."
    if not exposure_ok and cause != "no_fault_or_false_alarm":
        return "act", "Test conditions left tolerance. Record the deviation and notify the test owner (GEN-03)."
    if cause == "lamp_degradation":
        if head and head["days_to_max"] is not None:
            return "plan", (f"Test conditions are still in tolerance. At the current rate the lamp runs "
                            f"out of headroom in about {head['days_to_max']:.1f} days, then irradiance "
                            f"drops below setpoint.")
        return "plan", "Schedule lamp or filter service before drive reaches 100 %."
    if cause in ("temperature_control_fault", "humidity_seal_leak"):
        return "investigate", "Conditions are still in tolerance, but the fault usually gets worse."
    if cause == "irradiance_sensor_fault":
        return "monitor", "Check the sensor connection. A single glitch can be monitored."
    return "none", "Log the alert and keep monitoring."


def build_report(df: pd.DataFrame, alert: dict, ctx: dict, diagnosis: dict | None,
                 status: str, kb: KnowledgeBase) -> dict:
    cause = (diagnosis or {}).get("likely_cause")
    channels = CAUSE_CHANNELS.get(cause) or [alert["channel"]]
    if alert["channel"] not in channels:
        channels = [alert["channel"], *channels]
    head = lamp_headroom(df, alert, ctx) if "lamp_power_pct" in channels else None
    exposure = exposure_check(df, alert)
    if cause == "irradiance_sensor_fault":
        # A faulty sensor's own reading is not evidence the specimens got the wrong light.
        for r in exposure:
            if r["channel"] == "irradiance" and not r["ok"]:
                r["sensor_suspect"] = True
    exposure_ok = all(r["ok"] or r.get("sensor_suspect") for r in exposure)
    level, reason = _severity(cause, status, exposure_ok, head)
    normal = [LABELS[c] for c, i in ctx["channels"].items()
              if c not in channels and (i["peak_abs_z_during_alert"] or 0) < 5]
    verified = status in ("ok", "ok_after_retry")

    manual = []
    for cid in (diagnosis or {}).get("citations", []):
        if cid in kb.by_id:
            ch = kb.by_id[cid]
            manual.append({"id": cid, "title": ch.title, "text": ch.text,
                           "steps": section_steps(ch.text), "safety": safety_notes(ch.text)})

    return {
        "title": TITLES.get(cause, TITLES["unknown"]) if verified else TITLES["unknown"],
        "why_it_matters": WHY.get(cause, WHY["unknown"]) if verified else WHY["unknown"],
        "severity": {"level": level, "label": SEVERITY[level][0], "tone": SEVERITY[level][1],
                     "reason": reason},
        "alert": {"channel": alert["channel"], "label": LABELS.get(alert["channel"], alert["channel"]),
                  "started": str(df.timestamp.iloc[alert["start"]]),
                  "as_of": str(df.timestamp.iloc[alert["end"]]),
                  "hours_since_alert": ctx.get("alert_duration_hours")},
        "measured": {
            "facts": _facts(ctx, channels, head),
            "normal_channels": normal,
            "exposure": exposure,
            "exposure_ok": exposure_ok,
            "lamp_headroom": head,
            "series": _series(df, alert, channels),
        },
        "ai": None if not diagnosis else {
            "verified": verified, "status": status,
            "cause": cause, "confidence": diagnosis.get("confidence"),
            "summary": diagnosis.get("summary"),
            "actions": diagnosis.get("recommended_actions", []),
            "test_validity": diagnosis.get("test_validity_impact"),
            "citations": diagnosis.get("citations", []),
        },
        "manual": manual,
    }
