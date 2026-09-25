"""Tools the copilot can call, and the telemetry summaries behind them.

The model never sees the raw 7,200-row time series. Tools return compact,
rounded summaries: fewer tokens, less room for misreading, and every number
the model may quote is logged, which the grounding guardrail checks against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from weathering_ad.simulator import CHANNELS

from .kb import KnowledgeBase

SAMPLES_PER_HOUR = 10
UNITS = {"irradiance": "W/m² @340nm", "lamp_power_pct": "% drive", "black_panel_c": "°C",
         "chamber_air_c": "°C", "rh_pct": "% RH"}

CAUSES = ["lamp_degradation", "irradiance_sensor_fault", "temperature_control_fault",
          "humidity_seal_leak", "no_fault_or_false_alarm", "unknown"]

TOOL_SPECS = [
    {
        "name": "get_alert_context",
        "description": ("Summary of the alert under investigation: channel, timing, and for every "
                        "sensor channel the healthy baseline vs. values during the alert, the peak "
                        "deviation in standard deviations (z), and the lamp drive trend. Call this first."),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_channel_trend",
        "description": ("Hourly medians of steady-state light-phase readings for one channel, from "
                        "`hours_before` the alert start to `hours_after` it. Use to see how a signal evolved."),
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "enum": CHANNELS},
                "hours_before": {"type": "integer", "minimum": 1, "maximum": 72},
                "hours_after": {"type": "integer", "minimum": 0, "maximum": 72},
            },
            "required": ["channel", "hours_before", "hours_after"],
        },
    },
    {
        "name": "search_manual",
        "description": ("Search the chamber troubleshooting guide and field notes. Returns the most "
                        "relevant sections with their ids. Cite ids in the diagnosis."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "submit_diagnosis",
        "description": "Submit the final diagnosis. Must be the last call.",
        "parameters": {
            "type": "object",
            "properties": {
                "likely_cause": {"type": "string", "enum": CAUSES},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "summary": {"type": "string", "description": "2-3 sentences for an operator."},
                "evidence": {"type": "array", "items": {"type": "string"},
                             "description": "Observations from tool results. Quote numbers exactly."},
                "recommended_actions": {"type": "array", "items": {"type": "string"}},
                "citations": {"type": "array", "items": {"type": "string"},
                              "description": "Ids of manual sections that support the actions."},
                "test_validity_impact": {"type": "string",
                                         "description": "Whether the running test's exposure may be compromised."},
            },
            "required": ["likely_cause", "confidence", "summary", "evidence",
                         "recommended_actions", "citations", "test_validity_impact"],
        },
    },
]


def _r(x: float) -> float:
    return float(np.round(x, 3 if abs(x) < 1 else 2))


@dataclass
class ToolSession:
    """Everything the tools need for one alert, plus a log of what was returned."""
    df: pd.DataFrame
    detector: object          # fitted ResidualZ (for phase profile and aging rate)
    alert: dict               # {"start", "end", "channel"} sample indices
    kb: KnowledgeBase
    operator_notes: list[str] = field(default_factory=list)
    redact: bool = True       # input-side injection filter (see guardrails.redact_injections)
    injection_seen: bool = False
    returned_numbers: list[float] = field(default_factory=list)
    retrieved_ids: set[str] = field(default_factory=set)
    calls: list[dict] = field(default_factory=list)

    # ---- tool implementations -----------------------------------------
    def _steady(self) -> pd.Series:
        return (self.df.phase == "light") & (self.df.t_in_phase_min >= 30)

    def get_alert_context(self) -> dict:
        df, a = self.df, self.alert
        s, e = a["start"], a["end"]
        steady = self._steady()
        z = self.detector.profile_.zscores(df)
        base_win = slice(max(0, s - 24 * SAMPLES_PER_HOUR), s)
        alert_win = slice(s, e + 1)
        channels = {}
        for c in CHANNELS:
            b = df[c][base_win][steady[base_win]]
            d = df[c][alert_win][steady[alert_win]]
            zc = z[c].iloc[alert_win]
            channels[c] = {
                "unit": UNITS[c],
                "baseline_median_prior_24h": _r(b.median()) if len(b) else None,
                "median_during_alert": _r(d.median()) if len(d) else None,
                "min_during_alert": _r(d.min()) if len(d) else None,
                "max_during_alert": _r(d.max()) if len(d) else None,
                "peak_abs_z_during_alert": _r(zc.abs().max()) if len(zc) else 0.0,
            }
        lamp = df.lamp_power_pct[steady]
        days = (df.timestamp[steady] - df.timestamp.iloc[0]).dt.total_seconds() / 86400
        recent = days.index[(days.index <= e) & (days.index >= e - 72 * SAMPLES_PER_HOUR)]
        slope = np.polyfit(days[recent], lamp[recent], 1)[0] if len(recent) > 10 else np.nan
        out = {
            "alert_channel": a["channel"],
            "alert_start": str(df.timestamp[s]),
            "alert_duration_hours": _r((e - s + 1) / SAMPLES_PER_HOUR),
            "note_on_z": "z = deviation from a healthy chamber at the same point in the cycle, "
                         "in standard deviations; |z| above 5 is abnormal",
            "channels": channels,
            "lamp_drive_trend_pct_per_day_last_72h": _r(slope) if np.isfinite(slope) else None,
            "normal_lamp_aging_pct_per_day": _r(self.detector.profile_.aging_per_day_["lamp_power_pct"]),
            "irradiance_setpoint_light": 0.55,
            "rh_setpoint_light_pct": 50.0,
            "chamber_air_setpoint_light_c": 47.0,
        }
        if self.operator_notes:
            out["operator_notes"] = [self._clean(n) for n in self.operator_notes]
        return out

    def _clean(self, text: str) -> str:
        from .guardrails import redact_injections
        cleaned, found = redact_injections(text)
        self.injection_seen |= found
        return cleaned if self.redact else text

    def get_channel_trend(self, channel: str, hours_before: int, hours_after: int) -> dict:
        if channel not in CHANNELS:
            return {"error": f"unknown channel {channel}; valid: {CHANNELS}"}
        hours_before = int(np.clip(hours_before, 1, 72))
        hours_after = int(np.clip(hours_after, 0, 72))
        s = self.alert["start"]
        lo = max(0, s - hours_before * SAMPLES_PER_HOUR)
        hi = min(len(self.df), s + hours_after * SAMPLES_PER_HOUR)
        w = self.df.iloc[lo:hi]
        w = w[(w.phase == "light") & (w.t_in_phase_min >= 30)]
        hourly = w.set_index("timestamp")[channel].resample("1h").median().dropna()
        return {"channel": channel, "unit": UNITS[channel],
                "alert_start": str(self.df.timestamp[s]),
                "hourly_median": [{"t": str(t), "v": _r(v)} for t, v in hourly.items()]}

    def search_manual(self, query: str) -> dict:
        hits = self.kb.search(query, k=3)
        self.retrieved_ids.update(c.id for c, _ in hits)
        return {"results": [{"id": c.id, "title": c.title, "source": c.source,
                             "text": self._clean(c.text)}
                            for c, _ in hits]}

    # ---- dispatch -------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        fn = {"get_alert_context": lambda: self.get_alert_context(),
              "get_channel_trend": lambda: self.get_channel_trend(**args),
              "search_manual": lambda: self.search_manual(**args)}.get(name)
        if fn is None:
            result = {"error": f"unknown tool {name}"}
        else:
            try:
                result = fn()
            except TypeError as e:  # bad arguments from the model
                result = {"error": f"bad arguments for {name}: {e}"}
        self.calls.append({"tool": name, "args": args})
        text = json.dumps(result, default=str)
        self.returned_numbers.extend(_value_numbers(result))
        # Retrieved text is untrusted data. Wrap it so the model can tell data from instructions.
        return f"<tool_result name=\"{name}\">\n{text}\n</tool_result>"


def _value_numbers(obj) -> list[float]:
    """Numbers the model may quote: numeric values, and numbers inside string values.

    Keys ("prior_24h"), timestamps and section ids are skipped, so they cannot
    whitelist small integers the model never actually saw as a measurement.
    """
    from .guardrails import _numbers
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _value_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _value_numbers(v)]
    return _numbers(str(obj))
