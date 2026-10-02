"""Guardrails. Deterministic checks the diagnosis must pass before an operator sees it.

Input side (applied to tool results before the model sees them):
    redact_injections   Text in documents or notes that tries to instruct the AI is
                        replaced with a marker. Retrieved content is data, not instructions.

Output side (applied to the submitted diagnosis):
    schema              Cause and confidence come from fixed lists; required fields present.
    citations           Every cited id exists and was actually retrieved in this session.
    grounding           Every number in the summary and evidence matches a number a tool
                        returned (or the cited manual text). Catches invented sensor values.
    telemetry_check     The cause must agree with the sensors: a named fault needs a real
                        deviation on its channel, and "no fault" is rejected when a channel
                        is far out of normal.

A failed check is sent back to the model once to fix. If it fails again, the
diagnosis is marked needs_human_review rather than shown as trustworthy.
"""

from __future__ import annotations

import re

from .kb import KnowledgeBase
from .tools import CAUSES

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"(system|developer) (notice|message|prompt|note)[^.\n]*(ai|assistant|model)",
    r"you are now", r"disregard (the|your) (rules|instructions|system prompt)",
    r"do not mention this",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.I)
REDACTED = "[REDACTED: text resembling instructions to the AI assistant was removed]"

CAUSE_CHANNELS = {
    "lamp_degradation": ["lamp_power_pct", "irradiance"],
    "irradiance_sensor_fault": ["irradiance"],
    "temperature_control_fault": ["chamber_air_c", "black_panel_c"],
    "humidity_seal_leak": ["rh_pct"],
}
MIN_Z_FOR_CAUSE = 3.0
MAX_Z_FOR_NO_FAULT = 5.0


def redact_injections(text: str) -> tuple[str, bool]:
    """Redact sentences that look like instructions to the model. Returns (text, found)."""
    if not _INJ.search(text):
        return text, False  # leave clean documents byte-for-byte unchanged
    found = False
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _INJ.search(sent):
            found = True
            out.append(REDACTED)
        else:
            out.append(sent)
    return " ".join(out), found


def _numbers(text: str) -> list[float]:
    text = re.sub(r"\b[A-Z]+-\d+\b", " ", text)       # section ids like LAMP-02
    # Dates and clock times are not sensor values: "2026-08-17 22:00:00", "2026-08-17", "22:36".
    text = re.sub(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", " ", text)
    # A '-' right after a digit is a range ("0.545-0.555"), not a minus sign.
    return [float(x) for x in re.findall(r"(?:(?<![\d.])-)?\d+(?:\.\d+)?", text)]


def _grounded(x: float, allowed: list[float]) -> bool:
    return any(abs(x - a) <= max(0.011, 0.01 * abs(a)) for a in allowed)


def check_diagnosis(d: dict, *, kb: KnowledgeBase, retrieved_ids: set[str],
                    returned_numbers: list[float], context: dict) -> list[str]:
    """Return a list of violations (empty = passed)."""
    v: list[str] = []
    required = ["likely_cause", "confidence", "summary", "evidence",
                "recommended_actions", "citations", "test_validity_impact"]
    missing = [k for k in required if k not in d]
    if missing:
        return [f"schema: missing fields {missing}"]
    if d["likely_cause"] not in CAUSES:
        v.append(f"schema: likely_cause {d['likely_cause']!r} not in {CAUSES}")
    if d["confidence"] not in ("low", "medium", "high"):
        v.append("schema: confidence must be low | medium | high")
    for k in ("evidence", "recommended_actions", "citations"):
        if not isinstance(d[k], list) or not all(isinstance(x, str) for x in d[k]):
            v.append(f"schema: {k} must be a list of strings")
    if v:
        return v

    # citations
    if not d["citations"]:
        v.append("citations: cite at least one manual section id")
    for cid in d["citations"]:
        if cid not in kb.by_id:
            v.append(f"citations: {cid!r} does not exist in the manual")
        elif cid not in retrieved_ids:
            v.append(f"citations: {cid!r} was not retrieved in this session; search for it first")

    # numeric grounding
    allowed = list(returned_numbers)
    for cid in d["citations"]:
        if cid in kb.by_id:
            allowed += _numbers(kb.by_id[cid].text)
    claims = " ".join([d["summary"], *d["evidence"]])
    bad = sorted({x for x in _numbers(claims) if not _grounded(x, allowed)})
    if bad:
        v.append(f"grounding: numbers {bad} do not appear in any tool result or cited section; "
                 "quote values exactly as returned")

    # agreement with telemetry
    z = {c: (info.get("peak_abs_z_during_alert") or 0.0) for c, info in context["channels"].items()}
    cause = d["likely_cause"]
    if cause in CAUSE_CHANNELS:
        best = max(z[c] for c in CAUSE_CHANNELS[cause])
        if best < MIN_Z_FOR_CAUSE:
            v.append(f"telemetry_check: {cause} requires a deviation on {CAUSE_CHANNELS[cause]}, "
                     f"but peak |z| there is only {best}")
    if cause == "no_fault_or_false_alarm":
        worst = max(z, key=z.get)
        if z[worst] > MAX_Z_FOR_NO_FAULT:
            v.append(f"telemetry_check: no_fault_or_false_alarm contradicts {worst} at peak |z| "
                     f"{z[worst]}")
    return v
