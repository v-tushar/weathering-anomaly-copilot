"""Event-level evaluation.

Point-level precision/recall is misleading here. One long drift fault can
cover a third of the samples and swamp the score (that is what went wrong in
v1). Operators care about three things instead:

detection rate    Was each fault caught at all?
time to detect    How long after onset was someone paged?
false alarms/day  How often were they paged for nothing?

An alert counts as a detection if it starts inside [fault.start, fault.end + grace].
Any alert that matches no fault is a false alarm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import to_alerts
from .simulator import Fault

GRACE = 10  # samples (1 h) after a fault ends, for lagging symptoms


def match(alerts: list[dict], truth: list[Fault], sample_min: int = 6) -> dict:
    per_fault = []
    used = set()
    for f in truth:
        hit = [i for i, a in enumerate(alerts) if f.start <= a["start"] <= f.end + GRACE]
        if hit:
            used.update(hit)
            ttd_h = (alerts[hit[0]]["start"] - f.start) * sample_min / 60
            per_fault.append({"type": f.type, "detected": True, "ttd_h": ttd_h})
        else:
            per_fault.append({"type": f.type, "detected": False, "ttd_h": np.nan})
    false_alarms = [a for i, a in enumerate(alerts) if i not in used]
    return {"faults": per_fault, "false_alarms": false_alarms}


def evaluate(detector, runs: list[tuple[pd.DataFrame, list[Fault]]]) -> tuple[pd.DataFrame, float]:
    """Return a per-fault results table and the false-alarm rate per day across all runs."""
    rows, fa, days = [], 0, 0.0
    for df, truth in runs:
        alerts = to_alerts(detector.score(df))
        m = match(alerts, truth)
        for r in m["faults"]:
            rows.append({**r, "seed": int(df.run_seed.iloc[0])})
        fa += len(m["false_alarms"])
        days += len(df) * 6 / 1440
    return pd.DataFrame(rows), fa / days


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("type")
    return pd.DataFrame({
        "detection_rate": g.detected.mean(),
        "median_ttd_h": g.ttd_h.median(),
        "p90_ttd_h": g.ttd_h.quantile(0.9),
    })
