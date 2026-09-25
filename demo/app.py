"""Live demo server: the whole pipeline in a browser, for a screen-shared walkthrough.

    python -m demo            # http://127.0.0.1:8000

One page, three acts:

1. Simulate a 30-day chamber run with chosen faults, and see where each detector
   alerts. The point to land: a static setpoint alarm misses lamp degradation for
   days, because closed-loop control hides it.
2. Click an alert and watch the copilot investigate it live (tool calls streamed
   as they happen), then read the guarded diagnosis.
3. Compare the LLM against the rule-based baseline on the same alert, and flip
   the prompt-injection note on to watch the guardrails hold.

Everything is synthetic. No Atlas or customer data is involved.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from copilot.agent import MAX_STEPS, diagnose
from copilot.baseline import rule_diagnosis
from copilot.kb import KnowledgeBase
from copilot.llm import make_llm
from copilot.tools import CAUSES, ToolSession
from weathering_ad.detectors import DETECTORS, to_alerts
from weathering_ad.evaluate import GRACE
from weathering_ad.simulator import CHANNELS, FAULT_TYPES, ChamberConfig, simulate_run

STATIC = Path(__file__).parent / "static"
CALIB_SEEDS = range(0, 10)          # same healthy calibration runs the shipped model uses
SAMPLE_MIN = 6
MAX_POINTS = 900                    # downsample for the browser; the detector sees every sample

app = FastAPI(title="Weathering chamber AI demo")
STATE: dict = {"detectors": {}, "kb": None, "ready": False}
_RUNS: dict = {}                    # (seed, faults, noise) -> (df, truth)
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Startup: fit the detectors once on healthy calibration runs (~10 s)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _warm() -> None:
    print("Fitting detectors on healthy calibration runs...", flush=True)
    calib = [simulate_run(s)[0] for s in CALIB_SEEDS]
    for name, D in DETECTORS.items():
        STATE["detectors"][name] = D().fit(calib)
        print(f"  fitted {name}", flush=True)
    STATE["kb"] = KnowledgeBase()
    STATE["ready"] = True
    print("Ready -> http://127.0.0.1:8000", flush=True)


def _noisy_cfg(factor: float) -> ChamberConfig:
    cfg = ChamberConfig()
    cfg.noise = {k: v * factor for k, v in cfg.noise.items()}
    return cfg


def get_run(seed: int, faults: tuple[str, ...], noise: float):
    """Simulated runs are deterministic, so cache them: clicking around is instant."""
    key = (seed, faults, round(noise, 3))
    with _LOCK:
        if key not in _RUNS:
            cfg = _noisy_cfg(noise) if noise != 1.0 else None
            _RUNS[key] = simulate_run(seed, list(faults), cfg)
            if len(_RUNS) > 24:                      # bounded; a demo visits few runs
                _RUNS.pop(next(iter(_RUNS)))
    return _RUNS[key]


def _days(i: int) -> float:
    return round(i * SAMPLE_MIN / 1440, 4)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    providers = [{"id": "scripted", "label": "Scripted (offline, not an LLM)",
                  "available": True, "env": None}]
    for pid, key, label in [("anthropic", "ANTHROPIC_API_KEY", "Anthropic"),
                            ("openai", "OPENAI_API_KEY", "OpenAI")]:
        providers.append({"id": pid, "label": label, "available": bool(os.environ.get(key)),
                          "env": key})
    results = Path("outputs/results.md")
    return {
        "ready": STATE["ready"],
        "faults": FAULT_TYPES,
        "channels": CHANNELS,
        "detectors": list(DETECTORS),
        "causes": CAUSES,
        "providers": providers,
        "model": os.environ.get("COPILOT_MODEL", "provider default"),
        "benchmark": results.read_text(encoding="utf-8") if results.exists() else None,
    }


@app.get("/api/run")
def run(seed: int = 3001, faults: str = ",".join(FAULT_TYPES), noise: float = 1.0):
    """Simulate one run and score it with all three detectors."""
    chosen = tuple(f for f in faults.split(",") if f in FAULT_TYPES)
    df, truth = get_run(seed, chosen, noise)

    # Plot steady-state light-phase readings only: at a 30-day scale the raw
    # light/dark cycling turns every panel into a solid band.
    steady = ((df.phase == "light") & (df.t_in_phase_min >= 30)).to_numpy()
    idx = np.flatnonzero(steady)
    stride = max(1, len(idx) // MAX_POINTS)
    idx = idx[::stride]
    series = {c: [round(float(v), 4) for v in df[c].to_numpy()[idx]] for c in CHANNELS}
    series["day"] = [_days(int(i)) for i in idx]
    # The spike fault is a handful of samples; striding can skip it. Mark it separately.
    spikes = [{"day": _days(i), "v": round(float(df.irradiance.iloc[i]), 4)}
              for f in truth if f.type == "sensor_spike" for i in range(f.start, f.end + 1)]

    detectors = {}
    for name, det in STATE["detectors"].items():
        alerts = to_alerts(det.score(df))
        out = []
        for a in alerts:
            hit = next((f for f in truth if f.start <= a["start"] <= f.end + GRACE), None)
            out.append({"start": a["start"], "end": a["end"], "channel": a["channel"],
                        "day_start": _days(a["start"]), "day_end": _days(a["end"]),
                        "matched_fault": hit.type if hit else None,
                        "ttd_h": round((a["start"] - hit.start) * SAMPLE_MIN / 60, 1)
                                 if hit else None})
        ttd = {}
        for f in truth:
            first = next((a for a in out if a["matched_fault"] == f.type
                          and f.start <= a["start"] <= f.end + GRACE), None)
            ttd[f.type] = first["ttd_h"] if first else None
        detectors[name] = {"alerts": out, "ttd_h": ttd,
                           "false_alarms": sum(1 for a in out if a["matched_fault"] is None)}

    return {
        "seed": seed, "noise": noise, "n_samples": int(len(df)),
        "days": round(len(df) * SAMPLE_MIN / 1440, 1),
        "series": series, "spikes": spikes,
        "truth": [{"type": f.type, "start": f.start, "end": f.end,
                   "day_start": _days(f.start), "day_end": _days(f.end)} for f in truth],
        "detectors": detectors,
    }


@app.get("/api/diagnose")
def diagnose_stream(seed: int, start: int, end: int, channel: str,
                    faults: str = ",".join(FAULT_TYPES), noise: float = 1.0,
                    provider: str = "scripted", inject: bool = False,
                    redact: bool = True, model: str | None = Query(default=None)):
    """Server-sent events: the agent's steps as they happen, then the final result.

    Streaming matters for the demo. A live model takes 10-30 s per alert, and
    watching the tool calls arrive is what shows this is an agent rather than
    a single prompt.
    """
    chosen = tuple(f for f in faults.split(",") if f in FAULT_TYPES)
    df, _ = get_run(seed, chosen, noise)
    detector = STATE["detectors"]["residual_z"]
    kb = STATE["kb"]
    alert = {"start": start, "end": end, "channel": channel}
    notes = [kb.by_id["FN-02"].text] if inject else None

    def generate():
        q: queue.Queue = queue.Queue()

        def work():
            try:
                llm = make_llm(provider, model)
                ctx = ToolSession(df, detector, alert, kb).get_alert_context()
                q.put({"type": "context", "context": ctx, "rule_baseline": rule_diagnosis(ctx),
                       "max_steps": MAX_STEPS})
                r = diagnose(llm, df, detector, alert, kb=kb, operator_notes=notes,
                             redact=redact, on_event=q.put)
                q.put({"type": "done", "status": r.status, "diagnosis": r.diagnosis,
                       "violations": r.violations, "steps": r.steps,
                       "tool_calls": r.tool_calls, "latency_s": round(r.latency_s, 2),
                       "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                       "injection_seen": r.injection_seen, "error": r.error})
            except BaseException as e:  # SystemExit from a missing key must reach the UI too
                q.put({"type": "done", "status": "error", "diagnosis": None, "violations": [],
                       "steps": 0, "tool_calls": [], "latency_s": 0.0, "input_tokens": 0,
                       "output_tokens": 0, "injection_seen": False,
                       "error": f"{type(e).__name__}: {e}"})
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
