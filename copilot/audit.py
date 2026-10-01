"""Audit trail and monitoring for copilot diagnoses (SQLite, standard library only).

Every diagnosis is stored with what produced it: provider and model, prompt
version, knowledge-base version, detector version, every tool call, and the
guardrail result. Engineer feedback is stored as its own event that points to
the diagnosis it rates, so field use turns into labelled evaluation data.

Records are append-only and hash-chained: each event's hash covers the previous
hash and its own content. Editing or deleting a past record breaks `verify()`.
That is the property a quality-managed environment asks of an audit trail; a
production version would put the same schema in a managed database with
write-once storage and access control.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import statistics
import threading
import uuid
from datetime import datetime, timezone

from .tools import CAUSES

log = logging.getLogger("copilot.audit")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    id        TEXT UNIQUE NOT NULL,
    ts        TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('diagnosis', 'feedback')),
    ref       TEXT,
    payload   TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL
);
"""
GENESIS = "0" * 64
VERDICTS = ("correct", "incorrect")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(prev: str, id_: str, ts: str, kind: str, ref: str | None, payload: str) -> str:
    return hashlib.sha256("|".join([prev, id_, ts, kind, ref or "", payload]).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str = "outputs/audit.sqlite"):
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)

    # ---- writes -----------------------------------------------------------
    def _append(self, kind: str, payload: dict, ref: str | None = None) -> str:
        id_, ts = uuid.uuid4().hex[:12], _now()
        body = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            row = self._db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            self._db.execute(
                "INSERT INTO events (id, ts, kind, ref, payload, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (id_, ts, kind, ref, body, prev, _hash(prev, id_, ts, kind, ref, body)))
            self._db.commit()
        return id_

    def log_diagnosis(self, record: dict) -> str:
        id_ = self._append("diagnosis", record)
        # One structured line per diagnosis: what a log platform (Azure Monitor,
        # CloudWatch, Datadog) would ingest for dashboards and alerting.
        log.info(json.dumps({"event": "copilot_diagnosis", "id": id_,
                             **{k: record.get(k) for k in (
                                 "provider", "model", "prompt_version", "kb_version", "status",
                                 "likely_cause", "latency_s", "input_tokens", "output_tokens")}}))
        return id_

    def log_feedback(self, diagnosis_id: str, verdict: str, actual_cause: str | None = None,
                     note: str = "") -> str:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if actual_cause is not None and actual_cause not in CAUSES:
            raise ValueError(f"actual_cause must be one of {CAUSES}")
        with self._lock:
            found = self._db.execute("SELECT 1 FROM events WHERE id = ? AND kind = 'diagnosis'",
                                     (diagnosis_id,)).fetchone()
        if not found:
            raise KeyError(diagnosis_id)
        return self._append("feedback", {"verdict": verdict, "actual_cause": actual_cause,
                                         "note": note[:500]}, ref=diagnosis_id)

    # ---- reads ------------------------------------------------------------
    def _rows(self, kind: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, ts, ref, payload FROM events WHERE kind = ? "
                                    "ORDER BY seq", (kind,)).fetchall()
        return [{"id": i, "ts": t, "ref": r, **json.loads(p)} for i, t, r, p in rows]

    def diagnoses(self) -> list[dict]:
        """All diagnoses, oldest first, each with its latest engineer feedback (if any)."""
        latest = {f["ref"]: f for f in self._rows("feedback")}
        return [{**d, "feedback": latest.get(d["id"])} for d in self._rows("diagnosis")]

    def recent(self, limit: int = 15) -> list[dict]:
        keep = ("id", "ts", "alert_channel", "provider", "model", "prompt_version", "kb_version",
                "detector_version", "status", "likely_cause", "confidence", "latency_s",
                "input_tokens", "output_tokens", "injection_seen", "violations", "feedback")
        return [{k: d.get(k) for k in keep} for d in reversed(self.diagnoses()[-limit:])]

    def metrics(self) -> dict:
        ds = self.diagnoses()
        n = len(ds)
        if not n:
            return {"n": 0}

        def rate(pred) -> float:
            return round(sum(1 for d in ds if pred(d)) / n, 3)

        lat = sorted(d["latency_s"] for d in ds)
        rated = [d for d in ds if d["feedback"]]
        truth = [d for d in ds if d.get("ground_truth")]
        by_provider: dict = {}
        for d in ds:
            p = by_provider.setdefault(d["provider"], {"n": 0, "passed": 0, "review": 0})
            p["n"] += 1
            p["passed"] += d["status"] in ("ok", "ok_after_retry")
            p["review"] += d["status"] == "needs_human_review"
        return {
            "n": n,
            "passed_first_try": rate(lambda d: d["status"] == "ok"),
            "passed_after_retry": rate(lambda d: d["status"] == "ok_after_retry"),
            "needs_human_review": rate(lambda d: d["status"] == "needs_human_review"),
            "errors": rate(lambda d: d["status"] == "error"),
            "latency_median_s": round(statistics.median(lat), 2),
            "latency_p90_s": round(lat[min(n - 1, int(0.9 * n))], 2),
            "tokens_total": sum(d["input_tokens"] + d["output_tokens"] for d in ds),
            "tokens_per_diagnosis": round(sum(d["input_tokens"] + d["output_tokens"] for d in ds) / n),
            "injections_detected": sum(1 for d in ds if d.get("injection_seen")),
            "agreement_with_rule_baseline": rate(lambda d: d["likely_cause"] == d.get("rule_baseline")),
            "feedback_n": len(rated),
            "engineer_confirmed_accuracy": (round(sum(d["feedback"]["verdict"] == "correct"
                                                      for d in rated) / len(rated), 3)
                                            if rated else None),
            "simulation_accuracy": (round(sum(d["likely_cause"] == d["ground_truth"]
                                              for d in truth) / len(truth), 3) if truth else None),
            "by_provider": by_provider,
            "versions": sorted({(d["prompt_version"], d["kb_version"], d["model"]) for d in ds}),
        }

    def verify(self) -> dict:
        """Recompute the hash chain. Any edited, inserted or deleted record breaks it."""
        with self._lock:
            rows = self._db.execute("SELECT seq, id, ts, kind, ref, payload, prev_hash, hash "
                                    "FROM events ORDER BY seq").fetchall()
        prev = GENESIS
        for seq, id_, ts, kind, ref, payload, prev_hash, h in rows:
            if prev_hash != prev or _hash(prev, id_, ts, kind, ref, payload) != h:
                return {"ok": False, "n": len(rows), "broken_at_seq": seq}
            prev = h
        return {"ok": True, "n": len(rows), "broken_at_seq": None}

    def export_labelled(self) -> list[dict]:
        """Diagnoses an engineer has rated: ready to add to the evaluation set."""
        out = []
        for d in self.diagnoses():
            fb = d["feedback"]
            if not fb:
                continue
            label = d["likely_cause"] if fb["verdict"] == "correct" else fb["actual_cause"]
            out.append({"diagnosis_id": d["id"], "run_seed": d.get("run_seed"),
                        "alert": d.get("alert"), "predicted": d["likely_cause"],
                        "label": label, "verdict": fb["verdict"], "note": fb["note"],
                        "model": d["model"], "prompt_version": d["prompt_version"]})
        return out
