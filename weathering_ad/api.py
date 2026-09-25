"""Minimal scoring service.

    MODEL_PATH=outputs/residual_z.joblib uvicorn weathering_ad.api:app

POST /score takes one test run's telemetry, from run start up to now, and
returns alert events. Sending history from run start keeps the service
stateless: the per-lamp baseline is taken from the first 24 h, and the CUSUM
drift state is rebuilt on each call. The production version would keep that
state per instrument (see README, "Next steps").
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .detectors import BASELINE_SAMPLES, to_alerts
from .simulator import CHANNELS

MODEL_VERSION = "residual_z-0.1"
app = FastAPI(title="Weathering chamber anomaly scoring", version=MODEL_VERSION)


class Reading(BaseModel):
    timestamp: datetime
    phase: Literal["light", "dark"]
    t_in_phase_min: int = Field(ge=0)
    irradiance: float
    lamp_power_pct: float = Field(ge=0, le=110)
    black_panel_c: float
    chamber_air_c: float
    rh_pct: float = Field(ge=0, le=100)


class ScoreRequest(BaseModel):
    instrument_id: str
    readings: list[Reading]

    @model_validator(mode="after")
    def check_history(self):
        if len(self.readings) < BASELINE_SAMPLES:
            raise ValueError(f"need >= {BASELINE_SAMPLES} readings (24 h from run start) "
                             f"to establish the lamp baseline; got {len(self.readings)}")
        ts = [r.timestamp for r in self.readings]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("readings must be strictly increasing in time")
        return self


class Alert(BaseModel):
    channel: str
    start: datetime
    last_seen: datetime
    ongoing: bool


class ScoreResponse(BaseModel):
    instrument_id: str
    model_version: str
    n_readings: int
    alerts: list[Alert]


@lru_cache
def get_model():
    return joblib.load(os.environ.get("MODEL_PATH", "outputs/residual_z.joblib"))


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["timestamp"] = pd.to_datetime(df.timestamp)
    try:
        flags = get_model().score(df[["timestamp", "phase", "t_in_phase_min", *CHANNELS]])
    except Exception as e:  # never leak a stack trace to a client
        raise HTTPException(status_code=500, detail=f"scoring failed: {type(e).__name__}") from e
    last = len(df) - 1
    alerts = [Alert(channel=a["channel"], start=df.timestamp[a["start"]],
                    last_seen=df.timestamp[a["end"]], ongoing=last - a["end"] <= 20)
              for a in to_alerts(flags)]
    return ScoreResponse(instrument_id=req.instrument_id, model_version=MODEL_VERSION,
                         n_readings=len(df), alerts=alerts)
