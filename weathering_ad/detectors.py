"""Detectors for weathering-chamber telemetry.

Everything here is fit on known-good calibration runs only, and every
feature is causal: the score at time t uses data up to t. That means the
same code can run on a live stream.

Three detectors, from simplest to most complex:

StaticThreshold  What an instrument controller already does: alarm when a
                 steady-state reading leaves setpoint +/- tolerance. This is
                 the baseline the ML has to beat.
ResidualZ        Phase-aware residuals (reading minus what a healthy chamber
                 shows at this point in the cycle), a max-|z| rule, and CUSUM
                 drift checks on lamp power and irradiance.
IForestCusum     Same residual features scored by an Isolation Forest, with a
                 threshold set from the calibration score distribution, plus
                 the same CUSUM drift checks.

All detectors output per-sample flags. `to_alerts` then debounces the flags
and merges them into alert events with a probable-cause channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .simulator import CHANNELS

SELF_BASELINE = ["lamp_power_pct"]  # varies lamp to lamp, so baseline per run
BASELINE_SAMPLES = 240              # first 24 h at 6-min sampling
WARMUP_SAMPLES = 40                 # first 2 cycles: chamber stabilizing, no alarms
ROLL = 5


def _suppress_warmup(flags: pd.DataFrame) -> pd.DataFrame:
    flags.iloc[:WARMUP_SAMPLES] = False
    return flags


# ---------------------------------------------------------------------------
# Phase profile + residual features
# ---------------------------------------------------------------------------
class PhaseProfile:
    """Expected value and spread for each (phase, time-in-phase) for each channel."""

    aging_per_day_: dict = {}

    def fit(self, runs: list[pd.DataFrame]) -> "PhaseProfile":
        # Learn the normal aging trend of self-baselined channels (e.g. lamp drive
        # rising slowly as a healthy lamp ages) so drift detection only reacts
        # to abnormal aging.
        self.aging_per_day_ = {}
        for c in SELF_BASELINE:
            slopes = []
            for r in runs:
                lt = r[r.phase == "light"]
                days = (lt.timestamp - r.timestamp.iloc[0]).dt.total_seconds() / 86400
                slopes.append(np.polyfit(days, lt[c], 1)[0])
            self.aging_per_day_[c] = float(np.median(slopes))
        data = pd.concat([self._rebase(r) for r in runs])
        g = data.groupby(["phase", "t_in_phase_min"])[CHANNELS]
        self.mean_ = g.mean()
        # Floor the std so near-constant channels (e.g. dark irradiance) do not explode.
        self.std_ = g.std().clip(lower=data[CHANNELS].std() * 0.02, axis=1).fillna(1.0)
        return self

    def _rebase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Express self-baselined channels relative to this run's first 24 h,
        minus the expected normal aging since then."""
        out = df.copy()
        days = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds() / 86400
        for c in SELF_BASELINE:
            head = df.iloc[:BASELINE_SAMPLES]
            ref = head.loc[head.phase == "light", c].median()
            expected = ref + self.aging_per_day_.get(c, 0.0) * (days - 0.5)
            out[c] = np.where(df.phase == "light", df[c] - expected, 0.0)
        return out

    def zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        r = self._rebase(df)
        key = pd.MultiIndex.from_arrays([r.phase, r.t_in_phase_min])
        mu = self.mean_.reindex(key).to_numpy()
        sd = self.std_.reindex(key).to_numpy()
        z = (r[CHANNELS].to_numpy() - mu) / sd
        return pd.DataFrame(np.nan_to_num(z), columns=CHANNELS, index=df.index)


def residual_features(z: pd.DataFrame) -> pd.DataFrame:
    """Causal features on residual z-scores: current value, trailing mean/std, first difference."""
    feats = {f"z_{c}": z[c] for c in CHANNELS}
    for c in CHANNELS:
        feats[f"zmean_{c}"] = z[c].rolling(ROLL, min_periods=1).mean()
        feats[f"zstd_{c}"] = z[c].rolling(ROLL, min_periods=2).std().fillna(0)
        feats[f"zdiff_{c}"] = z[c].diff().fillna(0)
    return pd.DataFrame(feats)


def cusum(x: np.ndarray, k: float, active: np.ndarray, h: float | None = None):
    """One-sided upper CUSUM. Holds its value when `active` is False (e.g. dark phase).

    Without `h`, returns the CUSUM path (used for calibration). With `h`, returns
    a boolean signal array and resets the sum to 0 after each signal, which is
    standard CUSUM practice. A real, ongoing drift re-signals quickly, and one
    noisy excursion no longer latches an alarm on for days.
    """
    s = np.zeros(len(x))
    sig = np.zeros(len(x), dtype=bool)
    for i in range(1, len(x)):
        s[i] = max(0.0, s[i - 1] + x[i] - k) if active[i] else s[i - 1]
        if h is not None and s[i] > h:
            sig[i] = True
            s[i] = 0.0
    return s if h is None else sig


# ---------------------------------------------------------------------------
# Drift component shared by ResidualZ and IForestCusum
# ---------------------------------------------------------------------------
class DriftCusum:
    """CUSUM on lamp power (drifting up) and irradiance (drifting down), light phase only.

    The alarm threshold is set from calibration runs: a margin above the largest
    CUSUM value any healthy run reached.
    """

    k = 0.5
    margin = 1.5

    @staticmethod
    def _inputs(z: pd.DataFrame) -> dict[str, np.ndarray]:
        # Smooth over one cycle first so single noisy samples do not drive the sum.
        return {"lamp_power_pct": z["lamp_power_pct"].rolling(17, min_periods=1).mean().to_numpy(),
                "irradiance": (-z["irradiance"]).rolling(17, min_periods=1).mean().to_numpy()}

    def fit(self, zs: list[pd.DataFrame], lights: list[np.ndarray]) -> "DriftCusum":
        peaks = {c: 0.0 for c in ("lamp_power_pct", "irradiance")}
        for z, light in zip(zs, lights):
            for c, x in self._inputs(z).items():
                peaks[c] = max(peaks[c], cusum(x, self.k, light).max())
        self.h_ = {c: max(5.0, p * self.margin) for c, p in peaks.items()}
        return self

    def flags(self, z: pd.DataFrame, light: np.ndarray) -> dict[str, np.ndarray]:
        return {c: cusum(x, self.k, light, h=self.h_[c]) for c, x in self._inputs(z).items()}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
@dataclass
class Tolerance:
    irradiance: float = 0.03        # W/m² absolute
    black_panel_c: float = 3.0
    chamber_air_c: float = 3.0
    rh_pct: float = 10.0
    lamp_power_alarm: float = 98.0  # "lamp near end of life" alarm
    settle_min: int = 30            # ignore the transient after a phase change


class StaticThreshold:
    name = "static_threshold"

    def __init__(self, tol: Tolerance | None = None, setpoints: dict | None = None):
        self.tol = tol or Tolerance()
        self.sp = setpoints or {"irradiance": (0.55, 0.0), "black_panel_c": (70.0, 38.0),
                                "chamber_air_c": (47.0, 38.0), "rh_pct": (50.0, 95.0)}

    def fit(self, runs):
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        light = (df.phase == "light").to_numpy()
        settled = (df.t_in_phase_min >= self.tol.settle_min).to_numpy() & light
        out = {}
        for c, (sl, sd) in self.sp.items():
            target = np.where(light, sl, sd)
            out[c] = settled & (np.abs(df[c].to_numpy() - target) > getattr(self.tol, c))
        out["lamp_power_pct"] = light & (df.lamp_power_pct.to_numpy() >= self.tol.lamp_power_alarm)
        return _suppress_warmup(debounce(pd.DataFrame(out, index=df.index)[CHANNELS]))


class ResidualZ:
    name = "residual_z"
    z_thresh = 5.0

    def fit(self, runs):
        self.profile_ = PhaseProfile().fit(runs)
        zs = [self.profile_.zscores(r) for r in runs]
        self.drift_ = DriftCusum().fit(zs, [(r.phase == "light").to_numpy() for r in runs])
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self.profile_.zscores(df)
        light = (df.phase == "light").to_numpy()
        out = debounce(pd.DataFrame(np.abs(z.to_numpy()) > self.z_thresh,
                                    columns=CHANNELS, index=df.index))
        for c, f in self.drift_.flags(z, light).items():
            out[c] |= f
        return _suppress_warmup(out)


class IForestCusum:
    name = "iforest_cusum"
    target_fpr = 0.001  # per-sample false-positive rate on calibration data

    def fit(self, runs):
        self.profile_ = PhaseProfile().fit(runs)
        zs = [self.profile_.zscores(r) for r in runs]
        X = pd.concat([residual_features(z) for z in zs])
        self.model_ = IsolationForest(n_estimators=300, random_state=0).fit(X)
        scores = -self.model_.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1 - self.target_fpr))
        self.drift_ = DriftCusum().fit(zs, [(r.phase == "light").to_numpy() for r in runs])
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self.profile_.zscores(df)
        s = -self.model_.score_samples(residual_features(z))
        point = s > self.threshold_
        # Attribute point anomalies to the channel with the largest |z| (explainability).
        cause = np.abs(z.to_numpy()).argmax(axis=1)
        out = np.zeros((len(df), len(CHANNELS)), dtype=bool)
        out[np.arange(len(df)), cause] = point
        out = debounce(pd.DataFrame(out, columns=CHANNELS, index=df.index))
        light = (df.phase == "light").to_numpy()
        for c, f in self.drift_.flags(z, light).items():
            out[c] |= f
        return _suppress_warmup(out)


DETECTORS = {d.name: d for d in (StaticThreshold, ResidualZ, IForestCusum)}


# ---------------------------------------------------------------------------
# Alerting: debounce point flags, merge into per-channel alert events
# ---------------------------------------------------------------------------
def debounce(flags: pd.DataFrame, min_hits: int = 2, window: int = 3) -> pd.DataFrame:
    """Keep a point flag only when its channel has `min_hits` flags in the trailing `window`.

    Applied to point-anomaly flags only. CUSUM drift signals already accumulate
    evidence over time, so they are not debounced a second time.
    """
    hits = flags.astype(int).rolling(window, min_periods=1).sum()
    return flags & (hits >= min_hits)


def to_alerts(flags: pd.DataFrame, merge_gap: int = 20) -> list[dict]:
    """Turn per-sample (already debounced) flags into alert events, per channel.

    An alert starts at the first flagged sample, which is when an operator would
    be paged. Flags on the same channel within `merge_gap` samples extend the
    same alert. Alerts are per channel, so an open alert on one sensor cannot
    hide a new fault on another.
    """
    alerts: list[dict] = []
    for c in flags.columns:
        fire = np.flatnonzero(flags[c].to_numpy())
        cur = None
        for i in fire:
            if cur and i - cur["end"] <= merge_gap:
                cur["end"] = int(i)
            else:
                cur = {"start": int(i), "end": int(i), "channel": c}
                alerts.append(cur)
    return sorted(alerts, key=lambda a: a["start"])
