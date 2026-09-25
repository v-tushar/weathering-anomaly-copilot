"""
Weathering Instrument Anomaly Detection — Demo
================================================
Author: Tushar

Context
-------
Atlas (AMETEK) builds accelerated weathering / durability testing instruments
(xenon-arc Weather-Ometers, UV testers, corrosion cabinets, solar simulators)
used across automotive, paints & coatings, plastics, pharma, and electronics.
Their WXView data-acquisition software already streams telemetry (irradiance,
chamber temperature, black-panel temperature, humidity/relative humidity,
spray cycles) off these instruments during multi-week/multi-month test runs.

This demo shows an applied-ML layer on top of that kind of telemetry:
detecting anomalies (sensor faults, lamp degradation, control-loop drift,
seal/door failures) in near real time, plus a lightweight drift check that
would flag a slow degradation trend before it becomes an out-of-spec test run.

This is a self-contained, synthetic proof-of-concept — no proprietary Atlas
data used. It mirrors the anomaly-detection / time-series work from my
Hexaware (predictive maintenance, IoT sensor data) and Liberty Mutual
(fraud detection with Isolation Forest, drift detection, MLOps monitoring)
experience, applied to this instrument-telemetry domain.

Pipeline
--------
1. Simulate ~30 days of 15-min telemetry for one weathering chamber:
   irradiance (W/m^2), chamber air temp (C), black panel temp (C), RH (%).
   Includes normal cyclic behavior (light/dark cycles, daily thermal cycling)
   plus injected anomalies:
     - lamp degradation drift (slow irradiance decay over the back third of the run)
     - a sensor spike/dropout (bad reading)
     - a control-loop fault (temperature overshoot + oscillation)
     - a sustained humidity excursion (seal failure)
2. Feature engineering: rolling stats + rate-of-change per sensor.
3. Two detectors, deliberately simple + explainable:
     a. Isolation Forest (multivariate, point anomalies)
     b. Rolling z-score / CUSUM-style drift check (slow trend anomalies
        IsolationForest tends to miss)
4. Evaluate against the injected ground-truth anomaly windows (precision/recall).
5. Plot irradiance + chamber temp with flagged anomalies overlaid.
6. Print a short "production considerations" note — monitoring, retraining,
   alerting — the MLOps layer a production version would need.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Simulate telemetry
# ---------------------------------------------------------------------------
FREQ_MIN = 15
DAYS = 30
n = DAYS * 24 * (60 // FREQ_MIN)
timestamps = pd.date_range("2026-08-01", periods=n, freq=f"{FREQ_MIN}min")
t = np.arange(n)
hours = (t * FREQ_MIN / 60.0) % 24

# --- Irradiance: xenon-arc cycles between light/dark phases (e.g. 102 min light / 18 min dark,
# simplified here to an idealized duty cycle) with small sensor noise.
cycle_len = int(120 / (FREQ_MIN / 60))  # ~120 min full cycle
light_frac = 0.85
cycle_pos = t % cycle_len
base_irradiance = np.where(cycle_pos < cycle_len * light_frac, 550.0, 0.0)
irradiance = base_irradiance + rng.normal(0, 6, n)

# --- Chamber air temp: oscillates around a setpoint with mild daily influence
chamber_temp = 38 + 1.5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.4, n)

# --- Black panel temp: tracks irradiance (radiant heating) + chamber temp baseline
black_panel_temp = chamber_temp + 25 * (base_irradiance / 550.0) + rng.normal(0, 0.5, n)

# --- Relative humidity: fairly stable with small noise, occasional test-cycle dips
rh = 50 + rng.normal(0, 2, n)

# ---------------------------------------------------------------------------
# Inject anomalies with known ground-truth windows
# ---------------------------------------------------------------------------
ground_truth = np.zeros(n, dtype=bool)

# (a) Lamp degradation drift: irradiance slowly decays ~15% over last third of run
drift_start = int(n * 0.65)
decay = np.linspace(0, 0.15, n - drift_start)
irradiance[drift_start:] *= (1 - decay)
ground_truth[drift_start:] = True

# (b) Sensor spike/dropout: irradiance sensor glitch, short burst
spike_start = int(n * 0.20)
spike_len = 6  # 1.5 hours
irradiance[spike_start:spike_start + spike_len] = rng.uniform(900, 1000, spike_len)
ground_truth[spike_start:spike_start + spike_len] = True

# (c) Control-loop fault: chamber temp overshoot + oscillation
fault_start = int(n * 0.40)
fault_len = 20  # 5 hours
osc = 6 * np.sin(np.linspace(0, 6 * np.pi, fault_len))
chamber_temp[fault_start:fault_start + fault_len] += 5 + osc
black_panel_temp[fault_start:fault_start + fault_len] += 5 + osc
ground_truth[fault_start:fault_start + fault_len] = True

# (d) Humidity excursion: seal failure, sustained RH rise
humid_start = int(n * 0.82)
humid_len = 40  # 10 hours
rh[humid_start:humid_start + humid_len] += np.linspace(0, 25, humid_len)
ground_truth[humid_start:humid_start + humid_len] = True

df = pd.DataFrame({
    "timestamp": timestamps,
    "irradiance": irradiance,
    "chamber_temp": chamber_temp,
    "black_panel_temp": black_panel_temp,
    "rh": rh,
    "is_anomaly_true": ground_truth,
})

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
WIN = 8  # 2-hour rolling window
for col in ["irradiance", "chamber_temp", "black_panel_temp", "rh"]:
    df[f"{col}_roll_mean"] = df[col].rolling(WIN, min_periods=1).mean()
    df[f"{col}_roll_std"] = df[col].rolling(WIN, min_periods=1).std().fillna(0)
    df[f"{col}_rate"] = df[col].diff().fillna(0)

# Encode the instrument's known light/dark duty cycle as a feature so the
# model learns normal on/off transitions instead of flagging every one —
# in a real deployment this would come from the instrument's program/recipe
# state rather than being inferred.
df["cycle_phase"] = (t % cycle_len) / cycle_len

feature_cols = [c for c in df.columns if c.endswith(("_roll_mean", "_roll_std", "_rate"))] + \
    ["irradiance", "chamber_temp", "black_panel_temp", "rh", "cycle_phase"]

X = df[feature_cols].values

# ---------------------------------------------------------------------------
# 3a. Isolation Forest — multivariate point-anomaly detector
# ---------------------------------------------------------------------------
iso = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42,
)
df["iso_pred"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
df["iso_score"] = -iso.score_samples(X)  # higher = more anomalous
df["iso_anomaly"] = df["iso_pred"] == -1

# ---------------------------------------------------------------------------
# 3b. Rolling z-score / CUSUM-style drift check
# (catches the slow lamp-degradation trend IsolationForest under-weights)
#
# Important: irradiance itself cycles between ~0 (dark phase) and ~550
# (light phase) every ~2h, by design. Comparing raw irradiance to a rolling
# baseline would flag every normal "lamp off" transition as a crash. So the
# drift check only evaluates the *light-phase* irradiance level — in a real
# deployment this phase signal would come from the instrument's own
# program/recipe state, not be inferred from the data.
# ---------------------------------------------------------------------------
is_light_phase = cycle_pos < cycle_len * light_frac
irr_light_only = df["irradiance"].where(is_light_phase)

LONG_WIN = 24 * 4  # ~24h of samples, light-phase points only within that span
baseline_mean = irr_light_only.rolling(LONG_WIN, min_periods=20).mean()
baseline_std = irr_light_only.rolling(LONG_WIN, min_periods=20).std()
z = (irr_light_only - baseline_mean) / baseline_std.replace(0, np.nan)
df["irradiance_zscore"] = z

cusum = np.zeros(n)
threshold_k = 0.5  # slack
for i in range(1, n):
    if not is_light_phase[i] or np.isnan(z.iloc[i]):
        cusum[i] = cusum[i - 1]  # hold steady during dark phase / warm-up
        continue
    dev = -z.iloc[i]  # negative z = below-baseline = downward decay
    cusum[i] = max(0, cusum[i - 1] + dev - threshold_k)
df["drift_cusum"] = cusum
df["drift_anomaly"] = df["drift_cusum"] > 8.0  # alert threshold, tuned by inspection

# ---------------------------------------------------------------------------
# Combine detectors
# ---------------------------------------------------------------------------
df["is_anomaly_pred"] = df["iso_anomaly"] | df["drift_anomaly"]

# ---------------------------------------------------------------------------
# 4. Evaluate against ground truth
# ---------------------------------------------------------------------------
tp = int(((df["is_anomaly_pred"]) & (df["is_anomaly_true"])).sum())
fp = int(((df["is_anomaly_pred"]) & (~df["is_anomaly_true"])).sum())
fn = int(((~df["is_anomaly_pred"]) & (df["is_anomaly_true"])).sum())
tn = int(((~df["is_anomaly_pred"]) & (~df["is_anomaly_true"])).sum())

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print("=" * 60)
print("EVALUATION (point-level, vs. injected ground truth)")
print("=" * 60)
print(f"  True positives:  {tp}")
print(f"  False positives: {fp}")
print(f"  False negatives: {fn}")
print(f"  Precision:       {precision:.2f}")
print(f"  Recall:          {recall:.2f}")
print(f"  F1:              {f1:.2f}")
print()
print("Detected windows (contiguous flagged stretches):")
flagged = df["is_anomaly_pred"].values
in_run = False
for i in range(n):
    if flagged[i] and not in_run:
        start_i = i
        in_run = True
    if in_run and (i == n - 1 or not flagged[i + 1]):
        print(f"  {df['timestamp'][start_i]}  ->  {df['timestamp'][i]}  "
              f"({i - start_i + 1} points)")
        in_run = False

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

axes[0].plot(df["timestamp"], df["irradiance"], color="#d97706", lw=0.8, label="Irradiance (W/m²)")
axes[0].scatter(df.loc[df["is_anomaly_pred"], "timestamp"],
                 df.loc[df["is_anomaly_pred"], "irradiance"],
                 color="crimson", s=10, zorder=5, label="Flagged anomaly")
axes[0].set_ylabel("Irradiance (W/m²)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Simulated Weathering Chamber Telemetry — Anomaly Detection Demo")

axes[1].plot(df["timestamp"], df["chamber_temp"], color="#2563eb", lw=0.8, label="Chamber temp (°C)")
axes[1].plot(df["timestamp"], df["black_panel_temp"], color="#7c3aed", lw=0.8, label="Black panel temp (°C)")
axes[1].scatter(df.loc[df["is_anomaly_pred"], "timestamp"],
                 df.loc[df["is_anomaly_pred"], "chamber_temp"],
                 color="crimson", s=10, zorder=5)
axes[1].set_ylabel("Temp (°C)")
axes[1].legend(loc="upper right", fontsize=8)

axes[2].plot(df["timestamp"], df["rh"], color="#059669", lw=0.8, label="Relative humidity (%)")
axes[2].scatter(df.loc[df["is_anomaly_pred"], "timestamp"],
                 df.loc[df["is_anomaly_pred"], "rh"],
                 color="crimson", s=10, zorder=5)
axes[2].set_ylabel("RH (%)")
axes[2].set_xlabel("Time")
axes[2].legend(loc="upper right", fontsize=8)

for ax in axes:
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("anomaly_detection_demo.png", dpi=150)
print("\nSaved plot -> anomaly_detection_demo.png")

df.to_csv("simulated_telemetry_with_predictions.csv", index=False)
print("Saved data -> simulated_telemetry_with_predictions.csv")
