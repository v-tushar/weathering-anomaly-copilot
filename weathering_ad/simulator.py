"""Synthetic telemetry for a xenon-arc weathering chamber.

The exposure cycle is loosely modeled on automotive-style xenon test cycles
(for example SAE J2527-type programs): a 120-minute cycle of 102 minutes
light and 18 minutes dark, with different chamber setpoints in each phase.
The numbers are plausible but illustrative. They are not taken from any
Atlas instrument or dataset.

Channels
--------
irradiance       W/m² at 340 nm. Closed-loop controlled to setpoint during light.
lamp_power_pct   Lamp drive needed to hold irradiance. Rises as the lamp ages.
black_panel_c    Black-panel temperature.
chamber_air_c    Chamber air temperature.
rh_pct           Relative humidity.

The controller state (`phase`, `t_in_phase_min`) is emitted as columns,
because a real instrument knows which program step it is in. Detectors use
it; they do not have to infer it.

Faults (ground truth is returned with each run)
-----------------------------------------------
sensor_spike      Irradiance sensor reports garbage for a few samples.
control_fault     Chamber-temperature control loop oscillates for hours.
lamp_degradation  Lamp ages abnormally fast. Closed-loop control hides it in
                  irradiance (lamp power rises instead) until power hits
                  100 %, then irradiance falls. This is the early-warning case.
seal_leak         Humidity creeps above setpoint during light (door/seal issue).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ["irradiance", "lamp_power_pct", "black_panel_c", "chamber_air_c", "rh_pct"]
FAULT_TYPES = ["sensor_spike", "control_fault", "lamp_degradation", "seal_leak"]


@dataclass
class ChamberConfig:
    sample_min: int = 6
    days: int = 30
    light_min: int = 102
    dark_min: int = 18
    # setpoints: (light, dark)
    irradiance_sp: tuple = (0.55, 0.0)
    black_panel_sp: tuple = (70.0, 38.0)
    chamber_air_sp: tuple = (47.0, 38.0)
    rh_sp: tuple = (50.0, 95.0)
    thermal_tau_min: float = 9.0      # first-order lag toward setpoint
    lamp_power_new: tuple = (57.0, 63.0)  # new-lamp drive %, varies per lamp
    lamp_aging_pct_per_day: float = 0.05
    noise: dict = field(default_factory=lambda: {
        "irradiance": 0.004, "lamp_power_pct": 0.25, "black_panel_c": 0.35,
        "chamber_air_c": 0.3, "rh_pct": 1.2,
    })

    @property
    def cycle_min(self) -> int:
        return self.light_min + self.dark_min

    @property
    def n_samples(self) -> int:
        return self.days * 24 * 60 // self.sample_min


@dataclass
class Fault:
    type: str
    start: int  # sample index, inclusive
    end: int    # sample index, inclusive

    def as_dict(self) -> dict:
        return {"type": self.type, "start": self.start, "end": self.end}


def _first_order(setpoint: np.ndarray, tau_samples: float, x0: float) -> np.ndarray:
    alpha = 1 - np.exp(-1 / tau_samples)
    out = np.empty_like(setpoint)
    x = x0
    for i, s in enumerate(setpoint):
        x = x + alpha * (s - x)
        out[i] = x
    return out


def simulate_run(seed: int, faults: list[str] | None = None,
                 cfg: ChamberConfig | None = None) -> tuple[pd.DataFrame, list[Fault]]:
    """Simulate one test run. `faults` lists fault types to inject (each at most once)."""
    cfg = cfg or ChamberConfig()
    faults = faults or []
    unknown = set(faults) - set(FAULT_TYPES)
    if unknown:
        raise ValueError(f"unknown fault types: {unknown}")

    rng = np.random.default_rng(seed)
    n = cfg.n_samples
    per_day = 24 * 60 // cfg.sample_min
    minutes = np.arange(n) * cfg.sample_min
    t_cycle = minutes % cfg.cycle_min
    light = t_cycle < cfg.light_min
    t_in_phase = np.where(light, t_cycle, t_cycle - cfg.light_min)
    tau = cfg.thermal_tau_min / cfg.sample_min

    def sp(pair):
        return np.where(light, pair[0], pair[1]).astype(float)

    days_elapsed = minutes / 1440.0
    ambient = 0.4 * np.sin(2 * np.pi * days_elapsed)  # lab ambient day/night effect

    bp_sp = sp(cfg.black_panel_sp)
    air_sp = sp(cfg.chamber_air_sp) + ambient
    rh_sp = sp(cfg.rh_sp)
    irr_sp = sp(cfg.irradiance_sp)

    # Lamp: required drive to hold setpoint, capped at 100 %
    p0 = rng.uniform(*cfg.lamp_power_new)
    required = p0 + cfg.lamp_aging_pct_per_day * days_elapsed

    truth: list[Fault] = []
    # Choose non-overlapping onsets. Keep day 0-2 clean (per-run baseline window).
    earliest = 3 * per_day
    slots = {}

    def pick_start(length: int, light_only: bool = False) -> int:
        for _ in range(5000):
            s = int(rng.integers(earliest, n - length - per_day))
            if light_only and not light[s:s + length].all():
                continue
            if all(s > e + per_day or s + length < b - per_day for b, e in slots.values()):
                return s
        raise RuntimeError("could not place fault")

    # Place the long lamp fault first so the short faults fit around it.
    faults = sorted(faults, key=lambda f: f != "lamp_degradation")

    rh_offset = np.zeros(n)
    air_offset = np.zeros(n)
    irr_override = np.full(n, np.nan)

    for f in faults:
        if f == "sensor_spike":
            length = int(rng.integers(2, 6))
            s = pick_start(length, light_only=True)
            irr_override[s:s + length] = rng.choice(
                [rng.uniform(0.85, 1.2), 0.0], size=length)
            truth.append(Fault(f, s, s + length - 1))
        elif f == "control_fault":
            length = int(rng.integers(4, 9) * 60 / cfg.sample_min)  # 4-8 h
            s = pick_start(length)
            k = np.arange(length)
            period = rng.uniform(40, 80) / cfg.sample_min
            air_offset[s:s + length] = rng.uniform(4, 7) * np.sin(2 * np.pi * k / period) + 2
            truth.append(Fault(f, s, s + length - 1))
        elif f == "lamp_degradation":
            s = int(rng.integers(14 * per_day, 20 * per_day))
            rate = rng.uniform(3.0, 5.0)  # % per day, vs. 0.05 normal
            extra = np.clip(days_elapsed - days_elapsed[s], 0, None) * rate
            required = required + extra
            truth.append(Fault(f, s, n - 1))
            slots[f] = (s, n - 1)
            continue
        elif f == "seal_leak":
            length = int(rng.integers(12, 25) * 60 / cfg.sample_min)  # 12-24 h
            s = pick_start(length)
            ramp = min(length, int(6 * 60 / cfg.sample_min))
            prof = np.concatenate([np.linspace(0, 1, ramp), np.ones(length - ramp)])
            rh_offset[s:s + length] = rng.uniform(12, 22) * prof
            truth.append(Fault(f, s, s + length - 1))
        slots[f] = (truth[-1].start, truth[-1].end)

    lamp_power = np.minimum(required, 100.0)
    irr_capacity = np.minimum(1.0, 100.0 / required)  # irradiance falls once power is capped
    irr_true = _first_order(irr_sp * irr_capacity, 0.6, irr_sp[0])
    lamp_power_obs = np.where(light, lamp_power, 0.0)

    bp = _first_order(bp_sp + air_offset * 0.8, tau, bp_sp[0])
    air = _first_order(air_sp + air_offset, tau, air_sp[0])
    rh = _first_order(rh_sp + np.where(light, rh_offset, rh_offset * 0.2), tau * 1.3, rh_sp[0])

    nz = cfg.noise
    df = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-08-01") + pd.to_timedelta(minutes, unit="min"),
        "phase": np.where(light, "light", "dark"),
        "t_in_phase_min": t_in_phase,
        "irradiance": irr_true + rng.normal(0, nz["irradiance"], n) * light,
        "lamp_power_pct": lamp_power_obs + rng.normal(0, nz["lamp_power_pct"], n) * light,
        "black_panel_c": bp + rng.normal(0, nz["black_panel_c"], n),
        "chamber_air_c": air + rng.normal(0, nz["chamber_air_c"], n),
        "rh_pct": np.clip(rh + rng.normal(0, nz["rh_pct"], n), 0, 100),
    })
    spike = ~np.isnan(irr_override)
    df.loc[spike, "irradiance"] = irr_override[spike]
    df["run_seed"] = seed
    truth.sort(key=lambda f: f.start)
    return df, truth


def truth_mask(n: int, truth: list[Fault]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for f in truth:
        m[f.start:f.end + 1] = True
    return m
