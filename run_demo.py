"""End-to-end demo: calibrate on healthy runs, evaluate on held-out faulty runs.

    python run_demo.py            # full evaluation (~1 min)
    python run_demo.py --quick    # smaller evaluation

Outputs go to ./outputs/: results table (CSV + Markdown), chart, fitted model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from weathering_ad.detectors import DETECTORS, ResidualZ, to_alerts  # noqa: E402
from weathering_ad.evaluate import evaluate, match, summarize  # noqa: E402
from weathering_ad.simulator import FAULT_TYPES, ChamberConfig, simulate_run  # noqa: E402

OUT = Path("outputs")

# Seed ranges are disjoint. Protocol:
#   calibration  healthy runs the detectors are fit on (a fleet's history of good runs)
#   validation   faulty runs used during development to make design choices
#                (500-529 faulty, 600-609 clean). Not used in this script.
#   test         fresh runs, only used for the final numbers below.
# Seeds 1000-1049 / 2000-2019 were an earlier test set. It exposed the latching-
# CUSUM bug, so those seeds are retired and the final report uses new ones.
CALIB_SEEDS = range(0, 10)
TEST_SEEDS = range(3000, 3050)
CLEAN_SEEDS = range(4000, 4020)


def noisy_cfg(factor: float) -> ChamberConfig:
    cfg = ChamberConfig()
    cfg.noise = {k: v * factor for k, v in cfg.noise.items()}
    return cfg


def run(quick: bool) -> None:
    OUT.mkdir(exist_ok=True)
    test_seeds = list(TEST_SEEDS)[:10] if quick else list(TEST_SEEDS)
    clean_seeds = list(CLEAN_SEEDS)[:5] if quick else list(CLEAN_SEEDS)

    calib = [simulate_run(s)[0] for s in CALIB_SEEDS]
    test = [simulate_run(s, FAULT_TYPES) for s in test_seeds]
    clean = [simulate_run(s) for s in clean_seeds]
    # Distribution shift: sensors 50 % noisier than in calibration.
    shifted = [simulate_run(s, FAULT_TYPES, noisy_cfg(1.5)) for s in test_seeds]

    det_rows, fa_rows = [], []
    fitted = {}
    clean_days = len(clean) * 30
    for name, D in DETECTORS.items():
        det = D().fit(calib)
        fitted[name] = det
        res, fa_faulty = evaluate(det, test)
        _, fa_clean = evaluate(det, clean)
        res_s, fa_faulty_s = evaluate(det, shifted)
        _, fa_clean_s = evaluate(det, [(simulate_run(s, cfg=noisy_cfg(1.5))[0], [])
                                       for s in clean_seeds])
        summ, summ_s = summarize(res), summarize(res_s)
        for fault, r in summ.iterrows():
            det_rows.append({
                "detector": name, "fault": fault,
                "detection_rate": r.detection_rate,
                "median_ttd_h": r.median_ttd_h, "p90_ttd_h": r.p90_ttd_h,
                "detection_rate_noisy_sensors": summ_s.loc[fault, "detection_rate"],
            })
        fa_rows.append({"detector": name,
                        "fa_per_day_clean_runs": fa_clean,
                        "fa_per_day_faulty_runs": fa_faulty,
                        "fa_per_day_clean_runs_noisy": fa_clean_s,
                        "fa_per_day_faulty_runs_noisy": fa_faulty_s})

    detection = pd.DataFrame(det_rows)
    false_alarms = pd.DataFrame(fa_rows)
    detection.to_csv(OUT / "results_detection.csv", index=False)
    false_alarms.to_csv(OUT / "results_false_alarms.csv", index=False)
    note = (f"Test set: {len(test)} faulty runs, {len(clean)} clean runs "
            f"({clean_days} clean instrument-days). Zero false alarms in {clean_days} days "
            f"-> 95% upper bound ~{3 / clean_days:.3f}/day (rule of three). "
            f"'noisy' = all sensor noise x1.5 vs. calibration (distribution shift).")
    (OUT / "results.md").write_text(
        "## Detection (per fault type)\n\n" + detection.round(3).to_markdown(index=False)
        + "\n\n## False alarms per instrument-day\n\n"
        + false_alarms.round(3).to_markdown(index=False) + "\n\n" + note + "\n")
    print(detection.round(2).to_string(index=False))
    print()
    print(false_alarms.round(3).to_string(index=False))
    print(note)

    # Ship the simplest detector that wins (see results), not the fanciest.
    joblib.dump(fitted["residual_z"], OUT / "residual_z.joblib")
    meta = {"model": "residual_z", "calibration_seeds": list(CALIB_SEEDS),
            "cusum_h": {k: float(v) for k, v in fitted["residual_z"].drift_.h_.items()},
            "normal_lamp_aging_pct_per_day": fitted["residual_z"].profile_.aging_per_day_,
            "z_thresh": ResidualZ.z_thresh}
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))

    plot_example(fitted, test[0])


def _shade(ax, df, truth, colors):
    for f in truth:
        if f.type == "sensor_spike":  # too short to see as a band
            ax.axvline(df.timestamp[f.start], color=colors[f.type], lw=1.2, alpha=0.6)
        else:
            ax.axvspan(df.timestamp[f.start], df.timestamp[min(f.end + 1, len(df) - 1)],
                       color=colors[f.type], alpha=0.15, lw=0)


def plot_example(fitted: dict, run_: tuple) -> None:
    df, truth = run_
    colors = {"sensor_spike": "#e11d48", "control_fault": "#f59e0b",
              "lamp_degradation": "#7c3aed", "seal_leak": "#0891b2"}
    panels = [("irradiance", "Irradiance @340nm (W/m²)"), ("lamp_power_pct", "Lamp drive (%)"),
              ("chamber_air_c", "Chamber air (°C)"), ("rh_pct", "RH (%)")]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 3, 1.4]})
    # Show steady-state light-phase readings only; the raw light/dark cycling
    # turns every panel into a solid band at a 30-day scale.
    steady = (df.phase == "light") & (df.t_in_phase_min >= 30)
    spike_idx = [i for f in truth if f.type == "sensor_spike" for i in range(f.start, f.end + 1)]
    for ax, (c, label) in zip(axes, panels):
        ax.plot(df.timestamp[steady], df[c][steady], lw=0.7, color="#334155")
        if c == "irradiance" and spike_idx:
            ax.plot(df.timestamp[spike_idx], df[c].iloc[spike_idx], "o", ms=4,
                    color=colors["sensor_spike"])
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.2)
        _shade(ax, df, truth, colors)
    axes[0].set_title("Held-out test run: injected faults (shaded) vs. when each detector alerts "
                      "(steady-state light-phase readings)", fontsize=10)

    ax = axes[-1]
    names = ["static_threshold", "residual_z", "iforest_cusum"]
    for yi, name in enumerate(names):
        for a in to_alerts(fitted[name].score(df)):
            ax.plot([df.timestamp[a["start"]], df.timestamp[a["end"]]], [yi, yi],
                    lw=6, solid_capstyle="butt", color="#0f172a")
            ax.plot(df.timestamp[a["start"]], yi, "|", ms=14, mew=2, color="#dc2626")
    _shade(ax, df, truth, colors)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("Time (red tick = alert fires)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t], alpha=0.35) for t in colors]
    fig.legend(handles, list(colors), loc="upper right", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT / "detection_timeline.png", dpi=140)
    plt.close(fig)

    # Print the per-fault timing for this run (used in talking points)
    for name in names:
        m = match(to_alerts(fitted[name].score(df)), truth)
        print(name, [(r["type"], None if np.isnan(r["ttd_h"]) else round(r["ttd_h"], 1))
                     for r in m["faults"]], "false alarms:", len(m["false_alarms"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
