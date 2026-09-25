import copy

import numpy as np
import pandas as pd
import pytest

from weathering_ad.detectors import (BASELINE_SAMPLES, WARMUP_SAMPLES, DriftCusum, cusum,
                                     debounce, residual_features, to_alerts)
from weathering_ad.evaluate import match
from weathering_ad.simulator import CHANNELS, FAULT_TYPES, simulate_run

VAL_FAULTY = range(500, 506)
VAL_CLEAN = range(600, 605)


# --- unit tests for building blocks -----------------------------------------

def test_cusum_resets_after_signal():
    x = np.r_[np.zeros(20), np.full(40, 3.0)]
    active = np.ones_like(x, dtype=bool)
    sig = cusum(x, k=0.5, active=active, h=5.0)
    assert not sig[:20].any()
    assert sig.sum() > 5  # a sustained shift keeps re-signalling after each reset
    assert not cusum(np.zeros(100), 0.5, np.ones(100, bool), h=5.0).any()


def test_cusum_does_not_latch_after_transient():
    """Regression test: a latched drift alarm once hid real faults for days."""
    x = np.r_[np.zeros(10), np.full(10, 3.0), np.zeros(60)]
    sig = cusum(x, k=0.5, active=np.ones_like(x, dtype=bool), h=5.0)
    assert sig[10:20].any()
    assert not sig[22:].any()  # stops as soon as the excursion ends


def test_cusum_holds_value_when_inactive():
    x = np.full(10, 2.0)
    active = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    s = cusum(x, k=0.5, active=active)
    assert s[3] == s[2] == s[5]


def test_debounce_drops_isolated_flags_only():
    f = pd.DataFrame({"a": [0, 1, 0, 0, 1, 1, 0, 0], "b": [0] * 8}).astype(bool)
    out = debounce(f, min_hits=2, window=3)
    assert out.a.tolist() == [False, False, False, False, False, True, False, False]


def test_alerts_are_per_channel():
    f = pd.DataFrame(False, index=range(100), columns=CHANNELS)
    f.loc[10:40, "irradiance"] = True     # long-running alert on one sensor...
    f.loc[25:30, "rh_pct"] = True         # ...must not hide a new fault on another
    alerts = to_alerts(f)
    assert {a["channel"] for a in alerts} == {"irradiance", "rh_pct"}
    assert next(a for a in alerts if a["channel"] == "rh_pct")["start"] == 25


# --- methodology guarantees ---------------------------------------------------

def test_scoring_does_not_mutate_fitted_detector(fitted):
    det = fitted["residual_z"]
    before = copy.deepcopy(det.__dict__)
    df, _ = simulate_run(501, FAULT_TYPES)
    det.score(df)
    assert det.drift_.h_ == before["drift_"].h_
    pd.testing.assert_frame_equal(det.profile_.mean_, before["profile_"].mean_)


def test_features_are_causal(fitted):
    """Numeric features at t must not depend on data after t.

    Checked on the features, not only on the flags: flags are almost all False,
    so a leaky feature can hide behind them (a centered rolling window slipped
    through a flags-only test).
    """
    prof = fitted["residual_z"].profile_
    df, _ = simulate_run(502, FAULT_TYPES)
    z_full = prof.zscores(df)
    f_full = residual_features(z_full)
    d_full = DriftCusum._inputs(z_full)
    for k in (BASELINE_SAMPLES + 1, 2500, 5000):
        z_pre = prof.zscores(df.iloc[:k].copy())
        pd.testing.assert_frame_equal(z_pre, z_full.iloc[:k])
        pd.testing.assert_frame_equal(residual_features(z_pre), f_full.iloc[:k])
        for c, x in DriftCusum._inputs(z_pre).items():
            np.testing.assert_allclose(x, d_full[c][:k])


@pytest.mark.parametrize("name", ["residual_z", "iforest_cusum", "static_threshold"])
def test_scores_are_causal(fitted, name):
    """Flags at time t must not depend on data after t (so it can run on a live stream)."""
    df, _ = simulate_run(502, FAULT_TYPES)
    full = fitted[name].score(df)
    for k in (BASELINE_SAMPLES + 1, 2500, 5000):
        prefix = fitted[name].score(df.iloc[:k].copy())
        pd.testing.assert_frame_equal(prefix, full.iloc[:k])


def test_no_alarms_during_warmup(fitted):
    df, _ = simulate_run(503, FAULT_TYPES)
    for det in fitted.values():
        assert not det.score(df).iloc[:WARMUP_SAMPLES].any().any()


# --- behaviour on held-out validation runs -------------------------------------

@pytest.mark.parametrize("seed", VAL_CLEAN)
def test_residual_z_quiet_on_healthy_runs(fitted, seed):
    df, _ = simulate_run(seed)
    assert to_alerts(fitted["residual_z"].score(df)) == []


@pytest.mark.parametrize("seed", VAL_FAULTY)
def test_residual_z_catches_every_fault_type(fitted, seed):
    df, truth = simulate_run(seed, FAULT_TYPES)
    m = match(to_alerts(fitted["residual_z"].score(df)), truth)
    missed = [r["type"] for r in m["faults"] if not r["detected"]]
    assert missed == []
    assert m["false_alarms"] == []


@pytest.mark.parametrize("seed", VAL_FAULTY)
def test_lamp_degradation_caught_days_before_static_alarm(fitted, seed):
    df, truth = simulate_run(seed, FAULT_TYPES)

    def ttd(name):
        r = match(to_alerts(fitted[name].score(df)), truth)["faults"]
        return next(x["ttd_h"] for x in r if x["type"] == "lamp_degradation")

    assert ttd("residual_z") < 12
    assert np.isnan(ttd("static_threshold")) or ttd("static_threshold") - ttd("residual_z") > 72


def test_iforest_score_saturates_outside_training_range(fitted):
    """Characterizes why Isolation Forest is not the shipped detector.

    Trained on healthy data only, its score barely grows as a deviation gets
    more extreme (splits are bounded by the training range). A humidity
    excursion 100x larger than anything seen in training still scores below
    the alarm threshold.
    """
    det = fitted["iforest_cusum"]
    df, truth = simulate_run(504, ["seal_leak"])
    f = truth[0]
    X = residual_features(det.profile_.zscores(df)).iloc[[f.start + 80]]
    rh_cols = [c for c in X.columns if "rh_pct" in c]
    scores = []
    for mult in (1, 10, 100):
        Xm = X.copy()
        Xm[rh_cols] *= mult
        scores.append(-det.model_.score_samples(Xm)[0])
    assert scores[-1] - scores[0] < 0.1
    assert scores[-1] < det.threshold_
