import numpy as np
import pandas as pd
import pytest

from weathering_ad.simulator import FAULT_TYPES, ChamberConfig, simulate_run, truth_mask


def test_same_seed_is_reproducible():
    a, ta = simulate_run(7, FAULT_TYPES)
    b, tb = simulate_run(7, FAULT_TYPES)
    pd.testing.assert_frame_equal(a, b)
    assert [f.as_dict() for f in ta] == [f.as_dict() for f in tb]


def test_different_seeds_differ():
    a, _ = simulate_run(1)
    b, _ = simulate_run(2)
    assert not np.allclose(a.irradiance, b.irradiance)


def test_cycle_schedule_matches_config():
    cfg = ChamberConfig()
    df, _ = simulate_run(0, cfg=cfg)
    assert len(df) == cfg.n_samples
    # 102 of every 120 minutes are light
    assert (df.phase == "light").mean() == pytest.approx(cfg.light_min / cfg.cycle_min)
    # phase changes every cycle, not every few days (the v1 bug)
    changes = (df.phase != df.phase.shift()).sum() - 1
    assert changes == pytest.approx(2 * cfg.days * 24 * 60 / cfg.cycle_min, abs=2)


def test_clean_run_has_no_faults():
    _, truth = simulate_run(3)
    assert truth == []


@pytest.mark.parametrize("seed", range(20))
def test_faults_are_placed_validly(seed):
    df, truth = simulate_run(seed, FAULT_TYPES)
    n = len(df)
    assert sorted(f.type for f in truth) == sorted(FAULT_TYPES)
    per_day = 24 * 60 // 6
    for f in truth:
        assert 3 * per_day <= f.start <= f.end < n  # first 3 days are clean baseline
        if f.type == "sensor_spike":
            assert (df.phase.iloc[f.start:f.end + 1] == "light").all()
    short = sorted((f for f in truth if f.type != "lamp_degradation"), key=lambda f: f.start)
    lamp = next(f for f in truth if f.type == "lamp_degradation")
    for a, b in zip(short, short[1:]):
        assert b.start > a.end  # no overlap between faults
    assert all(f.end < lamp.start for f in short)
    assert truth_mask(n, truth).sum() == sum(f.end - f.start + 1 for f in truth)


def test_closed_loop_hides_lamp_aging_until_power_caps():
    df, truth = simulate_run(11, ["lamp_degradation"])
    lamp = truth[0]
    steady = (df.phase == "light") & (df.t_in_phase_min >= 30)
    after = df[steady & (df.index > lamp.start)]
    uncapped = after[after.lamp_power_pct < 97]
    capped = after[after.lamp_power_pct > 99.5]
    assert len(uncapped) and len(capped)
    # irradiance held at setpoint while the lamp still has headroom...
    assert uncapped.irradiance.mean() == pytest.approx(0.55, abs=0.005)
    # ...and only falls once lamp drive is maxed out
    assert capped.irradiance.iloc[-50:].mean() < 0.52


def test_unknown_fault_rejected():
    with pytest.raises(ValueError):
        simulate_run(0, ["lamp_explosion"])
