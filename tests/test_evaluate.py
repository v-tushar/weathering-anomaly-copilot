import pytest

from weathering_ad.evaluate import GRACE, match
from weathering_ad.simulator import Fault


def test_alert_inside_window_is_detection_with_ttd():
    m = match([{"start": 110, "end": 150, "channel": "rh_pct"}], [Fault("seal_leak", 100, 200)])
    assert m["faults"][0]["detected"]
    assert m["faults"][0]["ttd_h"] == pytest.approx(10 * 6 / 60)
    assert m["false_alarms"] == []


def test_alert_starting_before_fault_is_a_false_alarm():
    m = match([{"start": 90, "end": 150, "channel": "rh_pct"}], [Fault("seal_leak", 100, 200)])
    assert not m["faults"][0]["detected"]
    assert len(m["false_alarms"]) == 1


def test_grace_period_after_fault_end():
    truth = [Fault("sensor_spike", 100, 102)]
    assert match([{"start": 102 + GRACE, "end": 120, "channel": "x"}], truth)["faults"][0]["detected"]
    assert not match([{"start": 103 + GRACE, "end": 120, "channel": "x"}], truth)["faults"][0]["detected"]


def test_multiple_alerts_for_one_fault_are_not_false_alarms():
    alerts = [{"start": 105, "end": 106, "channel": "chamber_air_c"},
              {"start": 107, "end": 108, "channel": "black_panel_c"}]
    m = match(alerts, [Fault("control_fault", 100, 150)])
    assert m["false_alarms"] == []
