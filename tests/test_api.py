import json

import joblib
import pytest
from fastapi.testclient import TestClient

from weathering_ad import api
from weathering_ad.simulator import CHANNELS, simulate_run


@pytest.fixture(scope="module")
def client(tmp_path_factory, fitted):
    path = tmp_path_factory.mktemp("model") / "residual_z.joblib"
    joblib.dump(fitted["residual_z"], path)
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(path))
    api.get_model.cache_clear()
    yield TestClient(api.app)
    mp.undo()
    api.get_model.cache_clear()


def payload(df, instrument="Ci5000-demo-01"):
    cols = ["timestamp", "phase", "t_in_phase_min", *CHANNELS]
    return {"instrument_id": instrument,
            "readings": json.loads(df[cols].to_json(orient="records", date_format="iso"))}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_healthy_run_returns_no_alerts(client):
    df, _ = simulate_run(600)
    r = client.post("/score", json=payload(df))
    assert r.status_code == 200, r.text
    assert r.json()["alerts"] == []


def test_lamp_degradation_alert_is_returned_and_ongoing(client):
    df, truth = simulate_run(505, ["lamp_degradation"])
    r = client.post("/score", json=payload(df))
    body = r.json()
    lamp = [a for a in body["alerts"] if a["channel"] == "lamp_power_pct"]
    assert lamp and lamp[-1]["ongoing"]
    assert body["model_version"].startswith("residual_z")


def test_rejects_too_little_history(client):
    df, _ = simulate_run(600)
    r = client.post("/score", json=payload(df.iloc[:100]))
    assert r.status_code == 422
    assert "24 h" in r.text


def test_rejects_out_of_order_timestamps(client):
    df, _ = simulate_run(600)
    df = df.iloc[:300].copy()
    df.iloc[[10, 11]] = df.iloc[[11, 10]].to_numpy()
    assert client.post("/score", json=payload(df)).status_code == 422


def test_rejects_physically_impossible_values(client):
    df, _ = simulate_run(600)
    df = df.iloc[:300].copy()
    df.loc[5, "rh_pct"] = 150
    assert client.post("/score", json=payload(df)).status_code == 422
