import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weathering_ad.detectors import DETECTORS  # noqa: E402
from weathering_ad.simulator import simulate_run  # noqa: E402

CALIB_SEEDS = range(10)


@pytest.fixture(scope="session")
def calib_runs():
    return [simulate_run(s)[0] for s in CALIB_SEEDS]


@pytest.fixture(scope="session")
def fitted(calib_runs):
    return {name: D().fit(calib_runs) for name, D in DETECTORS.items()}
