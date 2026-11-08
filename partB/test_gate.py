import pytest
from partB.eta_mlflow import decide


PROD = {"mae": 10.0, "mae_rush": 11.0, "mae_rain": 12.0, "p95_ms": 5.0}
CAND = {"mae": 9.0, "mae_rush": 11.0, "mae_rain": 12.0, "p95_ms": 9.0}


def test_all_criteria_pass():
    result = decide(PROD, CAND)
    assert result["promote"] is True
    assert all(result["criteria"].values())


def test_relative_gain_at_boundary_passes():
    assert decide(PROD, {**CAND, "mae": 9.7})["promote"] is True


def test_relative_gain_just_outside_fails():
    result = decide(PROD, {**CAND, "mae": 9.701})
    assert result["criteria"]["relative_mae_gain"] is False
    assert result["promote"] is False


def test_latency_at_boundary_passes():
    assert decide(PROD, {**CAND, "p95_ms": 10.0})["promote"] is True


def test_latency_just_outside_fails():
    result = decide(PROD, {**CAND, "p95_ms": 10.001})
    assert result["criteria"]["latency"] is False
    assert result["promote"] is False


@pytest.mark.parametrize("slice_name", ["mae_rush", "mae_rain"])
def test_slice_at_boundary_passes(slice_name):
    candidate = {**CAND, slice_name: PROD[slice_name] + 0.05}
    assert decide(PROD, candidate)["promote"] is True


@pytest.mark.parametrize("slice_name", ["mae_rush", "mae_rain"])
def test_slice_just_outside_fails(slice_name):
    candidate = {**CAND, slice_name: PROD[slice_name] + 0.051}
    result = decide(PROD, candidate)
    expected = "rush_slice" if slice_name == "mae_rush" else "rain_slice"
    assert result["criteria"][expected] is False
    assert result["promote"] is False
