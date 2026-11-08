from uuid import UUID
import pandas as pd
from fastapi.testclient import TestClient

from partA import eta_app


client = TestClient(eta_app.app)
URL = "/v1/models/eta:predict"
ROW = {
    "distance_km": 12.5,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.4,
    "rain": 0,
    "weight_kg": 8.0,
}


def test_valid_request():
    response = client.post(URL, json={"instances": [ROW]})
    body = response.json()
    assert response.status_code == 200
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] == round(body["predictions"][0], 1)


def test_batch_order_is_preserved():
    rows = [ROW, {**ROW, "distance_km": 40}, {**ROW, "distance_km": 5}]
    expected = eta_app.model.predict(pd.DataFrame(rows)[eta_app.FEATURES])
    response = client.post(URL, json={"instances": rows})
    assert response.json()["predictions"] == [round(float(x), 1) for x in expected]


def test_out_of_range_value():
    response = client.post(URL, json={"instances": [{**ROW, "hour": 24}]})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"


def test_invalid_binary_value():
    response = client.post(URL, json={"instances": [{**ROW, "rain": 2}]})
    assert response.status_code == 422


def test_missing_feature():
    row = {key: value for key, value in ROW.items() if key != "weight_kg"}
    response = client.post(URL, json={"instances": [row]})
    assert response.status_code == 422
    assert any("weight_kg" in item["field"] for item in response.json()["details"])


def test_infeasible_delivery():
    row = {**ROW, "distance_km": 151, "n_stops": 21}
    response = client.post(URL, json={"instances": [row]})
    assert response.status_code == 422
    assert "infeasible" in str(response.json()).lower()


def test_empty_batch():
    assert client.post(URL, json={"instances": []}).status_code == 422


def test_batch_larger_than_500():
    assert client.post(URL, json={"instances": [ROW] * 501}).status_code == 422


def test_request_id_is_echoed():
    response = client.post(URL, json={"instances": [ROW]}, headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert response.json()["request_id"] == "abc-123"


def test_request_id_is_generated():
    response = client.post(URL, json={"instances": [ROW]})
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
    UUID(response.json()["request_id"])


def test_request_id_is_present_on_validation_error():
    response = client.post(URL, json={"instances": []}, headers={"X-Request-ID": "bad-request"})
    assert response.headers["X-Request-ID"] == "bad-request"
    assert response.json()["request_id"] == "bad-request"


def test_request_id_is_generated_on_validation_error():
    response = client.post(URL, json={"instances": []})
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
    UUID(response.json()["request_id"])


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == "v1"


def test_metrics_output_and_metrics_request_not_counted():
    response = client.get("/metrics")
    text = response.text
    assert response.status_code == 200
    assert "eta_requests_total" in text
    assert 'path="/v1/models/eta:predict"' in text
    assert "eta_request_latency_seconds_bucket" in text
    assert 'le="+Inf"' in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    assert eta_app.METRICS.get_sample_value(
        "eta_requests_total", {"path": "/metrics", "status": "200"}
    ) is None


def test_inference_error_is_generic(monkeypatch):
    def fail(_):
        raise RuntimeError("secret model detail")

    monkeypatch.setattr(eta_app.model, "predict", fail)
    response = client.post(URL, json={"instances": [ROW]})
    assert response.status_code == 500
    assert response.json()["error"] == "inference_failed"
    assert "secret" not in response.text
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
