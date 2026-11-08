import argparse
import time
import numpy as np
import requests


ROW = {
    "distance_km": 10.0,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.2,
    "rain": 0,
    "weight_kg": 8.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=float, default=80)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    path = "/v1/models/eta:predict"
    if base.endswith(path):
        predict_url = base
        base = base[: -len(path)]
    else:
        predict_url = f"{base}{path}"
    checks = []

    health = requests.get(f"{base}/healthz", timeout=5)
    checks.append(("health", health.status_code == 200))
    checks.append(
        ("health model version", health.json().get("model_version") == args.expect_version)
    )

    def predict(row):
        return requests.post(
            predict_url, json={"instances": [row]}, timeout=5
        )

    valid = predict(ROW)
    valid_body = valid.json()
    checks.append(("valid prediction", valid.status_code == 200))
    checks.append(
        ("prediction model version", valid_body.get("model_version") == args.expect_version)
    )

    baseline = valid_body["predictions"][0]
    longer = predict({**ROW, "distance_km": 30.0}).json()["predictions"][0]
    checks.append(("longer distance does not shorten ETA", longer >= baseline))

    heavier_traffic = predict({**ROW, "traffic_index": 0.9}).json()["predictions"][0]
    checks.append(("heavier traffic does not shorten ETA", heavier_traffic >= baseline))

    invalid = predict({**ROW, "hour": 24})
    checks.append(("invalid input returns 422", invalid.status_code == 422))

    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        predict(ROW)
        latencies.append((time.perf_counter() - start) * 1000)
    p95_ms = float(np.percentile(latencies, 95))
    checks.append((f"p95 {p95_ms:.1f} ms <= {args.max_p95_ms:.1f} ms", p95_ms <= args.max_p95_ms))

    for name, passed in checks:
        if passed:
            print("PASS:", name)
        else:
            print("FAIL:", name)
    raise SystemExit(0 if all(passed for _, passed in checks) else 1)


if __name__ == "__main__":
    main()
