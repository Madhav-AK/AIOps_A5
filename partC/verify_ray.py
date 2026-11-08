import requests
import ray
from ray import serve
from eta_ray import build


URL = "http://127.0.0.1:8100/v1/models/eta:predict"
ROW = {
    "distance_km": 12.5,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.4,
    "rain": 0,
    "weight_kg": 8.0,
}


def arm(customer_id, beta=False):
    headers = {"X-Beta": "1"} if beta else None
    response = requests.post(
        URL,
        json={"customer_id": customer_id, "instances": [ROW]},
        headers=headers,
    )
    response.raise_for_status()
    return response.json()["arm"]


def deploy(weight):
    serve.run(build(weight=weight), name="default", route_prefix="/")


def main():
    evidence = []
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        logging_level="ERROR",
        log_to_driver=False,
    )
    serve.start(
        http_options={"host": "127.0.0.1", "port": 8100},
        logging_config={"log_level": "ERROR"},
    )

    customers = [f"customer-{number}" for number in range(1000)]
    deploy(0.2)
    first = {customer: arm(customer) for customer in customers}
    share = sum(value == "v2" for value in first.values()) / len(first)
    assert 0.16 <= share <= 0.24
    evidence.append(f"candidate_share={share:.3f} (expected 0.20 +/- 0.04)")

    repeated = all(arm(customer) == first[customer] for customer in customers[:200])
    assert repeated
    evidence.append("sticky_second_call=PASS (200/200 unchanged)")

    deploy(0.2)
    after_redeploy = all(arm(customer) == first[customer] for customer in customers[:200])
    assert after_redeploy
    evidence.append("sticky_after_redeploy=PASS (200/200 unchanged)")

    deploy(0.5)
    raised = {customer: arm(customer) for customer in customers}
    monotonic = all(
        raised[customer] == "v2"
        for customer in customers
        if first[customer] == "v2"
    )
    assert monotonic
    evidence.append("monotonic_weight_increase=PASS (no v2 customer moved to v1)")

    override = all(arm(customer, beta=True) == "v2" for customer in customers[:50])
    assert override
    evidence.append("x_beta_override=PASS (50/50 routed to v2)")

    ray.shutdown()
    print("\n".join(evidence))


if __name__ == "__main__":
    main()
