import hashlib
import time
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle


FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg",]


@serve.deployment(
    ray_actor_options={"num_cpus": 0.25},
    max_ongoing_requests=8,
    logging_config={"log_level": "WARNING"},
)
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.version = bundle["version"]
        self.extra_work_s = extra_work_ms / 1000

    async def __call__(self, rows: list[dict]):
        predictions = self.model.predict(pd.DataFrame(rows)[FEATURES])
        if self.extra_work_s:
            time.sleep(self.extra_work_s)
        return {
            "predictions": [round(float(value), 1) for value in predictions],
            "model_version": self.version,
        }


class PredictRequest(BaseModel):
    customer_id: str
    instances: list[dict]


api = FastAPI()


def candidate_customer(customer_id: str, weight: float) -> bool:
    bucket = int.from_bytes(hashlib.sha256(customer_id.encode()).digest(), "big") / 2**256
    return bucket < weight


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    logging_config={"log_level": "WARNING"},
)
@serve.ingress(api)
class Router:
    def __init__(self, prod: DeploymentHandle, candidate: DeploymentHandle, weight: float,):
        self.prod = prod
        self.candidate = candidate
        self.weight = weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: PredictRequest, request: Request):
        use_candidate = request.headers.get("X-Beta") == "1" or candidate_customer(req.customer_id, self.weight)
        arm = "v2" if use_candidate else "v1"
        handle = self.candidate if use_candidate else self.prod
        result = await handle.remote(req.instances)
        return {**result, "arm": arm}


def build(weight=0.1, extra_work_ms=0.0):
    prod = EtaModel.options(name="EtaProd", num_replicas=2).bind("data/eta_v1.joblib", extra_work_ms)
    candidate = EtaModel.options(
        name="EtaCandidate",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 1,
            "upscale_delay_s": 2,
            "downscale_delay_s": 10,
        },
    ).bind("data/eta_v2.joblib", extra_work_ms)
    return Router.bind(prod, candidate, weight)


app = build()
