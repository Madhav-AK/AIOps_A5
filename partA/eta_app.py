import os
import time
from typing import Literal
from uuid import uuid4
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException


MODEL_PATH = os.getenv("ETA_MODEL", "data/eta_v1.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_columns"]
VERSION = bundle["version"]

METRICS = CollectorRegistry()
REQUESTS = Counter(
    "eta_requests_total",
    "Total ETA service requests",
    ["path", "status"],
    registry=METRICS,
)
LATENCY = Histogram(
    "eta_request_latency_seconds",
    "ETA service request latency",
    registry=METRICS,
)


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: Literal[0, 1]
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery")
        return self


class PredictRequest(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


app = FastAPI(title="SwiftParcel service")


def request_id(request):
    return request.state.request_id


@app.middleware("http")
async def observe(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse(
            status_code=500,
            content={"error": "inference_failed", "request_id": request_id(request)},
        )

    response.headers["X-Request-ID"] = request_id(request)
    if request.url.path != "/metrics":
        REQUESTS.labels(path=request.url.path, status=str(response.status_code)).inc()
        LATENCY.observe(time.perf_counter() - start)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "msg": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "details": details,
            "request_id": request_id(request),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    error = "inference_failed" if exc.status_code == 500 else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error, "request_id": request_id(request)},
    )


@app.get("/healthz")
def healthz(request: Request):
    return {
        "status": "ok",
        "model_version": VERSION,
        "request_id": request_id(request),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(METRICS), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rows = [delivery.model_dump() for delivery in req.instances]
    try:
        predictions = model.predict(pd.DataFrame(rows)[FEATURES])
    except Exception:
        raise HTTPException(status_code=500, detail="inference_failed") from None
    return {
        "predictions": [round(float(value), 1) for value in predictions],
        "model_version": VERSION,
        "request_id": request_id(request),
    }
