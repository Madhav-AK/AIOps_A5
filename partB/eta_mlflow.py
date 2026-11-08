import os
import time
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error
import eta_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "eta-model"

mlflow.set_tracking_uri(TRACKING_URI)

class EtaPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        bundle = joblib.load(context.artifacts["bundle"])
        self.model = bundle["model"]
        self.features = bundle["feature_columns"]
        self.version = bundle["version"]

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None,) -> pd.DataFrame:
        predictions = self.model.predict(pd.DataFrame(model_input)[self.features])
        return pd.DataFrame(
            {
                "eta_minutes": np.round(predictions, 1),
                "model_version": self.version,
            }
        )

def register():
    mlflow.set_experiment("eta-serving")
    sample = eta_data.make_data(5, seed=99)[eta_data.FEATURES].astype("float64")
    output_example = pd.DataFrame(
        {"eta_minutes": [10.0], "model_version": ["v1"]}
    )
    signature = infer_signature(sample, output_example)
    requirements = [
        "mlflow==3.15.1",
        "scikit-learn==1.9.0",
        "pandas==2.3.3",
        "joblib==1.5.3",
        "numpy==2.5.2",
    ]

    versions = {}
    for version in ("v1", "v2"):
        bundle = f"data/eta_{version}.joblib"
        with mlflow.start_run(run_name=f"eta-{version}"):
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=EtaPyfunc(),
                artifacts={"bundle": bundle},
                signature=signature,
                input_example=sample.head(2),
                pip_requirements=requirements,
                registered_model_name=MODEL_NAME,
            )
        versions[version] = info.registered_model_version

    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "prod", versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", versions["v2"])
    return {
        alias: client.get_model_version_by_alias(MODEL_NAME, alias).version
        for alias in ("prod", "candidate")
    }

def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    criteria = {
        "relative_mae_gain": bool((prod["mae"] - cand["mae"]) / prod["mae"] >= min_rel_gain),
        "latency": bool(cand["p95_ms"] <= max_p95_ms),
        "rush_slice": bool(cand["mae_rush"] <= prod["mae_rush"] + max_slice_regression),
        "rain_slice": bool(cand["mae_rain"] <= prod["mae_rain"] + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}

def _evaluate(alias, fresh):
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    features = fresh[eta_data.FEATURES].astype("float64")
    predictions = model.predict(features)["eta_minutes"].to_numpy()
    rush = fresh.hour.isin([7, 8, 9, 16, 17, 18, 19])
    rain = (fresh.rain == 1)

    latencies = []
    for row in features.head(200).to_dict(orient="records"):
        start = time.perf_counter()
        model.predict(pd.DataFrame([row]))
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mae": mean_absolute_error(fresh.eta_minutes, predictions),
        "mae_rush": mean_absolute_error(fresh.loc[rush, "eta_minutes"], predictions[rush]),
        "mae_rain": mean_absolute_error(fresh.loc[rain, "eta_minutes"], predictions[rain]),
        "p95_ms": float(np.percentile(latencies, 95)),
    }

def promotion_gate(**thresholds):
    fresh = eta_data.make_data(2000, seed=99)
    prod = _evaluate("prod", fresh)
    cand = _evaluate("candidate", fresh)
    return {"prod": prod, "candidate": cand, **decide(prod, cand, **thresholds)}

if __name__ == "__main__":
    print(register())
