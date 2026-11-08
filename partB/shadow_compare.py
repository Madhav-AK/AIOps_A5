import time
import numpy as np
import pandas as pd
import requests
import eta_data


rows = pd.read_csv("data/eta_fresh.csv")[eta_data.FEATURES].head(200)
latencies = {"prod": [], "candidate": []}
predictions = {"prod": [], "candidate": []}
urls = {
    "prod": "http://localhost:5101/invocations",
    "candidate": "http://localhost:5102/invocations",
}

for row in rows.to_dict(orient="records"):
    body = {"dataframe_records": [row]}
    for name, url in urls.items():
        start = time.perf_counter()
        response = requests.post(url, json=body)
        latencies[name].append((time.perf_counter() - start) * 1000)
        predictions[name].append(response.json()["predictions"][0]["eta_minutes"])

differences = np.abs(np.array(predictions["prod"]) - predictions["candidate"])
print(f"mean_absolute_difference_minutes={differences.mean():.3f}")
print(f"percentage_over_10_minutes={(differences > 10).mean() * 100:.1f}")
print(f"prod_p95_ms={np.percentile(latencies['prod'], 95):.1f}")
print(f"candidate_p95_ms={np.percentile(latencies['candidate'], 95):.1f}")
