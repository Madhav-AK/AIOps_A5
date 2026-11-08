AI OPs Assignment 5 Submission
- By Madhav AK
- DA24B012

This README file goes through all major submission points.

- Main report can be found as `REPORT.pdf` in the main directory.
- AI usage details can be found in `AI_DISCLOSURE.md`.
- Video Link: https://drive.google.com/file/d/153MN5Xhv1hs7z5O9uS6dcIJ7_jp3flIW/view?usp=sharing

- `eta_data.py` creates the data and the two model bundles used in the assignment.
- `partA` contains the Part A submission files, and a screenshot folder of proofs.
- `partB` contains the Part B submission files, and a screenshot folder of proofs.
- `partC` contains the Part C submission files, and a screenshot folder of proofs.
- `partD` contains the Part D submission files, and a screenshot folder of proofs.


### Setup

```bash
cd ~/AIOps_A5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
```

### Part A

Start the FastAPI service:

```bash
cd ~/AIOps_A5
source .venv/bin/activate
ETA_MODEL=data/eta_v1.joblib uvicorn partA.eta_app:app --port 8000
```

Then to run the tests:

```bash
cd ~/AIOps_A5
source .venv/bin/activate

pytest -q partA/test_eta_app.py

python partA/load_test.py \
  --url http://localhost:8000/v1/models/eta:predict \
  --data data/eta_sample.csv \
  --concurrency 1 2 4 8 16 \
  --duration 4 \
  --out partA/loadtest_results.json
```

### Part B

To run the Part B code:

```bash
cd ~/AIOps_A5
source .venv/bin/activate

python -m partB.eta_mlflow
pytest -q partB/test_gate.py
python -m partB.print_both_gates
```

To run the shadow comparison:

```bash
# Terminal 1: prod
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m "models:/eta-model@prod" -p 5101 --env-manager local

# Terminal 2: candidate
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m "models:/eta-model@candidate" -p 5102 --env-manager local

# Terminal 3: shadow comparison
python -m partB.shadow_compare
```

### Part C

To run the verification script:

```bash
cd ~/AIOps_A5
source .venv/bin/activate
python partC/verify_ray.py
```

### Part D

Start the Part A service first. Then to run the smoke test:

```bash
cd ~/AIOps_A5
source .venv/bin/activate

python partD/smoke_eta.py \
  --url http://localhost:8000 \
  --expect-version v1 \
  --max-p95-ms 80
```
