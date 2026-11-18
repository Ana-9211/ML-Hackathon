# PathSnap M4 — Two-Laptop Evaluation App

**Submission unit:** This folder — FastAPI on Laptop 1, HTML + Leaflet on Laptop 2, server-side inference/scoring, request logs and deployment instructions.

## Quick start and data access

The archive opens directly into the application folder; there is no extra `m4_app/m4_app` level. From this folder, run:

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\scripts\start_laptop1.ps1
```

The default mode is **demo**, so `/api/trips` immediately lists the two included synthetic trips (`demo_01`, `demo_02`). On Laptop 2, run `powershell -ExecutionPolicy Bypass -File .\scripts\start_laptop2.ps1`, open `http://localhost:5500`, enter Laptop 1's address, and click **Connect and load trips**.

The ZIP contains no real evaluator dataset or trained M3 model. For actual evaluation, place each trip's matching `.track`, `.nodes`, and `.arcs` files in `private_evaluator/` on Laptop 1, then start with `-Mode trained`. To use a data folder elsewhere, set `M4_EVAL_DIR` to its path. Relative paths resolve from this application folder, regardless of the terminal's working directory. Set `M4_MODE=demo` to return to the included samples.

## What this package implements

| M4 requirement | Implemented file | Verification |
|---|---|---|
| FastAPI backend (Laptop 1) | `backend/main.py` | `GET /api/health`, `GET /api/trips`, `POST /api/predict`; `/docs` |
| Private trip loading | `backend/dataset.py` | Supports flat/nested trip folders and validates `.track`, `.nodes`, `.arcs` |
| M3 model hook | `backend/pipeline.py` | Joblib binary candidate classifier or `M4_TEAM_PREDICTOR` adapter |
| Sequence route reconstruction | `backend/pipeline.py` | Candidate generation, emission scores, Viterbi demo/default adapter; use team M3 implementation as appropriate |
| Server-side reference scoring | `backend/scoring.py` | `.route` loaded **only after inference**, not passed to the predictor or UI |
| HTML/Leaflet UI (Laptop 2) | `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` | Trip dropdown, API POST request, route vs noisy GPS, metrics |
| Evaluator request logs | `backend/main.py` | Terminal `M4 REQUEST` lines and JSONL log (`logs/requests.jsonl`) |
| Integration and deployment | This README | Same-Wi-Fi/IPv4 network steps for two laptops |
| Tests | `tests/` | `python -m pytest -q` |

**Current state:** This is a runnable **M4 application and integration interface**, but it is **not a finished real-data evaluation** until your team supplies the actual 10 evaluator trips, independently trained M3 model/HMM pipeline and final validation. The supplied student-materials ZIP contains **five guideline PDFs, not the GPS dataset or trained weights**. Do not substitute the synthetic demonstration data in final evaluation.

## 1. Project structure

```text
m4_app/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints, CORS, request logs
│   ├── config.py            # environment-based configuration
│   ├── dataset.py           # input-only .track/.nodes/.arcs loader
│   ├── pipeline.py          # M3 model hook, candidate features, decoder
│   └── scoring.py           # PRIVATE .route reading and scoring
├── frontend/
│   ├── index.html           # Leaflet evaluation page
│   ├── styles.css
│   └── app.js
├── demo_data/               # tiny SYNTHETIC samples for smoke tests ONLY
├── private_evaluator/       # real evaluator files, Laptop 1 only, GIT-IGNORED
├── artifacts/               # insert final independently trained M3 model
├── logs/                    # request logs, GIT-IGNORED
├── tests/                   # contract, validation and integration tests
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

## 2. M4 data contract

A trip must provide a matching file trio:

- `trip_name.track`: headerless whitespace-separated `longitude latitude timestamp_seconds`.
- `trip_name.nodes`: headerless `longitude latitude`; node IDs are zero-based row indices.
- `trip_name.arcs`: headerless integer `from_node_id to_node_id`; arc IDs are zero-based row indices.
- `trip_name.route`: optional **server-private** reference sequence, one `arc_id` per row in chronological order (for scoring only).

Flat layout (`private_evaluator/eval_01.track`, `eval_01.nodes`, etc.) **and** one subfolder per trip (`private_evaluator/eval_01/*.track`, etc.) are supported. Nested trips may use another filename stem if it is the only complete trio inside their folder. Both layouts appear as `eval_01` in the UI.

A `.route` file is **not** one label per GPS observation. Accordingly, the built-in score is **normalized edit-distance sequence similarity** between consecutive-deduplicated predicted road arc IDs and the reference traversed arc sequence, as well as reference arc coverage. Do **not** present it as pointwise Top-1 accuracy. If organizers provide a GPS-to-reference alignment mapping, replace or extend the server-only `scoring.py` with a formally correct observation-level metric, leaving `.route` out of the prediction pipeline.

## 3. Laptop 1 setup (Windows PowerShell)

Open PowerShell in `m4_app`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place your real trip files under `m4_app/private_evaluator/` (or point `M4_EVAL_DIR` to their private folder). Never put this folder on Laptop 2 or push its contents to public GitHub.

Connect **your team's own M3 model**, using ONE of the two supported interfaces in section 5. Then:

```powershell
$env:M4_MODE = "trained"
$env:M4_EVAL_DIR = ".\private_evaluator"
$env:M4_MODEL_PATH = ".\artifacts\final_model.joblib"
# Optionally configure a custom M3 predictor, if your model needs it:
# $env:M4_TEAM_PREDICTOR = "your_package.inference:predict_trip"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
```

The backend must remain running **on Laptop 1** while the evaluator uses Laptop 2. `0.0.0.0` is the *bind address*, not a URL you enter in the UI. Find Laptop 1's LAN IP using `ipconfig` (for example, `192.168.1.10`). Both computers must share a reachable network; allow incoming TCP port **8000** through Windows Firewall for the private/local network.

Sanity-check the API in Laptop 1's browser:

- `http://localhost:8000/api/health` — returns status, mode, trip count, model-ready flag.
- `http://localhost:8000/api/trips` — lists available trip IDs (ten for the actual evaluator data).
- `http://localhost:8000/docs` — interactive FastAPI endpoint documentation.

### Linux/macOS equivalent

```bash
cd m4_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export M4_MODE=trained
export M4_EVAL_DIR=./private_evaluator
export M4_MODEL_PATH=./artifacts/final_model.joblib
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
```

## 4. Laptop 2 UI setup

Copy only the `frontend/` folder to Laptop 2 (or the whole `m4_app/` package *without* private evaluation data). In a new terminal, in `m4_app`:

```powershell
py -3 -m http.server 5500 --directory frontend --bind 0.0.0.0
```

Open **on Laptop 2**:

`http://localhost:5500`

Enter `http://<LAPTOP_1_IPV4>:8000` in the **Backend URL** box, e.g., `http://192.168.1.10:8000`, and click **Connect and load trips**. The interface obtains trip options from the backend, not hardcoded local arrays. Select a trip, click **Run map matching**, and inspect the blue noisy GPS trace, green road segments, sequence score and continuity. The UI does **not** load `.route` or run a local model.

**Network note:** The shipped CORS whitelist allows `http://localhost:5500` and `http://127.0.0.1:5500` as UI origins. If you open the UI via `http://<LAPTOP_2_IPV4>:5500`, add that exact origin on **Laptop 1** before starting FastAPI:

```powershell
$env:M4_ALLOWED_ORIGINS = "http://localhost:5500,http://127.0.0.1:5500,http://192.168.1.11:5500"
```

This UI loads Leaflet from a CDN and map tiles from OpenStreetMap, so an Internet connection is required for the background map. The API and data remain LAN-only; only map-library/tile assets are loaded from the Internet. The plain UI and numerical metrics can still load if map tiles fail.

## 5. Integration with M3 (must be done for the real submission)

### A. Standard joblib classifier

If your team's independently trained M3 candidate model accepts these **six features in this precise order**, save its fitted binary `predict_proba()` estimator/Pipeline to `artifacts/final_model.joblib` using `joblib.dump()`:

1. `gps_road_distance_m`
2. `bearing_difference_deg`
3. `road_segment_length_m`
4. `gps_step_distance_m`
5. `seconds_since_previous_observation`
6. `candidate_distance_rank`

**Positive class** must be `1` or `True`, indicating a candidate road segment is correct; the model should expose `.classes_` and `.predict_proba()` (e.g., an sklearn Pipeline). This code creates up to six road candidates per GPS point, supplies learned probabilities to Viterbi and returns inferred arc IDs. The exact features and candidate generation need to match your training workflow; **do not feed a differently trained model into these six arbitrary features and claim that it is validated**.

### B. Existing custom M3 pipeline (preferred when its features/decoder differ)

Provide a team-owned Python function with this contract:

```python
# your_package/inference.py -- IMPLEMENT THIS USING YOUR TEAM'S M3 CODE
from backend.dataset import TripInput

def predict_trip(trip: TripInput) -> list[int]:
    # trip.observations: (lon, lat, timestamp)
    # trip.nodes: (lon, lat)
    # trip.arcs: (0-based from_node_id, to_node_id)
    # Load your actual fitted model, feature builder and HMM/Viterbi decoder.
    # Return exactly one valid 0-based arc_id per observation.
    # There is intentionally NO reference route in TripInput.
    raise NotImplementedError("Connect independently trained M3 implementation")
```

Set `M4_TEAM_PREDICTOR=your_package.inference:predict_trip`. The FastAPI layer will use this predictor instead of its built-in joblib candidate model/decoder. Do not set the adapter to an untrained or dummy function for final evaluation. Keep it importable from the directory from which uvicorn starts.

The backend validates the output length and arc indexes. The backend, not the adapter, loads private `.route` for scoring **after inference**. Teams using a custom decoder should report their actual M3 algorithm and comparison experiments in the final technical report.

## 6. API contract

### GET `/api/health`

Readiness, environment (`trained` / `demo`), available trip count, configured-model flag. Status is `not_ready` when required inputs are missing. It does not inspect private reference labels.

### GET `/api/trips`

```json
{"count": 10, "trips": [{"id": "eval_01"}], "mode": "trained"}
```

Illustrative JSON schema only: the actual IDs and count come from the supplied files.

### POST `/api/predict`

```json
{"trip_id": "eval_01"}
```

Returns the noisy GPS coordinates; one predicted arc per observation; consecutive-deduplicated predicted arc sequence; each predicted arc's `[[latitude, longitude],[latitude, longitude]]` road geometry; directed continuity percentage; and **only aggregate** server-side evaluation metrics. Does not return the reference route/ground-truth arc IDs.

### Example using PowerShell on Laptop 1

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/predict" `
  -Method POST -ContentType "application/json" `
  -Body '{"trip_id":"eval_01"}'
```

Errors are surfaced as `404` for unknown trips, `422` for invalid data/IDs, and `503` for unavailable M3 models. Inference is **never** faked by returning preset responses for evaluator trips.

## 7. Request logs: mandatory during demonstration

Each request creates an `M4 REQUEST` line on Laptop 1's console and an append-only JSON record at `logs/requests.jsonl` with UTC time, request ID, endpoint, status, processing duration, originating client IP and trip ID. GPS positions and reference routes are **not** logged. Uvicorn `--access-log` adds standard HTTP access lines.

To watch the JSON lines in another PowerShell window:

```powershell
Get-Content .\logs\requests.jsonl -Wait -Tail 15
```

During the demo, keep Laptop 1's terminal visible when the evaluator presses **Run map matching** on Laptop 2 to prove a genuine network request reached the API.

## 8. Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check: demo readiness, trip listing, API prediction, safe filenames, model-not-ready errors, server-side scoring, sequence score semantics, all-ten-trip endpoint contract with synthetic fixtures, missing labels, nested dataset layouts, Viterbi output shapes, and joblib model integration with a *toy* classifier. **Test results do not establish real model accuracy.**

## 9. Local synthetic demo (when team M3/evaluator files are not ready)

```powershell
$env:M4_MODE = "demo"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
```

In another terminal:

```powershell
py -3 -m http.server 5500 --directory frontend
```

Open `http://localhost:5500`, enter `http://localhost:8000`, connect and try `demo_01`/`demo_02`. This **only tests network, API, logging, visualisation and sequence scoring wiring**. The geometric demo scorer is not one of the five required trained ML models, and demo metrics must **never** be shown as actual PathSnap experimental results.

## 10. Final evaluator dry-run checklist

- [ ] Your team independently trained and evaluated the M3 model, integrated the exact same feature/preprocessing/HMM logic used in training.
- [ ] `M4_MODE=trained`; `/api/health` shows `ready`.
- [ ] Exactly ten **real** reserved trip IDs appear on Laptop 2.
- [ ] All ten run through the API without hardcoded predictions or missing-file errors.
- [ ] Reference `.route` data is stored only on Laptop 1 and is used only for scoring.
- [ ] Route maps show GPS observations and predicted road segments, with no invented road connections.
- [ ] Computed metric is clearly defined and represents what the supplied reference can support.
- [ ] Laptop 2 communicates with Laptop 1 via its real private-network IP address (not `localhost`).
- [ ] Laptop 1 terminal displays HTTP requests and `M4 REQUEST` entries while demo runs.
- [ ] Capture **real screenshots** of the UI and API console for the technical report.
- [ ] GitHub includes source, dependencies, model-loading code and setup instructions; no evaluator data or secrets.
- [ ] Team can explain the code and disclose AI assistance as required by the academic-integrity policy.

## 11. Troubleshooting

| Problem | Check |
|---|---|
| Laptop 2 cannot connect | Laptop 1 IP, same network, `--host 0.0.0.0`, port 8000 firewall, backend URL |
| Browser reports CORS error | Add exact Laptop 2 UI origin via `M4_ALLOWED_ORIGINS`; restart API |
| `/api/trips` returns no trips | Check `M4_EVAL_DIR`, filenames, and `.track/.nodes/.arcs` trio |
| HTTP 503 prediction | Connect actual M3 artifact or adapter; demo mode is not a substitute |
| Route accuracy missing | `.route` not provided; backend correctly refuses to invent a score |
| Labels differ from GPS points | Expected: `.route` is an **arc sequence**; use proper matching metric/alignment |
| Map background blank | Check external Leaflet library/CDN and OSM tiles; API can still run on LAN |
| Prediction yields discontinuities | M3 sequence decoder must handle road connectivity, gaps and invalid transitions |

**Project scope:** M4 is the evaluator-facing integration milestone, not a substitute for training five models, running Bayesian optimisation, or validating the chosen HMM/Viterbi approach in M1–M3.
