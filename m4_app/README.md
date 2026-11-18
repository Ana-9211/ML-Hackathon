# M4: API, evaluator UI and two-laptop deployment

Laptop 1 runs a FastAPI backend that lists the evaluator trips, predicts one road segment per GPS
observation with the team pipeline, scores the prediction against the reference route **on the
server**, and logs every request. Laptop 2 runs a static Leaflet web page that calls the API and
draws the GPS points and the predicted route.

```
Laptop 2 (browser UI)  --HTTP-->  Laptop 1 (FastAPI :8000)
                                    ├─ load trip (.track .nodes .arcs; never .route)
                                    ├─ predict: m3_hmm.inference:predict_trip
                                    │     M1 features -> M2 tuned Gradient Boosting -> M3 HMM/Viterbi
                                    └─ score after inference (reads .route, server-side only)
```

## Folder layout

```
m4_app/
├── backend/
│   ├── main.py        # FastAPI app: /api/health, /api/trips, /api/predict, request logging
│   ├── config.py      # settings from environment variables (defaults below)
│   ├── dataset.py     # input-only trip loader + validation (never reads .route)
│   ├── pipeline.py    # calls the team predictor; demo geometry baseline; route geometry
│   └── scoring.py     # server-side scoring: Top-1, route similarity, coverage
├── frontend/          # index.html, app.js, styles.css (Leaflet UI for Laptop 2)
├── scripts/           # start_laptop1.ps1, start_laptop2.ps1
├── tests/             # pytest suite
├── demo_data/         # two tiny synthetic trips for wiring checks only
├── requirements.txt   # API deps + shared pinned project requirements
└── .env.example       # environment variables reference
```

## Laptop 1: API

1. Clone the repository and install (from the repository root):
   ```powershell
   git clone https://github.com/Ana-9211/ML-Hackathon.git
   cd ML-Hackathon
   python -m pip install -r m4_app/requirements.txt
   ```
   This installs **scikit-learn 1.7.2 exactly**: the saved models in `m2_models/models/` only load on that version.
2. Put the evaluator trips at `data/evaluator_test/<trip_id>/<trip_id>.track|.nodes|.arcs|.route`
   (repository-root `data/` is git-ignored; never commit it).
3. Start the API:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\m4_app\scripts\start_laptop1.ps1
   ```
   Options: `-EvalDir <folder>` to read trips from another folder, `-Mode demo` for the synthetic wiring check.
   The script prints the mode, predictor and trip folder, then serves on `0.0.0.0:8000`.
4. Allow port 8000 through the Windows firewall once (PowerShell as Administrator):
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
5. Find Laptop 1's IPv4 address with `ipconfig` (Wi-Fi adapter).

## Laptop 2: UI

```powershell
git clone https://github.com/Ana-9211/ML-Hackathon.git
cd ML-Hackathon
powershell -ExecutionPolicy Bypass -File .\m4_app\scripts\start_laptop2.ps1
```
Open `http://localhost:5500`, enter `http://<LAPTOP1_IPV4>:8000`, click **Connect and load trips**,
choose a trip and run map matching. Both laptops must be on the same network. Laptop 2 needs only Python
(for the static file server) and internet access for the map tiles.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | mode, readiness, number of trips found |
| `GET /api/trips` | trip IDs only |
| `POST /api/predict` `{"trip_id": "..."}` | GPS points, predicted arc per observation, predicted route geometry, point continuity, and the server-side evaluation |

`evaluation` contains:

| Field | Meaning |
|---|---|
| `top1_accuracy_percent` | **Observation-level Top-1 accuracy** (the project's primary metric): share of GPS points whose predicted arc equals the reference arc aligned to that point by `m1_data.labels.align_route` |
| `route_accuracy_percent` | Route similarity: `100 × (1 − Levenshtein(predicted, reference) / max length)` on the ordered arc sequences |
| `reference_arc_coverage_percent` | Share of reference arcs that appear in the predicted route |

`predicted_route.continuity_percent` is point continuity: the share of consecutive GPS observations whose
predicted arcs are the same arc or directly connected (same definition as `m3_hmm/metrics.py`).

The reference route is read only by `backend/scoring.py`, after inference has finished, and only counts and
percentages are returned: no reference arc IDs reach the predictor, the UI or the logs.

## Request logs

Every request prints one `M4 REQUEST {...}` line in Laptop 1's console (time, method, path, status, duration,
client IP, trip ID) and is appended to `m4_app/logs/requests.jsonl`. Show this console during evaluation.

## Environment variables

Set by `start_laptop1.ps1`; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `M4_MODE` | `trained` | `trained` = team pipeline; `demo` = synthetic trips + geometry baseline |
| `M4_TEAM_PREDICTOR` | `m3_hmm.inference:predict_trip` | prediction function (`module:function`) |
| `M4_EVAL_DIR` | `<repo>/data/evaluator_test` | folder of evaluator trips |
| `M4_ALLOWED_ORIGINS` | `http://localhost:5500,http://127.0.0.1:5500` | CORS origins of the UI |
| `M4_MAX_POINTS` / `M4_MAX_ARCS` | `50000` / `1000000` | input size limits (training networks reach 307,392 arcs) |
| `M4_LOG_PATH` | `logs/requests.jsonl` | request log file |

## Tests

```powershell
cd m4_app
python -m pip install pytest httpx
python -m pytest -q tests
```

## Verified results (student-test partition through the running API)

Running the API with `-EvalDir data\student_test` reproduces the M3 per-trip results
(`m3_hmm/outputs/per_trip_student_test_results.csv`), for example trip 00000052: Top-1 98.85 %,
trip 00000077: Top-1 83.87 %. Each prediction takes 0.2–2.5 s on a laptop CPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI cannot connect | Same Wi-Fi? Firewall rule added? Use Laptop 1's IPv4, not `localhost` |
| `No trips found` | Check `data/evaluator_test/<id>/<id>.track` exists, or pass `-EvalDir` |
| HTTP 503 `Could not import M3 predictor` | Run from the repository clone (the script sets `PYTHONPATH`); reinstall requirements |
| Model load error mentioning `_loss` or version warnings | Install `scikit-learn==1.7.2` exactly |
| Map is grey | Laptop 2 needs internet access for map tiles; routes and metrics still display |
