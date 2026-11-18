# PathSnap — M4 Milestone Submission Summary

**Submission folder:** `m4_app/`

**Milestone:** M4 — API, UI and two-laptop demonstration

**Implemented:**

- FastAPI backend for trip discovery, health checks and route predictions (`backend/main.py`).
- HTML/Leaflet evaluator interface, dynamically listing available trips and making real API requests (`frontend/`).
- Input-only `.track`, `.nodes`, `.arcs` loading and M3 inference interface (`backend/dataset.py`, `backend/pipeline.py`).
- Server-only `.route` loading and an explicitly defined route-sequence evaluation score (`backend/scoring.py`).
- HTTP request logs on the API laptop, plus persistent JSONL logging (`logs/requests.jsonl`).
- Windows PowerShell run scripts for both laptops (`scripts/`), dependencies, deployment instructions and automated tests.

**Locally verified:** Automated M4 contract tests pass using synthetic inputs; HTTP requests to FastAPI and the independent UI web server return successful responses. The synthetic fixture is **not** real evaluator data or proof of ML performance.

**Required before the real two-laptop assessment:**

1. Put the 10 actual reserved evaluation trips on Laptop 1 only. Those files were not supplied with the materials shared in this conversation.
2. Connect the actual independently trained M3 candidate classifier or team HMM/Viterbi predictor with identical training-time features and preprocessing. No trained M3 model was supplied.
3. Perform the official real-data evaluator run with Laptop 1 API, Laptop 2 UI, visible incoming HTTP logs and screenshots.
4. Upload this folder (with approved model artifacts and your team's remaining milestone code) to the required public GitHub repository. Do **not** include evaluator data, credentials or manufactured experimental results.

**Scoring note:** The provided dataset guide describes `.route` as a sequential list of traversed road arc IDs, **not GPS-point labels**. Built-in scoring compares arc sequences via normalized edit distance. It must not be represented as GPS-point classification accuracy. Replace the scorer with a valid label alignment when the assessment provides one.

**How to run:** See `README.md`, sections 3–4 and 9. For a synthetic integration smoke test, start Laptop 1 using `M4_MODE=demo` and Laptop 2 normally. For the actual evaluation, use `M4_MODE=trained` and the real M3/evaluator assets.
