# M4 summary

**Scope:** FastAPI backend (Laptop 1), Leaflet evaluator UI (Laptop 2), server-side scoring, request logging and
two-laptop deployment scripts.

**Prediction pipeline:** the API calls `m3_hmm.inference:predict_trip`, which builds M1's candidate features,
scores candidates with M2's tuned Gradient Boosting and decodes the route with M3's HMM/Viterbi. It returns one
arc ID per GPS observation.

**Scoring:** after inference, the server aligns the reference route to the GPS observations and reports
observation-level Top-1 accuracy, plus route similarity (Levenshtein) and reference coverage. The reference route
never reaches the predictor, the UI or the logs.

**Verification:**
- `pytest` suite: 12 tests pass (see `TEST_RESULTS.txt`).
- Running API on the 10 student-test trips: all 10 predictions succeed and the Top-1 per trip matches
  `m3_hmm/outputs/per_trip_student_test_results.csv`.

**Run:** see `README.md` (Laptop 1 and Laptop 2 sections).
