# M4 — API, UI and deployment

Owner: M4

Scope:
- `api/`: FastAPI backend on Laptop 1 (lists the 10 evaluator trips, runs the full prediction pipeline, scores against the reference route server-side, logs every request)
- `ui/`: evaluator interface on Laptop 2 (trip list, map of GPS + predicted route, results)
- Two-laptop deployment instructions

Rule: ground-truth routes stay on the server and are never sent to the UI or used by the prediction code. `m1_data.loader.load_trip` loads a trip without its route; `load_route` is separate and only for scoring.
