# M1 — Data processing, labels, candidates, features, EDA

## Quick start (everyone runs this once)

```bash
pip install -r m1_data/requirements.txt
# put the dataset in data/train and data/student_test  (see data/README.md)
python -m m1_data.build_dataset      # ~1 min -> m1_data/outputs/features/*.parquet
python -m m1_data.eda                # ~1 min -> m1_data/outputs/eda/ (plots, Folium maps, stats)
```

Always run from the **repository root** with `python -m ...` so `import m1_data` works.

## What M2 / M3 / M4 can use

| Need | Use |
|---|---|
| Training / test table | `pd.read_parquet("m1_data/outputs/features/features_train.parquet")` (and `features_student_test.parquet`) |
| Feature column list | `from m1_data.features import FEATURE_COLUMNS` |
| Load a trip **without** labels (inference, API) | `from m1_data.loader import load_trip, list_trips` → `load_trip("evaluator_test", "00000012")` |
| Load the reference route (training / scoring only) | `from m1_data.loader import load_route` |
| Build the candidate + feature table for any trip (API) | `from m1_data.features import build_trip_table` → `df, stats = build_trip_table(trip)` (no route → no label columns) |
| Road network (lengths, adjacency, geometry) | `from m1_data.network import RoadNetwork` → `net = RoadNetwork(trip)` |
| Per-point labels for scoring | `from m1_data.labels import align_route` |
| Folium maps in the same style | `from m1_data import maps` → `trip_map`, `add_route`, `add_legend`, `finish` |

## Feature table: one row per (GPS point, candidate arc)

**Keys / bookkeeping (not model inputs)**

| column | meaning |
|---|---|
| `trip_id` | trip id (use as the **group** for GroupKFold) |
| `point_idx` | index of the GPS point within the cleaned track (0..N-1) |
| `time` | GPS timestamp (s) |
| `arc_id` | candidate arc = row index in `.arcs` (same ids as `.route`) |
| `arc_from`, `arc_to` | node ids of the arc (for continuity checks / HMM transitions) |
| `proj_t` | where the GPS point projects onto the arc, 0 = start, 1 = end |

**Model features** (`FEATURE_COLUMNS`)

| column | meaning |
|---|---|
| `dist` | perpendicular distance point → arc (m) |
| `dist_rank` | 0 = nearest candidate (ties share a rank) |
| `dist_gap` | `dist` − nearest candidate's distance (m) |
| `dist_ratio` | (`dist`+1)/(nearest+1) |
| `heading_diff` | angle between direction of travel and arc direction, 0–180° (90 if heading unknown) |
| `heading_cos` | cos(`heading_diff`), 0 if heading unknown |
| `heading_valid` | 1 if the car moved ≥ 3 m over ±2 s, so heading is trustworthy |
| `speed` | m/s from points i−2 → i+2 |
| `dist_prev`, `dist_next` | distance of the same arc to the GPS points 3 s before / after |
| `overshoot` | how far (m) the point projects beyond the arc's ends |
| `arc_len` | arc length (m) |
| `next_choices` | number of arcs leaving the arc's end node |
| `two_way` | 1 if the reverse arc exists |
| `n_cand` | candidates at this point |
| `fallback` | 1 if no arc within 50 m and the 200 m fallback radius was used |

**Labels** (only in train / student_test tables — never at inference)

| column | meaning |
|---|---|
| `label` | 1 if this candidate is the point's aligned reference arc |
| `label_arc` | the aligned reference arc for the point |
| `label_dist` | distance from the point to that arc (m) |
| `label_reliable` | 1 if `label_dist` ≤ 50 m. **Recommended: train on `label_reliable == 1` only** (see data-quality notes) |

## Key design decisions

- **Per-trip local projection.** Every trip is in a different city, so distances use a local equirectangular projection centred on the trip (error < 1 % at trip scale, far below GPS noise). See `geo.py`.
- **Labels by monotonic alignment.** `.route` has no timestamps, so each GPS point is matched to a route arc with a dynamic-programming alignment that can only move forward along the route (handles loops and U-turns). See `labels.py`.
- **Candidates.** Arcs within 50 m, at most 12 per point, 200 m fallback if none. Both directions of a two-way street are separate candidates with identical geometry, so heading is the only feature that separates them.
- **Separation of labels.** `load_trip` never reads `.route`; only `load_route` does.

## Data-quality notes (from `outputs/eda/`)

- 80 train + 10 student-test trips, all 4 files present for every trip, no invalid or duplicate GPS rows.
- Sampling is ~1 Hz, but there are gaps, up to ~42 h inside trip 90.
- Road networks range from 649 to 307,392 arcs; ~95 % of arcs are two-way pairs.
- 11 reference routes contain at least one break (consecutive arcs not connected), so the reference is not perfect ground truth.
- Trips 54 and 33 end with long slow (~2 m/s) stretches 120–140 m from any road (parked or off-network), and trip 2 has an off-road section. ~11 % of training points therefore have `label_reliable = 0`. Among reliable points the candidate miss rate is ~0.6 %.
