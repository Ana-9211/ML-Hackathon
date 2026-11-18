# PathSnap: GPS map matching with machine learning and HMM decoding

A vehicle's GPS fixes drift off the road. PathSnap takes a noisy GPS trajectory and the road network and
predicts **which road segment the vehicle was on at every GPS observation**, then reconstructs a connected route.

```
raw trip files ─► M1 candidates + 16 features ─► M2 classifier (tuned Gradient Boosting)
               ─► M3 HMM / Viterbi decoding ─► M4 API (Laptop 1) ─► evaluator UI (Laptop 2)
```

## Results (student-testing partition, 10 trips, 19,160 GPS observations)

| Stage | Observation-level Top-1 | Point continuity |
|---|---|---|
| Nearest road only (no ML) | 55.5 % | — |
| Classifier only (tuned Gradient Boosting) | 88.2 % | 93.0 % |
| **Classifier + HMM/Viterbi (final system)** | **93.1 %** | **98.5 %** |

Point continuity = share of consecutive GPS observations on the same or a directly connected road (mean over trips).

Six classifiers were compared with 5-fold GroupKFold cross-validation on the 80 training trips
(`m2_models/results/`); the HMM ablations are in `m3_hmm/outputs/`.

## Repository layout

| Folder | Owner | Contents |
|---|---|---|
| `m1_data/` | M1 | Data loading and cleaning, per-trip projection, road network, label alignment, candidate generation, feature engineering, EDA, Folium maps |
| `m2_models/` | M2 | Six classifiers, GroupKFold CV, student-test evaluation, Optuna Bayesian optimisation, comparison plots, saved models |
| `m3_hmm/` | M3 | HMM transition models, Viterbi decoder, decoder tuning, classifier-vs-HMM evaluation, Folium route maps, `inference.predict_trip` |
| `m4_app/` | M4 | FastAPI backend, Leaflet UI, server-side scoring, request logs, two-laptop scripts, tests |
| `data/` | — | Dataset location (git-ignored, see `data/README.md`) |

Each folder has its own README with details.

## Installation

Python 3.10–3.14. From the repository root:

```bash
python -m pip install -r requirements.txt        # pipeline (m1, m2, m3)
python -m pip install -r m4_app/requirements.txt # + API (Laptop 1)
```

`scikit-learn` is pinned to **1.7.2**: the saved models in `m2_models/models/` were trained with it and the
Gradient Boosting pickles do not load on other versions.

Place the dataset as `data/train/<id>/`, `data/student_test/<id>/` and `data/evaluator_test/<id>/`
(each with `<id>.track`, `.nodes`, `.arcs`, `.route`).

## Reproducing the results

Run from the repository root, in this order:

```bash
python -m m1_data.build_dataset        # candidate + feature tables  (~1 min)
python -m m1_data.eda                  # EDA statistics, plots, Folium maps
python -m m1_data.pipeline_figure      # preprocessing pipeline figure
python -m m2_models.cross_validation   # 5-fold GroupKFold CV, 6 models
python -m m2_models.train_models       # train on 80 trips, save models
python -m m2_models.evaluate_models    # student-test evaluation
python -m m2_models.optuna_optimization  # Bayesian optimisation (30 trials)
python -m m2_models.compare_models     # comparison plots
python -m m3_hmm.tune_decoder          # HMM decoder tuning (training trips only)
python -m m3_hmm.evaluate_hmm          # classifier-only vs HMM comparison
python -m m3_hmm.visualize_folium      # predicted-vs-reference route maps
```

## Deployment (two laptops)

**Laptop 1 (API):**
```powershell
python -m pip install -r m4_app/requirements.txt
powershell -ExecutionPolicy Bypass -File .\m4_app\scripts\start_laptop1.ps1
```
It serves trips from `data/evaluator_test` on port 8000 (allow port 8000 in the Windows firewall; find the IP with `ipconfig`).

**Laptop 2 (UI):**
```powershell
powershell -ExecutionPolicy Bypass -File .\m4_app\scripts\start_laptop2.ps1
```
Open `http://localhost:5500`, enter `http://<LAPTOP1_IPV4>:8000`, connect, pick a trip and run map matching.
Full instructions: `m4_app/README.md`.

## Data handling rules followed

- Evaluator-testing trips are used only by the deployed API; they were never used for training, tuning or analysis.
- Cross-validation is grouped by trip, so no trip appears in both training and validation folds.
- Reference routes are read only for labelling (training) and for server-side scoring after inference.
