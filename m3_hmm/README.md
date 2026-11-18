# M3 — HMM and Viterbi Sequential Route Reconstruction

Owner: **M3**

## Overview
This module implements the sequential route reconstruction stage for **PathSnap**, converting noisy pointwise candidate probabilities into a physically coherent, connected vehicle trajectory across the road network.

### Key Innovations
1. **3-Technique HMM Transition Ensemble:**
   Instead of relying on a single heuristic transition rule, `m3_hmm` implements three complementary transition probability formulations and combines them in log-space inside a single Viterbi pass:
   - **Technique 1: Topological Graph Step-Cost (`log_A_topo`)** — Enforces directed road-network legality (`0-hop` same arc forward vs. backward jitter, `1-hop` direct connection `arc_to == next_arc_from`, `2-hop` / `3-hop` shortest-path bridges, U-turn penalty, and disconnected floor penalty).
   - **Technique 2: Kinematic Distance-Consistency (`log_A_kin`, Newson & Krumm)** — Scores physical speed/distance realism via exponential decay $\exp(-|d_{\text{road}} - d_{\text{GPS}}| / (\beta \cdot \Delta t))$, using exact projection coordinates (`proj_t`, `arc_len`) and intermediate bridge lengths.
   - **Technique 3: Data-Driven Empirical Transition Priors (`log_A_emp`)** — Learns an empirical lookup table $P(\text{relation} \mid \Delta t\text{ bucket}, \text{speed bucket})$ directly from the 80 training trips' ground-truth sequences.
   - **Soft Log-Linear Ensemble:** Combines $w_{\text{topo}} \log A_{\text{topo}} + w_{\text{kin}} \log A_{\text{kin}} + w_{\text{emp}} \log A_{\text{emp}}$ inside a single Viterbi pass (`< 30ms` per trip), and also supports soft-voting across the top 3 ML classifiers (`RandomForest + HistGradientBoosting + ExtraTrees`) for emission probabilities.

2. **GPS Dropout & Gap Handling:**
   - **Medium gaps ($2.5\text{s} < \Delta t < 30\text{s}$):** Scales kinematic tolerance $\beta \cdot \Delta t$ and relaxes multi-hop topological penalties so vehicles can naturally traverse intermediate road segments during brief signal dropouts.
   - **Large dropouts ($\Delta t \ge 30\text{s}$):** Resets transition penalties to zero so trajectories cleanly re-anchor after long tunnels or parking stops.

3. **Shortest-Path Bridge Route Reconstruction:**
   - Keeps length-$T$ observation-level predictions intact for pointwise evaluation (`Top-1 Accuracy`, `Macro F1`).
   - Collapses consecutive duplicate arcs and inserts shortest-path bridge arcs (up to 3 hops) when reconstructing the final `.route` sequence, achieving near-100% road-network continuity.

---

## Quick Start (Run from Repo Root)

```bash
# 1. Tune HMM decoder parameters (Optuna) + evaluate all 7 methods on student_test + generate plots:
python -m m3_hmm.evaluate_hmm

# 2. Generate 5-layer interactive Folium HTML maps for all 10 student_test trips:
python -m m3_hmm.visualize_folium

# 3. Connect to M4's FastAPI Backend on Laptop 1:
export M4_TEAM_PREDICTOR="m3_hmm.inference:predict_trip"
```

---

## Module Structure
```text
m3_hmm/
├── __init__.py
├── transitions.py          # TripGraph (0..3-hop index) + Topological, Kinematic, Empirical & Ensemble models
├── viterbi.py              # Log-space Viterbi decoder, Majority-Vote baseline, and shortest-path route bridging
├── metrics.py              # Pointwise Top-1, Macro F1, Route Continuity %, Levenshtein Sim %, and Coverage %
├── model_bridge.py         # Bridges M1 feature tables and M2 models/predictions
├── tune_decoder.py         # Optuna Bayesian tuning of HMM & ensemble weights on training OOF predictions
├── evaluate_hmm.py         # Full ablation comparison across student_test trips + diagnostic plots
├── visualize_folium.py     # 5-Layer diagnostic Folium map generator (using m1_data.maps)
├── inference.py            # Live inference adapter for M4 (`m3_hmm.inference:predict_trip`)
├── artifacts/
│   ├── hmm_config.json                     # Tuned HMM & ensemble parameters
│   ├── empirical_priors.npy                # Learned empirical transition matrix
│   └── m3_classifier_bundle.joblib         # Cached classifier bundle for standalone/live inference
├── outputs/
│   ├── hmm_ablation_comparison.csv         # Summary table comparing all 7 decoding methods
│   ├── per_trip_student_test_results.csv   # Detailed metrics for every student_test trip
│   ├── error_analysis_summary.csv          # Fixed-by-HMM vs. hurt-by-HMM diagnostic breakdown
│   ├── decoder_tuning_trials.csv           # Optuna tuning trial history
│   └── plots/                              # Comparison bar charts & tuning curves for Section 5
└── maps/
    └── trip_<id>_hmm_comparison.html       # Interactive 5-layer Folium maps for Section 6
```
