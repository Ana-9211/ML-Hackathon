# M3 — HMM and Viterbi Sequential Route Reconstruction

Owner: **M3**

## Overview
This module implements the sequential route reconstruction stage for **PathSnap**, decoding one road segment (`arc_id`) per GPS observation with a log-space Viterbi Hidden Markov Model (HMM).

- **Emission Model:** Uses candidate probabilities from M2's final selected classifier (`m2_models/models/tuned_gradient_boosting.pkl`, trained on raw 16-feature inputs with hyperparameters from `m2_models/results/best_parameters.json`).
- **Transition Model:** Uses a weighted log-linear blend of three transition probability formulations (topological graph step cost, kinematic Newson-Krumm distance decay, and an empirical transition matrix learned from reliable training alignments), tuned with Optuna on 20 training trips using 5-fold `GroupKFold` out-of-fold probabilities.
- **Performance on `student_test` (10 trips, 19,160 GPS observations):**
  - Observation Top-1 Accuracy (All points): **88.18%** (Classifier-Only) $\rightarrow$ **93.13%** (HMM Ensemble)
  - Observation Top-1 Accuracy (Reliable points, weighted): **88.31%** $\rightarrow$ **93.26%** (**94.00%** trip mean)
  - Reconstructed Route Continuity: **75.05%** $\rightarrow$ **99.49%**
  - Sequence Levenshtein Similarity: **59.39%** $\rightarrow$ **93.38%**

---

## Technical Documentation

### 1. Hidden States
At each GPS observation step $t \in \{0, 1, \dots, T-1\}$, the hidden state $S_t$ represents the directed road segment (`arc_id`) traversed by the vehicle at timestamp $\text{time}_t$. The state space at step $t$ is restricted to the $K_t \le 12$ candidate road segments $\{c_{t,1}, \dots, c_{t,K_t}\}$ generated within $50\text{ m}$ (or $200\text{ m}$ fallback radius), keeping Viterbi complexity at $O(T \cdot K^2)$ ($\sim 40.8\text{ ms}$ per trip). Each candidate retains its 0-based `arc_id`, start node `arc_from`, end node `arc_to`, and projection fraction $\text{proj\_t} \in [0, 1]$.

### 2. Emission Calculation
For candidate $c_{t,k}$ at step $t$, M2's tuned Gradient Boosting model (`tuned_gradient_boosting.pkl`) predicts $P_{\text{ML}}(y_{t,k} = 1 \mid \mathbf{x}_{t,k})$. Probabilities are clipped at $\epsilon = 10^{-4}$, normalized across the $K_t$ candidates at step $t$, and scaled by `emission_weight` ($\alpha_{\text{em}} = 0.648$):
$$\log E_t(k) = \alpha_{\text{em}} \cdot \log \left( \frac{\max(P_{\text{ML}}(c_{t,k}), \epsilon)}{\sum_{m=1}^{K_t} \max(P_{\text{ML}}(c_{t,m}), \epsilon)} \right)$$

### 3. The Three Transition Techniques and Tuned Weights
For candidate transition $(a_i \to a_j)$ over time step $\Delta t = \text{time}_t - \text{time}_{t-1}$, three log-transition matrices are combined inside Viterbi:
$$\log A_{\text{ensemble}}(a_i \to a_j \mid \Delta t) = w_{\text{topo}} \log A_{\text{topo}} + w_{\text{kin}} \log A_{\text{kin}} + w_{\text{emp}} \log A_{\text{emp}}$$
with Optuna-tuned weights (`m3_hmm/artifacts/hmm_config.json`):
- **$w_{\text{topo}} = 0.389$ (Technique 1 — Topological Graph Step-Cost):** Uses `TripGraph` adjacency to assign step log-costs:
  - Same arc forward ($\Delta s \ge -5.0\text{ m}$): `cost_same_fwd = 0.0`
  - Same arc backward ($\Delta s < -5.0\text{ m}$): `cost_same_bwd = -2.2`
  - Direct 1-hop connection (`arc_to[a_i] == arc_from[a_j]`): `cost_hop1 = -0.35`
  - 2-hop connection (1 intermediate arc): `cost_hop2 = -1.6` (scaled by $\text{gap\_factor}$ when $\Delta t > 2.5\text{ s}$)
  - 3-hop connection (2 intermediate arcs): `cost_hop3 = -2.9` (scaled by $\text{gap\_factor}$ when $\Delta t > 2.5\text{ s}$)
  - Immediate reverse-arc U-turn: `cost_uturn = -4.2`
  - Disconnected beyond 3 hops: `cost_disconnected = -9.5`
- **$w_{\text{kin}} = 0.346$ (Technique 2 — Kinematic Distance-Consistency, Newson & Krumm):** Penalizes mismatch between road-network path distance $d_{\text{road}}$ (from `proj_t`, `arc_len`, and shortest bridge length) and Euclidean GPS step displacement $d_{\text{GPS}}$:
  $$\log A_{\text{kin}}(a_i \to a_j \mid \Delta t) = -\min\left(10.0,\, \frac{|d_{\text{road}} - d_{\text{GPS}}|}{\beta \cdot \max(1.0, \Delta t)}\right), \quad \beta = 18.0\text{ m/s}$$
- **$w_{\text{emp}} = 0.265$ (Technique 3 — Empirical Transition Matrix):** Lookup table $\log P(\text{relation} \mid \Delta t\text{ bucket}, \text{speed bucket})$ fitted on reliable (`label_reliable == 1`) ground-truth transitions across the 80 training trips (`m3_hmm/artifacts/empirical_priors.npy`).

### 4. GPS Gap Handling
- **Medium gaps ($\Delta t > \text{gap\_relax\_sec} = 2.5\text{ s}$):** Multi-hop topological costs (`cost_hop2`, `cost_hop3`) are multiplied by $\max(0.35, 2.5 / \Delta t)$ and the kinematic scale expands as $\beta \cdot \max(1.0, \Delta t)$ so multi-segment traversal during brief signal loss is not over-penalized.
- **Large gaps ($\Delta t \ge \text{gap\_reset\_sec} = 30.0\text{ s}$):** All three transition matrices reset to $\mathbf{0}$ so the decoder starts an unconstrained trajectory segment after long signal dropouts.

### 5. Disconnected-Segment Handling
- **During Viterbi Decoding:** Candidate pairs with no path within 3 hops receive `cost_disconnected = -9.5` in $\log A_{\text{topo}}$, `-10.0` in $\log A_{\text{kin}}$, and the disconnected bin probability in $\log A_{\text{emp}}$.
- **During Route Reconstruction (`reconstruct_connected_route`):** After collapsing consecutive duplicate `arc_id`s from the length-$T$ Viterbi path, if two consecutive distinct arcs $(a_{\text{prev}}, a_{\text{curr}})$ with $\Delta t < 30.0\text{ s}$ are 2 or 3 hops apart in the road graph, `TripGraph.find_bridge` inserts the shortest connecting intermediate `arc_id`(s) (`max_bridge_hops = 3`), raising reconstructed route continuity from **75.05%** to **99.49%**.

### 6. Where HMM Decoding Helped vs. Where It Did Not Help
From `m3_hmm/outputs/error_analysis_summary.csv` across the 10 `student_test` trips (`19,160` total points, `19,133` reliable points):
- **Overall Impact:** HMM corrected **1,331** classifier errors while introducing **383** errors (` hurt_by_hmm = 383`), yielding a net gain of **+948** correct observations and improving both Top-1 accuracy and route continuity on all 10 trips.
- **Where HMM Did Not Help (Failure Modes & Worst Trips):**
  1. **Dense urban intersections and multi-second drift (Trip `00000077` and Trip `00000056`):** Trip `00000077` had the lowest final Top-1 accuracy (**83.87%**, up from **74.38%** classifier-only), with `578` points fixed but `207` points hurt (`424` points wrong in both). Trip `00000056` had `249` points fixed and `67` points hurt (`95.00%` all-point Top-1). When noisy GPS observations drift near complex junctions over several consecutive seconds, transition smoothing can carry a wrong branch choice across adjacent timestamps or delay a turn by 1–2 seconds.
  2. **Parallel connected roads (Trip `00000065` at 88.28% and Trip `00000009` at 90.85%):** On Trip `00000065` (`35` fixed, `12` hurt, `78` wrong in both) and Trip `00000009` (`56` fixed, `22` hurt, `74` wrong in both), parallel connected road segments run within the candidate radius. Because both parallel paths are topologically connected (`0-hop`/`1-hop`) and kinematically consistent, HMM transition constraints cannot override sustained GPS bias toward the parallel street.

---

## Summary Results (`student_test`)

| Method | Reliable Top-1 (Weighted %) | Reliable Top-1 (Trip Mean %) | All-Point Top-1 (%) | Candidate Macro F1 | Pointwise Continuity (%) | Route Continuity (%) | Levenshtein Sim (%) | Route Coverage (%) | Route F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only (No HMM)** | 88.31% | 89.66% | 88.18% | 0.9407 | 92.96% | 75.05% | 59.39% | 89.33% | 88.31% |
| **HMM 1: Topological Step-Cost** | 91.69% | 92.74% | 91.56% | 0.9580 | 97.60% | 99.72% | 86.92% | 98.15% | 95.74% |
| **HMM 2: Kinematic Distance-Decay** | 92.07% | 92.40% | 91.94% | 0.9554 | 97.69% | **99.74%** | 88.93% | **98.30%** | 96.33% |
| **HMM 3: Empirical Transition Matrix** | 91.59% | 91.70% | 91.46% | 0.9534 | **99.08%** | 96.09% | 86.89% | 88.63% | 91.92% |
| **HMM Majority Vote (Output Voting 1+2+3)** | 92.45% | 93.09% | 92.32% | 0.9600 | 97.76% | 99.66% | 88.67% | 97.97% | 96.14% |
| **HMM Ensemble (3-Transition Soft Average)** | **93.26%** | **94.00%** | **93.13%** | **0.9662** | 98.53% | 99.49% | 93.38% | 96.46% | 96.38% |
| **HMM Full Ensemble (3-Trans + 3-Classifier)** | 92.71% | 93.61% | 92.58% | 0.9641 | 98.55% | 99.54% | **93.40%** | 96.56% | **96.39%** |

---

## Reproducing From Repo Root

```bash
pip install -r m3_hmm/requirements.txt
python -m m3_hmm.tune_decoder
python -m m3_hmm.evaluate_hmm
python -m m3_hmm.visualize_folium
```
