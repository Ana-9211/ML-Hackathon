# Section 5 & 6 Technical Report Content (M3: HMM & Viterbi Route Reconstruction)

---

## 5. HMM and Viterbi Route Reconstruction

### 1. What are the hidden states in your implementation?
At each GPS observation step $t \in \{0, 1, \dots, T-1\}$, the hidden state $S_t$ represents the directed road segment (`arc_id`) on which the vehicle is travelling at timestamp $\text{time}_t$.
To maintain $O(T \cdot K^2)$ decoding complexity ($\sim 40.8\text{ ms}$ per trip), the state space at timestamp $t$ is restricted to the $K_t \le 12$ candidate road segments $\{c_{t,1}, c_{t,2}, \dots, c_{t,K_t}\}$ retrieved within the $50\text{ m}$ spatial search radius (or $200\text{ m}$ fallback radius). Every candidate preserves its 0-based `arc_id`, `arc_from`, `arc_to`, and longitudinal projection fraction $\text{proj\_t} \in [0, 1]$.

---

### 2. How are emission probabilities calculated?
For each candidate $c_{t,k}$ at step $t$, M2's final tuned Gradient Boosting classifier (`m2_models/models/tuned_gradient_boosting.pkl`, trained on raw 16-feature inputs) predicts the candidate probability $P_{\text{ML}}(y_{t,k} = 1 \mid \mathbf{x}_{t,k})$.
Raw probabilities are clipped at $\epsilon = 10^{-4}$, normalized across the $K_t$ candidates at step $t$, and scaled by $\alpha_{\text{em}} = 0.648$:
$$\log E_t(k) = \alpha_{\text{em}} \cdot \log \left( \frac{\max(P_{\text{ML}}(c_{t,k}), \epsilon)}{\sum_{m=1}^{K_t} \max(P_{\text{ML}}(c_{t,m}), \epsilon)} \right)$$
An ablation configuration (`HMM Full Ensemble`) also evaluates a 3-classifier emission blend ($0.40 P_{\text{RF}} + 0.40 P_{\text{HGB}} + 0.20 P_{\text{MLP}}$).

---

### 3. How are transition probabilities calculated?
Between candidate $a_i = (u_i \to v_i)$ at step $t-1$ and candidate $a_j = (u_j \to v_j)$ at step $t$ (with $\Delta t = \text{time}_t - \text{time}_{t-1}$), three transition models are combined in log-space inside Viterbi:

1. **Technique 1 — Topological Graph Step-Cost ($\log A_{\text{topo}}$):**
   Classifies each candidate transition $(a_i \to a_j)$ into 7 directed graph relations:
   - Same arc forward ($a_i = a_j, \Delta s \ge -5.0\text{ m}$): $0.0$
   - Same arc backward ($a_i = a_j, \Delta s < -5.0\text{ m}$): $-2.20$
   - Direct 1-hop connection ($v_i = u_j$, non-reverse): $-0.35$
   - 2-hop shortest bridge ($a_i \to b_1 \to a_j$): $-1.60 \times \text{gap\_factor}(\Delta t)$
   - 3-hop shortest bridge ($a_i \to b_1 \to b_2 \to a_j$): $-2.90 \times \text{gap\_factor}(\Delta t)$
   - Immediate U-turn ($v_i = u_j \land u_i = v_j$): $-4.20$
   - Disconnected (no path within 3 hops): $-9.50$

2. **Technique 2 — Kinematic Distance-Consistency ($\log A_{\text{kin}}$, Newson & Krumm):**
   Compares road-network travel distance $d_{\text{road}}$ (from `(1 - proj_t[t-1]) * len(a_i) + len(bridge) + proj_t[t] * len(a_j)`) against Euclidean GPS displacement $d_{\text{GPS}}$:
   $$\log A_{\text{kin}}(a_i \to a_j \mid \Delta t) = -\min\left(10.0,\, \frac{|d_{\text{road}} - d_{\text{GPS}}|}{\beta \cdot \max(1.0, \Delta t)}\right), \quad \beta = 18.0\text{ m/s}$$

3. **Technique 3 — Empirical Transition Priors ($\log A_{\text{emp}}$):**
   Estimated from the 80 training trips filtered to reliable ground-truth alignments (`label == 1` and `label_reliable == 1`) conditioned on $(\Delta t\text{ bucket}, \text{speed bucket})$ with Laplace smoothing.

4. **Tuned Soft Log-Linear Transition Ensemble:**
   Optimized via 40 trials of Optuna TPE on 20 training trips using 5-fold `GroupKFold` out-of-fold probabilities (`m3_hmm/artifacts/hmm_config.json`):
   $$\log A_{\text{ensemble}}(a_i \to a_j \mid \Delta t) = 0.389 \log A_{\text{topo}} + 0.346 \log A_{\text{kin}} + 0.265 \log A_{\text{emp}}$$

---

### 4. How are missing candidates or large GPS gaps handled?
- **Missing Candidates:** During candidate generation (`m1_data/candidates.py`), if no road segment lies within $50\text{ m}$, the search radius expands to $200\text{ m}$ (`fallback = 1`). In `m3_hmm/inference.py`, any observation with zero candidates even at $200\text{ m}$ is filled from the nearest valid Viterbi state.
- **Medium GPS Gaps ($2.5\text{ s} < \Delta t < 30.0\text{ s}$):** Multi-hop topological penalties are scaled by $\text{gap\_factor} = \max(0.35, 2.5/\Delta t)$ and the kinematic denominator scales as $\beta \cdot \max(1.0, \Delta t)$.
- **Large GPS Dropouts ($\Delta t \ge 30.0\text{ s}$):** Transition matrices reset to $\mathbf{0}$ so the decoder starts a fresh trajectory segment after long signal dropouts without artificial transition penalties.

---

### 5. What happens when consecutive GPS observations suggest disconnected road segments?
1. **During Viterbi Decoding:** Disconnected candidate transitions receive `cost_disconnected = -9.5` in $\log A_{\text{topo}}$, `-10.0` in $\log A_{\text{kin}}$, and low empirical probability in $\log A_{\text{emp}}$, preventing isolated GPS spikes from snapping to disconnected cross-streets.
2. **During Route Reconstruction (`reconstruct_connected_route`):** When collapsing consecutive duplicate `arc_id`s into the final route sequence, if two consecutive distinct predicted arcs $(a_{\text{prev}}, a_{\text{curr}})$ with $\Delta t < 30.0\text{ s}$ are separated by 1 or 2 intermediate road segments, `TripGraph.find_bridge` inserts the shortest connecting bridge `arc_id`(s) (`max_bridge_hops = 3`), increasing route continuity from **75.05%** to **99.49%**.

---

### 6. Did HMM decoding improve the final route? Experimental Comparison & Analysis

#### Table 5.1A: Validation Split Comparison (`train` 5-Fold GroupKFold OOF, 20 Trips)

| Decoding Method | Reliable Top-1 Acc (Weighted %) | Reliable Top-1 Acc (Trip Mean %) | All-Point Top-1 Acc (%) | Candidate Macro F1 | Pointwise Continuity (%) | Reconstructed Route Continuity (%) | Sequence Levenshtein Similarity (%) | Reference Arc Coverage (%) | Route F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only (No HMM)** | 88.06% | 89.02% | 87.05% | 0.9382 | 92.77% | 77.63% | 64.48% | 89.13% | 88.15% |
| **HMM 1: Topological Step-Cost** | 90.58% | 91.55% | 89.44% | 0.9527 | 97.49% | 99.29% | 86.03% | 96.36% | 94.51% |
| **HMM 2: Kinematic Distance-Decay** | 91.52% | 91.76% | 90.43% | 0.9539 | 97.50% | **99.35%** | 87.88% | **96.87%** | 95.05% |
| **HMM 3: Empirical Transition Matrix** | 91.68% | 91.72% | 90.44% | 0.9537 | **99.00%** | 96.04% | 88.35% | 90.27% | 92.55% |
| **HMM Majority Vote (Output Voting 1+2+3)** | 91.75% | **92.20%** | 90.57% | **0.9564** | 97.66% | 99.13% | 87.60% | 96.16% | 94.87% |
| **HMM Ensemble (3-Transition Soft Average)** | **91.98%** | 91.86% | **90.73%** | 0.9546 | 98.32% | 99.01% | 91.87% | 95.67% | 95.35% |
| **HMM Full Ensemble (3-Trans + 3-Classifier)** | 91.90% | 92.01% | 90.65% | 0.9554 | 98.39% | 99.01% | **91.98%** | 95.71% | **95.39%** |

#### Table 5.1B: Student Test Split Comparison (`student_test` 10 Trips, 19,160 Points)

| Decoding Method | Reliable Top-1 Acc (Weighted %) | Reliable Top-1 Acc (Trip Mean %) | All-Point Top-1 Acc (%) | Candidate Macro F1 | Pointwise Continuity (%) | Reconstructed Route Continuity (%) | Sequence Levenshtein Similarity (%) | Reference Arc Coverage (%) | Route F1 (%) | Inference Time (ms/trip) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only (No HMM)** | 88.31% | 89.66% | 88.18% | 0.9407 | 92.96% | 75.05% | 59.39% | 89.33% | 88.31% | 40.8 ms |
| **HMM 1: Topological Step-Cost** | 91.69% | 92.74% | 91.56% | 0.9580 | 97.60% | 99.72% | 86.92% | 98.15% | 95.74% | 40.8 ms |
| **HMM 2: Kinematic Distance-Decay** | 92.07% | 92.40% | 91.94% | 0.9554 | 97.69% | **99.74%** | 88.93% | **98.30%** | 96.33% | 40.8 ms |
| **HMM 3: Empirical Transition Matrix** | 91.59% | 91.70% | 91.46% | 0.9534 | **99.08%** | 96.09% | 86.89% | 88.63% | 91.92% | 40.8 ms |
| **HMM Majority Vote (Output Voting 1+2+3)** | 92.45% | 93.09% | 92.32% | 0.9600 | 97.76% | 99.66% | 88.67% | 97.97% | 96.14% | 40.8 ms |
| **HMM Ensemble (3-Transition Soft Average)** | **93.26%** | **94.00%** | **93.13%** | **0.9662** | 98.53% | 99.49% | 93.38% | 96.46% | 96.38% | 40.8 ms |
| **HMM Full Ensemble (3-Trans + 3-Classifier)** | 92.71% | 93.61% | 92.58% | 0.9641 | 98.55% | 99.54% | **93.40%** | 96.56% | **96.39%** | 40.8 ms |

#### Table 5.2: Per-Trip Error Correction Breakdown on `student_test` (`Classifier-Only` vs. `HMM Ensemble`)

| Trip ID | Total Points | Reliable Points | Fixed by HMM (Clf Wrong → HMM Right) | Hurt by HMM (Clf Right → HMM Wrong) | Net Points Gained | Classifier Top-1 Acc (All %) | HMM Ensemble Top-1 Acc (All %) | Classifier Route Continuity (%) | HMM Route Continuity (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2,503 | 2,503 | 56 | 13 | **+43** | 95.05% | **96.76%** | 69.78% | **98.92%** |
| `00000007` | 1,354 | 1,354 | 99 | 1 | **+98** | 92.25% | **99.48%** | 98.61% | **100.00%** |
| `00000009` | 1,049 | 1,049 | 56 | 22 | **+34** | 87.61% | **90.85%** | 72.64% | **99.28%** |
| `00000040` | 2,995 | 2,995 | 100 | 28 | **+72** | 94.42% | **96.83%** | 75.27% | **100.00%** |
| `00000052` | 262 | 262 | 5 | 1 | **+4** | 97.33% | **98.85%** | 77.65% | **100.00%** |
| `00000056` | 3,958 | 3,931 | 249 | 67 | **+182** | 90.40% | **95.00%** | 71.38% | **99.77%** |
| `00000065` | 768 | 768 | 35 | 12 | **+23** | 85.29% | **88.28%** | 77.78% | **100.00%** |
| `00000076` | 1,486 | 1,486 | 91 | 13 | **+78** | 91.12% | **96.37%** | 65.06% | **98.95%** |
| `00000077` | 3,911 | 3,911 | 578 | 207 | **+371** | 74.38% | **83.87%** | 74.69% | **100.00%** |
| `00000091` | 874 | 874 | 62 | 19 | **+43** | 88.10% | **93.02%** | 67.66% | **97.99%** |
| **Total / Mean** | **19,160** | **19,133** | **1,311** | **383** | **+928** | **88.18%** | **93.13%** | **75.05%** | **99.49%** |

#### Discussion of Improvements and Cases Where HMM Did Not Help
- **Improvements:** Across `student_test`, `HMM Ensemble (3-Transition Soft Average)` raised All-Point Top-1 Accuracy from **88.18% to 93.13%** (**88.31% to 93.26%** weighted reliable Top-1, **89.66% to 94.00%** trip mean), Reconstructed Route Continuity from **75.05% to 99.49%**, and Levenshtein sequence similarity from **59.39% to 93.38%**. It fixed **1,311** classifier errors against **383** points hurt (**+928** net gain).
- **Where HMM Did Not Help:**
  1. **Intersection turn lag and complex urban junctions (`00000077` and `00000056`):** Out of `383` total points hurt by HMM across the 10 trips, `207` occurred on trip `00000077` (lowest final accuracy at `83.87%`, with `424` points wrong in both classifier and HMM) and `67` on trip `00000056`. When GPS observations exhibit sustained multi-second drift across dense intersections, transition smoothing can carry an incorrect branch assignment across adjacent fixes or shift the exact turn timestamp by 1–2 seconds.
  2. **Parallel connected roads (`00000065` at 88.28% and `00000009` at 90.85%):** Trip `00000065` (`12` hurt, `78` wrong in both) and trip `00000009` (`22` hurt, `74` wrong in both) contain parallel connected streets within the 50m candidate window where both paths are topologically connected and kinematically plausible.
