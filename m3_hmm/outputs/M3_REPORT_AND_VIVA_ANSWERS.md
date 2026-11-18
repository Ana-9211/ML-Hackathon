# Section 5 & 6 Technical Report Content (M3: HMM & Viterbi Route Reconstruction)

---

## 5. HMM and Viterbi Route Reconstruction

### Q1. What are the hidden states in your implementation?
At each GPS observation step $t \in \{0, 1, \dots, T-1\}$, the hidden state $S_t$ represents the **true directed road segment (`arc_id`)** on which the vehicle is travelling at timestamp $\text{time}_t$.
To maintain computational tractability ($O(T \cdot K^2)$ instead of $O(T \cdot |E|^2)$ across up to $307,392$ road segments), the state space at timestamp $t$ is restricted to the $K_t \le 12$ candidate road segments $\{c_{t,1}, c_{t,2}, \dots, c_{t,K_t}\}$ retrieved within the $50\text{ m}$ spatial search radius (or $200\text{ m}$ fallback radius). Every candidate preserves its exact 0-based `arc_id`, `arc_from`, `arc_to`, and longitudinal projection fraction $\text{proj\_t} \in [0, 1]$.

---

### Q2. How are emission probabilities calculated?
For each candidate $c_{t,k}$ at step $t$, M2's official trained candidate classifiers (`gradient_boosting.pkl`, `random_forest.pkl`, and `mlp.pkl` in `m2_models/models/`) output the pointwise probability $P_{\text{ML}}(y_{t,k} = 1 \mid \mathbf{x}_{t,k})$.
We convert these raw probabilities into scaled log-emission scores normalized across the $K_t$ candidates at timestamp $t$:
$$\log E_t(k) = \alpha_{\text{em}} \cdot \log \left( \frac{\max(P_{\text{ML}}(c_{t,k}), \epsilon)}{\sum_{m=1}^{K_t} \max(P_{\text{ML}}(c_{t,m}), \epsilon)} \right)$$
where $\epsilon = 10^{-4}$ prevents $\log(0)$ numerical underflow and $\alpha_{\text{em}} = 0.719$ (tuned via Optuna on training out-of-fold predictions) balances emission confidence against road-network transition constraints.

---

### Q3. How are transition probabilities calculated? (3-Technique Log-Linear Transition Ensemble)
Rather than relying on a single heuristic rule, we formulated **three distinct transition models** between candidate $a_i = (u_i \to v_i)$ at step $t-1$ and candidate $a_j = (u_j \to v_j)$ at step $t$ (separated by $\Delta t = \text{time}_t - \text{time}_{t-1}$) and combined them in log-space inside a single Viterbi pass:

1. **Technique 1 — Topological Graph Step-Cost ($\log A_{\text{topo}}$):**
   Uses a pre-indexed directed road graph (`TripGraph`) to classify each pair $(a_i \to a_j)$ into 7 topological relations:
   - **Same arc forward ($a_i = a_j, \Delta s \ge -5\text{ m}$):** $0.0$
   - **Same arc backward ($a_i = a_j, \Delta s < -5\text{ m}$):** $-2.20$ (penalizes reverse movement beyond GPS jitter)
   - **Direct 1-hop connection ($v_i = u_j$, non-reverse):** $-0.35$
   - **2-hop shortest bridge ($a_i \to b_1 \to a_j$):** $-1.60 \times \text{gap\_factor}(\Delta t)$
   - **3-hop shortest bridge ($a_i \to b_1 \to b_2 \to a_j$):** $-2.90 \times \text{gap\_factor}(\Delta t)$
   - **Immediate U-turn ($v_i = u_j \land u_i = v_j$):** $-4.20$
   - **Disconnected (no path within 3 hops):** $-9.50$

2. **Technique 2 — Kinematic Distance-Consistency ($\log A_{\text{kin}}$, Newson & Krumm):**
   Compares the distance travelled along the road graph $d_{\text{road}}$ (computed from longitudinal projections `(1 - proj_t[t-1]) * len(a_i) + len(bridge) + proj_t[t] * len(a_j)`) against Euclidean GPS displacement $d_{\text{GPS}}$:
   $$\log A_{\text{kin}}(a_i \to a_j \mid \Delta t) = -\min\left(10.0,\, \frac{|d_{\text{road}} - d_{\text{GPS}}|}{\beta \cdot \max(1.0, \Delta t)}\right), \quad \beta = 18.0\text{ m/s}$$

3. **Technique 3 — Data-Driven Empirical Transition Priors ($\log A_{\text{emp}}$):**
   Estimated directly from the 80 training trips (`208,417` GPS observations) by counting actual ground-truth transition frequencies conditioned on $(\Delta t\text{ bucket}, \text{speed bucket})$ with Laplace smoothing.

4. **Soft Log-Linear Ensemble Transition Score:**
   Optimized via 40 trials of Optuna TPE on 20 validation training trips:
   $$\log A_{\text{ensemble}}(a_i \to a_j \mid \Delta t) = 0.165 \log A_{\text{topo}} + 0.477 \log A_{\text{kin}} + 0.358 \log A_{\text{emp}}$$

---

### Q4. How are missing candidates or large GPS gaps handled?
- **Missing Candidates:** During candidate generation (`m1_data/candidates.py`), if no road segment lies within $50\text{ m}$, the search radius automatically expands to a $200\text{ m}$ fallback radius (`fallback = 1`). In the live inference pipeline (`m3_hmm/inference.py`), any extreme off-road point with zero candidates even at $200\text{ m}$ is interpolated from the nearest valid Viterbi state.
- **Medium GPS Gaps ($2.5\text{ s} < \Delta t < 30\text{ s}$):** Multi-hop topological penalties are relaxed by $\text{gap\_factor} = \max(0.35, 2.5/\Delta t)$ and the kinematic scale $\beta \cdot \Delta t$ widens proportionally to elapsed time, allowing the decoder to naturally traverse 2–3 connected road segments during brief signal dropouts.
- **Large GPS Dropouts ($\Delta t \ge 30\text{ s}$):** When a vehicle enters a long tunnel or parks (e.g., multi-minute gaps), transition matrices reset to $\mathbf{0}$, allowing the trajectory after the gap to re-anchor cleanly based on emission probabilities without artificial disconnection penalties.

---

### Q5. What happens when consecutive GPS observations suggest disconnected road segments?
1. **During Viterbi Decoding:** Disconnected candidate transitions receive heavy negative log-scores across all three transition models ($\log A_{\text{topo}} = -9.5$, $\log A_{\text{kin}} = -10.0$, and low empirical probability $\log A_{\text{emp}}$). Consequently, Viterbi overrides single-point GPS spikes onto side streets or parallel lanes unless a candidate sequence remains consistently supported over many consecutive observations.
2. **During Post-Viterbi Route Reconstruction (`reconstruct_connected_route`):** When collapsing the length-$T$ observation sequence into the deduplicated `.route` sequence, if two consecutive distinct predicted arcs $(a_{\text{prev}}, a_{\text{curr}})$ are separated by 1 or 2 intermediate road segments (e.g., crossing a short intersection link between 1 Hz pings), `TripGraph.find_bridge` inserts the shortest connecting bridge `arc_id`(s), raising reconstructed route continuity to **99.37%–99.75%**.

---

### Q6. Did HMM decoding improve the final route? Validation & Test Results

#### Table 5.1A: Validation Split Comparison (`train` 5-Fold GroupKFold OOF Validation Trips)

| Decoding Method | Reliable Top-1 Acc (Weighted %) | Reliable Top-1 Acc (Trip Mean %) | All-Point Top-1 Acc (%) | Candidate Macro F1 | Pointwise Continuity (%) | Reconstructed Route Continuity (%) | Sequence Levenshtein Similarity (%) | Reference Arc Coverage (%) | Route F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only (No HMM)** | 88.04% | 88.97% | 86.98% | 0.9379 | 92.57% | 76.99% | 64.17% | 89.04% | 87.99% |
| **HMM 1: Topological Step-Cost** | 90.53% | 91.54% | 89.44% | 0.9526 | 97.44% | 99.22% | 85.64% | 96.44% | 94.54% |
| **HMM 2: Kinematic Distance-Decay** | 90.98% | 91.83% | 89.85% | 0.9543 | 97.47% | **99.35%** | 87.56% | **96.72%** | 94.89% |
| **HMM 3: Empirical Transition Matrix** | 91.74% | 91.73% | 90.52% | 0.9538 | **98.95%** | 95.86% | 88.17% | 90.24% | 92.36% |
| **HMM Majority Vote (Output Voting 1+2+3)** | 91.69% | 92.41% | 90.51% | 0.9576 | 97.62% | 99.05% | 87.51% | 96.22% | 94.90% |
| **HMM Ensemble (3-Transition Soft Average)** | **92.68%** | **92.80%** | **91.45%** | **0.9598** | 98.39% | 98.80% | 91.83% | 95.41% | 95.21% |
| **HMM Full Ensemble (3-Trans + 3-Classifier)** | 92.63% | 91.96% | 91.37% | 0.9551 | 98.50% | 98.76% | **92.17%** | 95.48% | **95.34%** |

#### Table 5.1B: Student Test Split Comparison (`student_test` 10 Trips, 19,160 Points — Official M2 `.pkl` Models)

| Decoding Method | Reliable Top-1 Acc (Weighted %) | Reliable Top-1 Acc (Trip Mean %) | All-Point Top-1 Acc (%) | Candidate Macro F1 | Pointwise Continuity (%) | Reconstructed Route Continuity (%) | Sequence Levenshtein Similarity (%) | Reference Arc Coverage (%) | Route F1 (%) | Inference Time (ms/trip) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only (M2 Gradient Boosting)** | 88.31% | 89.62% | 88.18% | 0.9405 | 92.88% | 74.80% | 59.69% | 89.40% | 88.20% | 39.1 ms |
| **HMM 1: Topological Step-Cost** | 91.05% | 92.28% | 90.92% | 0.9553 | 97.56% | 99.68% | 86.11% | 97.98% | 95.52% | 39.1 ms |
| **HMM 2: Kinematic Distance-Decay** | 91.95% | 92.28% | 91.82% | 0.9547 | 97.71% | **99.75%** | 88.69% | **98.30%** | 96.34% | 39.1 ms |
| **HMM 3: Empirical Transition Matrix** | 90.87% | 91.33% | 90.75% | 0.9513 | **99.06%** | 96.05% | 86.81% | 88.77% | 91.90% | 39.1 ms |
| **HMM Majority Vote (Output Voting 1+2+3)** | 91.80% | 92.63% | 91.67% | 0.9572 | 97.71% | 99.62% | 87.92% | 97.81% | 95.96% | 39.1 ms |
| **HMM Ensemble (3-Transition Soft Average)** | **93.36%** | **93.92%** | **93.23%** | **0.9659** | 98.58% | 99.34% | 93.84% | **96.38%** | **96.47%** | **39.1 ms** |
| **HMM Full Ensemble (3-Trans + 3-Classifier)** | 93.34% | 93.75% | 93.20% | 0.9650 | 98.69% | 99.37% | **93.89%** | 96.20% | 96.38% | **39.1 ms** |

#### Table 5.2: Per-Trip Error Correction Breakdown on `student_test` (Official M2 Models + M3 HMM Ensemble)

| Trip ID | Total Points | Reliable Points | Fixed by HMM (Clf Wrong → HMM Right) | Hurt by HMM (Clf Right → HMM Wrong) | Net Points Gained | Classifier Top-1 Acc (All %) | HMM Ensemble Top-1 Acc (All %) | Classifier Route Continuity (%) | HMM Route Continuity (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2,503 | 2,503 | 56 | 9 | **+47** | 95.17% | **97.04%** | 69.96% | **98.37%** |
| `00000007` | 1,354 | 1,354 | 100 | 1 | **+99** | 92.17% | **99.48%** | 98.60% | **100.00%** |
| `00000009` | 1,049 | 1,049 | 42 | 36 | **+6** | 87.61% | **88.18%** | 73.11% | **98.79%** |
| `00000040` | 2,995 | 2,995 | 120 | 12 | **+108** | 94.49% | **98.10%** | 73.88% | **100.00%** |
| `00000052` | 262 | 262 | 5 | 2 | **+3** | 97.33% | **98.47%** | 77.65% | **100.00%** |
| `00000056` | 3,958 | 3,931 | 247 | 74 | **+173** | 90.40% | **94.77%** | 71.38% | **99.62%** |
| `00000065` | 768 | 768 | 31 | 14 | **+17** | 85.03% | **87.24%** | 77.37% | **100.00%** |
| `00000076` | 1,486 | 1,486 | 91 | 14 | **+77** | 91.32% | **96.50%** | 65.28% | **98.95%** |
| `00000077` | 3,911 | 3,911 | 582 | 194 | **+388** | 74.33% | **84.25%** | 74.79% | **100.00%** |
| `00000091` | 874 | 874 | 65 | 21 | **+44** | 87.76% | **92.79%** | 65.94% | **97.97%** |
| **Total / Mean** | **19,160** | **19,133** | **1,339** | **377** | **+962** | **88.18%** | **93.23%** | **74.80%** | **99.37%** |
