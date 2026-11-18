# M2 — Model Experimentation, Evaluation & Bayesian Optimisation

## Overview
This module implements the candidate-classification pipeline for **PathSnap**:
1. **Candidate Classification:** Predicts the probability $P(y=1 \mid \mathbf{x})$ that each candidate road segment corresponds to the true traversed road segment using M1's 16 engineered features (`m1_data.features.FEATURE_COLUMNS`).
2. **Six Algorithm Families Evaluated:**
   - **Logistic Regression** (Linear / Probabilistic baseline, with `StandardScaler` and `class_weight="balanced"`)
   - **Random Forest Classifier** (Tree ensemble / Bagging)
   - **Extra Trees Classifier** (Extremely Randomized Trees)
   - **Support Vector Machine** (Kernel-based RBF `SVC`, with `StandardScaler`, 50,000-row training subsample)
   - **Gradient Boosting** (`HistGradientBoostingClassifier`, sequential gradient boosted decision trees)
   - **MLP Classifier** (Multi-layer perceptron neural network `(64, 32)` with `StandardScaler` and `early_stopping=True`)
3. **Leakage-Free Validation:** 5-fold `GroupKFold` across the 80 training trips grouped by `trip_id` (trained on reliable observations `label_reliable == 1`).
4. **Primary Competition Metric:** **Observation-level Top-1 Accuracy** (selecting the candidate with the highest predicted probability per `(trip_id, point_idx)`). Also reports Macro F1-score, Precision, Recall, Candidate Accuracy, Log Loss, Classifier-Only Route Continuity, Candidate Miss Rate, and Inference Latency.

---

## Cross-Validation & Student-Test Results

### 1. 5-Fold GroupKFold Cross-Validation (`m2_models/results/cv_results.csv`, 80 Training Trips)

| Model | Algorithm Family | CV Top-1 Accuracy (Mean ± Std) | CV Macro F1 (Mean ± Std) | CV Log Loss | Mean Train Time (s) | Mean Infer Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Ensemble (Bagging) | **87.83% ± 3.36%** | **0.9283 ± 0.0165** | **0.0854** | 189.83s | 1.37s |
| **Gradient Boosting** | Boosting | 87.70% ± 3.46% | 0.9268 ± 0.0168 | 0.0901 | 17.80s | 1.06s |
| **MLP Classifier** | Neural Network | 87.67% ± 3.28% | 0.9239 ± 0.0163 | 0.0930 | 307.30s | 0.59s |
| **Extra Trees** | Ensemble (Randomized Trees) | 87.08% ± 3.36% | 0.9200 ± 0.0172 | 0.0986 | 78.92s | 1.80s |
| **Logistic Regression** | Linear / Probabilistic | 87.03% ± 3.26% | 0.8642 ± 0.0175 | 0.1838 | **4.18s** | **0.02s** |
| **Support Vector Machine** | Kernel-based (RBF) | 86.89% ± 3.32% | 0.9158 ± 0.0185 | 0.1175 | 305.67s | 577.36s |

### 2. Student-Test Evaluation (`m2_models/results/student_test_results.csv`, 10 Trips, 147,061 Candidate Rows)

| Model | Observation Top-1 Acc (%) | Macro F1 | Precision | Recall | Candidate Acc (%) | Log Loss | Route Continuity (%) | Candidate Miss Rate (%) | Inference Time (s) | Latency (ms / 1k cands) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Classifier** | **88.23%** | 0.9240 | 0.8717 | 0.8638 | 96.58% | 0.0819 | 93.02% | 0.21% | 0.027s | 0.19 ms |
| **Gradient Boosting** | 88.18% | 0.9261 | 0.8856 | 0.8569 | 96.70% | 0.0802 | 92.88% | 0.21% | 0.070s | 0.48 ms |
| **Tuned Gradient Boosting (Optuna)** | 88.18% | 0.9270 | 0.8875 | 0.8584 | 96.74% | **0.0786** | 92.96% | 0.21% | 0.113s | 0.77 ms |
| **Random Forest** | 87.97% | **0.9275** | **0.8943** | 0.8536 | **96.79%** | 0.0787 | **93.09%** | 0.21% | 0.099s | 0.67 ms |
| **Logistic Regression** | 86.87% | 0.8618 | 0.6448 | **0.9495** | 92.54% | 0.1752 | 92.35% | 0.21% | **0.001s** | **0.01 ms** |
| **Extra Trees** | 86.80% | 0.9197 | 0.8941 | 0.8275 | 96.48% | 0.0922 | 92.15% | 0.21% | 0.096s | 0.66 ms |
| **Support Vector Machine** | 86.78% | 0.9179 | 0.8933 | 0.8223 | 96.41% | 0.1079 | 92.24% | 0.21% | 12.902s | 87.74 ms |

---

## Bayesian Hyperparameter Optimisation (`optuna_optimization.py`)

- **Optimisation Objective:** Maximize mean 5-fold `GroupKFold` Cross-Validation **Macro F1-score** (at threshold `0.5`) across the 80 training trips using Optuna's Tree-structured Parzen Estimator (`TPESampler`, **30 trials**), while tracking **Observation-level Top-1 Accuracy** as the primary competition metric.
- **Search Space (`HistGradientBoostingClassifier`):**
  - `learning_rate`: $[0.03, 0.25]$ (log-uniform)
  - `max_iter`: $\{50, 70, 90, 110, 130, 150, 170\}$ (`step=20`)
  - `max_depth`: $[6, 18]$ (integer)
  - `min_samples_leaf`: $\{10, 20, \dots, 100\}$ (`step=10`)
  - `l2_regularization`: $[10^{-4}, 10.0]$ (log-uniform)
- **Best Hyperparameters (`m2_models/results/best_parameters.json`, Trial #25):**
  - `learning_rate`: `0.047292`
  - `max_iter`: `170`
  - `max_depth`: `14`
  - `min_samples_leaf`: `90`
  - `l2_regularization`: `0.001464`

### Untuned vs. Optuna-Tuned Gradient Boosting

| Metric | Untuned Gradient Boosting | Tuned Gradient Boosting (Optuna) | Change |
| :--- | :---: | :---: | :---: |
| **5-Fold CV Macro F1** | 0.9268 | **0.9280** | **+0.0012** |
| **Student-Test Observation Top-1 Accuracy** | **88.18%** (`0.881837`) | **88.18%** (`0.881837`) | Tied (`0.00%`) |
| **Student-Test Macro F1-Score** | 0.9261 | **0.9270** | **+0.0009** |
| **Student-Test Log Loss** | 0.0802 | **0.0786** | **-0.0016 (Better)** |
| **Student-Test Route Continuity (Pointwise)** | 92.88% | **92.96%** | **+0.08%** |
| **Student-Test Inference Time (147k rows)** | 0.070s (`0.48 ms/1k`) | 0.113s (`0.77 ms/1k`) | `+0.043s` |

---

## Final Selected Classifier: Tuned Gradient Boosting (`tuned_gradient_boosting.pkl`)

**Final selected classifier: tuned Gradient Boosting** (`m2_models/models/tuned_gradient_boosting.pkl`).
Reasons supported by experimental results:
1. **Best-Calibrated Probabilities for HMM Emissions (`Log Loss = 0.0786`):** Tuned Gradient Boosting achieves the lowest Log Loss across all evaluated models (`0.0786`, improved from `0.0802` untuned and lower than MLP's `0.0819` and Random Forest's `0.0787`), providing well-calibrated candidate probability estimates for M3's log-space Viterbi emission model.
2. **Top-Tier Observation Top-1 Accuracy & Macro F1:** Achieves **88.18%** student-test Top-1 accuracy (matching untuned Gradient Boosting and within `0.05%` of MLP) and **0.9270** student-test Macro F1 (**0.9280** CV Macro F1).
3. **Fast, Self-Contained Inference for the Live API:** Operates directly on raw features without requiring a external `StandardScaler`, has a compact `639 KB` model artifact (vs. `86 MB` for `random_forest.pkl`), and evaluates all 10 student-test trips in **0.113s** (`0.77 ms / 1k candidates`), unlike the RBF SVM (`12.90s`, `87.74 ms / 1k candidates`).

---

## Reproducing All M2 Results (Run from Repository Root)

```bash
pip install -r m2_models/requirements.txt

# 1. Run 5-fold GroupKFold cross-validation across all 6 algorithm families:
python -m m2_models.cross_validation

# 2. Train all 6 baseline models on the 80 training trips (or a subset via --models):
python -m m2_models.train_models
python -m m2_models.train_models --models svm

# 3. Run Optuna Bayesian hyperparameter optimisation (30 trials) and save tuned_gradient_boosting.pkl:
python -m m2_models.optuna_optimization

# 4. Evaluate all 7 models on the 10 student-test trips and save confusion matrices:
python -m m2_models.evaluate_models

# 5. Generate epoch-wise training/validation curves (MLP and Gradient Boosting):
python -m m2_models.training_curves

# 6. Generate model comparison bar charts and latency trade-off plots:
python -m m2_models.compare_models
```
