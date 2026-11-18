# M2 — Model Experimentation, Evaluation & Bayesian Optimisation

## Overview
This module implements the complete candidate-classification pipeline for **PathSnap**:
1. **Candidate Classification**: Predicts the likelihood that each road candidate corresponds to the true traversed road segment.
2. **6 Algorithm Families**:
   - Logistic Regression (Linear / Probabilistic baseline)
   - Random Forest Classifier (Tree ensemble / Bagging)
   - Extra Trees Classifier (Extremely Randomized Trees)
   - Support Vector Classifier (Kernel-based RBF)
   - HistGradientBoostingClassifier (Sequential gradient boosted decision trees)
   - MLP Classifier (Multi-layer perceptron neural network)
3. **Leakage-Free Validation**: Evaluated strictly using 5-fold `GroupKFold` across 80 training trips grouped by `trip_id`.
4. **Metrics Evaluated**:
   - Observation-Level Top-1 Accuracy (Primary competition metric)
   - Macro F1-score (Class-balanced evaluation)
   - Precision, Recall, Accuracy, Log Loss
   - Training & Inference Latency
5. **Bayesian Optimization**: Optuna-driven hyperparameter tuning on the best performing model.

## Folder Structure
```text
m2_models/
├── data/                    # features_train.parquet & features_student_test.parquet
├── models/                  # Serialized .pkl model artifacts
├── results/                 # cv_results.csv, student_test_results.csv, optuna_trials.csv
├── plots/                   # cv_comparison.png, accuracy_comparison.png, etc.
├── utils.py                 # Core evaluation metrics & Top-1 observation logic
├── cross_validation.py      # 5-fold GroupKFold CV runner
├── train_models.py          # Trains all 6 models on full 80 trips
├── evaluate_models.py       # Evaluates saved models on 10 student-test trips
├── optuna_optimization.py   # Bayesian hyperparameter tuning with Optuna
├── compare_models.py        # Generates comparison tables and visualizations
└── README.md
```
