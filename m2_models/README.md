# M2 — Model experimentation and Bayesian optimisation

Owner: M2

Scope:
- Train and compare at least five models (group-aware CV on the 80 training trips, grouped by `trip_id`)
- Student-test metrics: Top-1 accuracy, macro F1, log loss, precision/recall, confusion matrix, training and inference time, route continuity
- Comparison plots
- Bayesian hyperparameter optimisation of the chosen model, with the trial log kept
- Export the final model artifact for M3/M4

Input from M1: `m1_data/outputs/features/features_<partition>.parquet` (see `m1_data/README.md` for the columns).
