"""
m2_models.training_curves
~~~~~~~~~~~~~~~~~~~~~~~~~
Training and validation loss curves for the models that record an iterative training
history (read from the saved models, no retraining):

- MLP Classifier: training loss per epoch (`loss_curve_`) and the accuracy on its internal
  early-stopping validation split (`validation_scores_`).
- Gradient Boosting (untuned and Optuna-tuned): binary log loss per boosting iteration on the
  training data and on the internal early-stopping validation split
  (`train_score_` / `validation_score_`; scikit-learn stores the negative loss).

Logistic Regression, Random Forest, Extra Trees and the SVM do not produce epoch-wise
training histories, so no curves are drawn for them.

Usage (repo root):  python -m m2_models.training_curves
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = Path("m2_models/models")
PLOTS = Path("m2_models/plots")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)

    mlp = joblib.load(MODELS / "mlp.pkl")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    epochs = range(1, len(mlp.loss_curve_) + 1)
    ax1.plot(epochs, mlp.loss_curve_, color="#2b5c8f", marker="o", ms=3, label="Training loss (log loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#2b5c8f")
    ax2 = ax1.twinx()
    ax2.plot(epochs, mlp.validation_scores_, color="#d32f2f", marker="s", ms=3,
             label="Validation accuracy (early-stopping split)")
    ax2.set_ylabel("Validation accuracy", color="#d32f2f")
    ax1.set_title(f"MLP Classifier training history ({len(mlp.loss_curve_)} epochs, early stopping)")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="center right")
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curve_mlp.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (fname, title) in zip(axes, [("gradient_boosting.pkl", "Gradient Boosting (untuned)"),
                                         ("tuned_gradient_boosting.pkl", "Gradient Boosting (Optuna-tuned)")]):
        m = joblib.load(MODELS / fname)
        it = range(len(m.train_score_))
        ax.plot(it, -m.train_score_, color="#2b5c8f", label="Training log loss")
        ax.plot(it, -m.validation_score_, color="#d32f2f", label="Validation log loss (early-stopping split)")
        ax.set_title(f"{title}: {m.n_iter_} iterations")
        ax.set_xlabel("Boosting iteration")
        ax.legend()
    axes[0].set_ylabel("Binary log loss")
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curve_gradient_boosting.png", dpi=200)
    plt.close(fig)
    print("Saved training_curve_mlp.png and training_curve_gradient_boosting.png")


if __name__ == "__main__":
    main()
