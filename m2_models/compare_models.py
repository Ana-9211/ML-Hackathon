"""
m2_models.compare_models
~~~~~~~~~~~~~~~~~~~~~~~~
Generates all official comparison visualizations, tables, and charts
for the M2 report and submission:
1. `plots/cv_comparison.png`: Mean CV Macro F1 & Top-1 Accuracy across all 6 models with error bars
2. `plots/accuracy_comparison.png`: Student Test Top-1 Accuracy bar chart
3. `plots/f1_comparison.png`: Student Test Macro F1 comparison
4. `plots/log_loss_comparison.png`: Student Test Log Loss comparison
5. `plots/latency_vs_accuracy.png`: Inference latency vs Top-1 accuracy trade-off scatter plot
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def generate_all_plots(
    results_dir: str = "m2_models/results",
    plots_dir: str = "m2_models/plots",
):
    res_path = Path(results_dir)
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    cv_file = res_path / "cv_results.csv"
    test_file = res_path / "student_test_results.csv"

    if not cv_file.exists() or not test_file.exists():
        logging.error(f"Required result CSVs missing in {results_dir}")
        return

    df_cv = pd.read_csv(cv_file)
    df_test = pd.read_csv(test_file)

    # Palette
    colors = ["#2b5c8f", "#388e3c", "#d32f2f", "#7b1fa2", "#f57c00", "#0097a7", "#c2185b"]

    # -------------------------------------------------------------
    # 1. cv_comparison.png (Top-1 Accuracy & Macro F1 with error bars)
    # -------------------------------------------------------------
    logging.info("Generating cv_comparison.png...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df_cv))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        df_cv["cv_top1_accuracy_mean"] * 100,
        width,
        yerr=df_cv["cv_top1_accuracy_std"] * 100,
        capsize=5,
        label="Observation Top-1 Accuracy (%)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
    )
    rects2 = ax.bar(
        x + width / 2,
        df_cv["cv_macro_f1_mean"] * 100,
        width,
        yerr=df_cv["cv_macro_f1_std"] * 100,
        capsize=5,
        label="Macro F1 Score (×100)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel("Metric Value (%)", fontweight="bold")
    ax.set_title("5-Fold GroupKFold Cross-Validation Performance Across 6 Algorithm Families", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_cv["model"], rotation=15, ha="right", fontweight="medium")
    ax.set_ylim(70, 100)
    ax.legend(loc="lower right", frameon=True)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    cv_plot_path = plots_path / "cv_comparison.png"
    plt.savefig(cv_plot_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. accuracy_comparison.png (Student Test Top-1 Accuracy)
    # -------------------------------------------------------------
    logging.info("Generating accuracy_comparison.png...")
    fig, ax = plt.subplots(figsize=(9, 5))
    df_sorted = df_test.sort_values(by="top1_accuracy", ascending=True)

    bars = ax.barh(
        df_sorted["model"],
        df_sorted["top1_accuracy"] * 100,
        color="#1976d2",
        edgecolor="black",
        height=0.6,
    )
    ax.set_xlabel("Observation-Level Top-1 Accuracy (%)", fontweight="bold")
    ax.set_title("Student Testing Set: Observation Top-1 Accuracy by Model", fontweight="bold", pad=15)
    ax.set_xlim(80, 95)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", ha="left", fontweight="bold")

    plt.tight_layout()
    acc_plot_path = plots_path / "accuracy_comparison.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. f1_comparison.png (Macro F1 comparison)
    # -------------------------------------------------------------
    logging.info("Generating f1_comparison.png...")
    fig, ax = plt.subplots(figsize=(9, 5))
    df_f1 = df_test.sort_values(by="macro_f1", ascending=True)

    bars = ax.barh(
        df_f1["model"],
        df_f1["macro_f1"],
        color="#388e3c",
        edgecolor="black",
        height=0.6,
    )
    ax.set_xlabel("Macro F1-Score", fontweight="bold")
    ax.set_title("Student Testing Set: Macro F1-Score (Handling Class Imbalance)", fontweight="bold", pad=15)
    ax.set_xlim(0.80, 0.98)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontweight="bold")

    plt.tight_layout()
    f1_plot_path = plots_path / "f1_comparison.png"
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. log_loss_comparison.png (Log Loss comparison)
    # -------------------------------------------------------------
    logging.info("Generating log_loss_comparison.png...")
    fig, ax = plt.subplots(figsize=(9, 5))
    df_loss = df_test.sort_values(by="log_loss", ascending=False)

    bars = ax.barh(
        df_loss["model"],
        df_loss["log_loss"],
        color="#d32f2f",
        edgecolor="black",
        height=0.6,
    )
    ax.set_xlabel("Log Loss (Lower is Better)", fontweight="bold")
    ax.set_title("Student Testing Set: Probability Calibration & Log Loss", fontweight="bold", pad=15)
    ax.set_xlim(0, max(df_loss["log_loss"]) * 1.2)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontweight="bold")

    plt.tight_layout()
    loss_plot_path = plots_path / "log_loss_comparison.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. latency_vs_accuracy.png (Latency vs Top-1 Accuracy Scatter)
    # -------------------------------------------------------------
    logging.info("Generating latency_vs_accuracy.png...")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, row in df_test.iterrows():
        ax.scatter(
            row["latency_ms_per_1k_cand"],
            row["top1_accuracy"] * 100,
            s=180,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            row["model"],
            xy=(row["latency_ms_per_1k_cand"], row["top1_accuracy"] * 100),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms per 1,000 candidates, log scale)", fontweight="bold")
    ax.set_ylabel("Observation Top-1 Accuracy (%)", fontweight="bold")
    ax.set_title("Trade-off Analysis: Inference Latency vs. Top-1 Accuracy", fontweight="bold", pad=15)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    lat_plot_path = plots_path / "latency_vs_accuracy.png"
    plt.savefig(lat_plot_path, dpi=300)
    plt.close()

    logging.info(f"All 5 comparison plots generated and saved to {plots_path}!")


if __name__ == "__main__":
    generate_all_plots()
