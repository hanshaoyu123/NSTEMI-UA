from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .metrics import decision_curve


def _save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_binary_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
    calibration_bins: int = 10,
) -> None:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    threshold_curve, model_nb, all_nb = decision_curve(labels, probabilities)
    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.plot(fpr, tpr, color="#B94A48", linewidth=2)
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axis.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (OOF)")
    _save_all(fig, output_dir / "roc_oof")

    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.plot(recall, precision, color="#3B6E8F", linewidth=2)
    axis.axhline(labels.mean(), color="#777777", linestyle="--", linewidth=1)
    axis.set(xlabel="recall", ylabel="precision", title="Precision-recall (OOF)")
    _save_all(fig, output_dir / "pr_oof")

    fraction_positive, mean_predicted = calibration_curve(
        labels,
        probabilities,
        n_bins=calibration_bins,
        strategy="quantile",
    )
    fig, axis = plt.subplots(figsize=(4.8, 4.5))
    axis.plot(mean_predicted, fraction_positive, marker="o", color="#4F7C5A")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axis.set(xlabel="mean predicted probability", ylabel="observed proportion", title="Calibration (OOF)")
    _save_all(fig, output_dir / "calibration_oof")

    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.plot(threshold_curve, model_nb, color="#B94A48", linewidth=2, label="Model")
    axis.plot(threshold_curve, all_nb, color="#333333", linestyle="--", linewidth=1, label="Treat all")
    axis.axhline(0.0, color="#777777", linestyle=":", linewidth=1, label="Treat none")
    axis.set(xlabel="threshold probability", ylabel="net benefit", title="Decision curve (OOF)")
    axis.legend(frameon=False)
    _save_all(fig, output_dir / "decision_curve_oof")

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(4.2, 3.8))
    image = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center", color="black")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Predicted negative", "Predicted positive"],
        yticklabels=["True negative", "True positive"],
        title=f"Confusion matrix @ {threshold:.3f}",
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save_all(fig, output_dir / "confusion_matrix_oof")


def plot_fold_metrics(fold_metrics: list[dict], output_dir: Path) -> None:
    if not fold_metrics:
        return
    folds = [item["fold"] for item in fold_metrics]
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "f1"]
    fig, axis = plt.subplots(figsize=(8, 3.8))
    width = 0.15
    for index, metric in enumerate(metrics):
        values = [item.get(metric, np.nan) for item in fold_metrics]
        axis.bar(np.arange(len(folds)) + index * width, values, width=width, label=metric)
    axis.set_xticks(np.arange(len(folds)) + width * 2)
    axis.set_xticklabels([f"fold {fold}" for fold in folds])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("metric")
    axis.set_title("Cross-validation performance")
    axis.legend(frameon=False, ncol=5)
    fig.tight_layout()
    _save_all(fig, output_dir / "fold_metrics")

