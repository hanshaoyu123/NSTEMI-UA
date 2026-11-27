from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def find_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.round(probabilities, 6))
    if len(candidates) > 4096:
        candidates = np.quantile(candidates, np.linspace(0, 1, 4096))
    candidates = np.unique(np.clip(np.r_[0.0, candidates, 1.0], 0.0, 1.0))
    best_threshold, best_j = 0.5, -np.inf
    positives = labels == 1
    negatives = ~positives
    for threshold in candidates:
        predictions = probabilities >= threshold
        sensitivity = np.mean(predictions[positives]) if positives.any() else 0.0
        specificity = np.mean(~predictions[negatives]) if negatives.any() else 0.0
        j_score = sensitivity + specificity - 1.0
        if j_score > best_j:
            best_threshold, best_j = float(threshold), float(j_score)
    return best_threshold


def find_threshold_for_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positive_scores = np.sort(probabilities[labels == 1])[::-1]
    if not len(positive_scores):
        return 0.5
    rank = int(np.ceil(target_sensitivity * len(positive_scores))) - 1
    return float(np.clip(positive_scores[max(0, rank)], 0.0, 1.0))


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    result = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((1 - labels).sum()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result


def multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    predictions = probabilities.argmax(axis=1)
    try:
        macro_auroc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=list(range(num_classes)),
        )
    except ValueError:
        macro_auroc = float("nan")
    return {
        "n": int(len(labels)),
        "num_classes": num_classes,
        "macro_auroc": float(macro_auroc),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(num_classes))).tolist(),
    }


def bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Patient-level bootstrap for binary discrimination metrics."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "sensitivity": [],
        "specificity": [],
        "brier": [],
    }
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        y = labels[indices]
        p = probabilities[indices]
        if len(np.unique(y)) < 2:
            continue
        metrics = binary_metrics(y, p, threshold)
        for key in values:
            values[key].append(metrics[key])
    return {
        key: [
            float(np.nanpercentile(metric_values, 2.5)),
            float(np.nanpercentile(metric_values, 97.5)),
        ]
        for key, metric_values in values.items()
        if metric_values
    }


def calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if mask.any():
            total += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(total)


def calibration_slope_intercept(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
    model.fit(logits, labels)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    lower: float = 0.05,
    upper: float = 10.0,
) -> float:
    """Fit binary temperature scaling on a held-out calibration set."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = logits / temperature
        loss = np.logaddexp(0.0, scaled) - labels * scaled
        return float(np.mean(loss))

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": 1.0e-8},
    )
    return float(np.exp(result.x))


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=-1)
    sorted_values = np.take_along_axis(values, order, axis=-1)
    ranks = np.empty_like(values, dtype=float)
    for row_index in range(values.shape[0] if values.ndim > 1 else 1):
        sorted_row = sorted_values[row_index] if values.ndim > 1 else sorted_values
        output_row = ranks[row_index] if values.ndim > 1 else ranks
        i = 0
        while i < len(sorted_row):
            j = i
            while j < len(sorted_row) and sorted_row[j] == sorted_row[i]:
                j += 1
            output_row[i:j] = 0.5 * (i + j - 1) + 1.0
            i = j
    return np.take_along_axis(ranks, np.argsort(order, axis=-1), axis=-1)


def _fast_delong(
    predictions: np.ndarray,
    positive_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    m, n = positive_count, predictions.shape[1] - positive_count
    positive_scores = predictions[:, :m]
    negative_scores = predictions[:, m:]
    tx = _compute_midrank(positive_scores)
    ty = _compute_midrank(negative_scores)
    tz = _compute_midrank(predictions)
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_auc_test(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
) -> dict[str, float]:
    """Paired DeLong test for two binary prediction scores on identical patients."""
    labels = np.asarray(labels, dtype=int)
    probabilities_a = np.asarray(probabilities_a, dtype=float)
    probabilities_b = np.asarray(probabilities_b, dtype=float)
    if not (len(labels) == len(probabilities_a) == len(probabilities_b)):
        raise ValueError("Prediction arrays must have equal length")
    positive = labels == 1
    negative = labels == 0
    scores = np.hstack(
        [
            np.vstack([probabilities_a[positive], probabilities_b[positive]]),
            np.vstack([probabilities_a[negative], probabilities_b[negative]]),
        ]
    )
    aucs, covariance = _fast_delong(scores, int(positive.sum()))
    delta = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    z_score = delta / np.sqrt(max(variance, 1.0e-15))
    p_value = float(2.0 * norm.sf(abs(z_score)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_difference": delta,
        "z": float(z_score),
        "p_value": p_value,
    }


def decision_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds)
    prevalence = labels.mean()
    model_net_benefit = np.zeros_like(thresholds, dtype=float)
    for index, threshold in enumerate(thresholds):
        predictions = probabilities >= threshold
        tp = np.sum(predictions & (labels == 1))
        fp = np.sum(predictions & (labels == 0))
        model_net_benefit[index] = tp / len(labels) - fp / len(labels) * threshold / (1 - threshold)
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    return thresholds, model_net_benefit, treat_all
