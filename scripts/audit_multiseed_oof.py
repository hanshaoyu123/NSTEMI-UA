#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_RUNS = [
    "runs/main_nstemi_vs_ua",
    "runs/main_nstemi_vs_ua_seed43",
    "runs/main_nstemi_vs_ua_seed44",
    "runs/main_nstemi_vs_ua_seed45",
    "runs/main_nstemi_vs_ua_seed46",
]


def audit_run(run_dir: Path) -> dict[str, float | int | str | bool]:
    config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "oof_predictions.csv")
    required = {"_label", "probability", "fold", "Patient_id"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{run_dir} is missing columns: {sorted(missing)}")

    labels = predictions["_label"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    folds = predictions["fold"].to_numpy(dtype=int)
    rank_scores = np.empty_like(probabilities)
    fold_aurocs: list[float] = []
    fold_median_probabilities: list[float] = []

    for fold in np.unique(folds):
        mask = folds == fold
        fold_labels = labels[mask]
        fold_probabilities = probabilities[mask]
        if len(np.unique(fold_labels)) < 2:
            raise ValueError(f"{run_dir} fold {fold} does not contain both classes")
        fold_aurocs.append(float(roc_auc_score(fold_labels, fold_probabilities)))
        fold_median_probabilities.append(float(np.median(fold_probabilities)))
        rank_scores[mask] = (
            rankdata(fold_probabilities, method="average") - 0.5
        ) / mask.sum()

    medians = np.asarray(fold_median_probabilities)
    return {
        "run": run_dir.name,
        "seed": int(config["seed"]),
        "n": int(len(predictions)),
        "positive": int(labels.sum()),
        "pooled_auroc": float(roc_auc_score(labels, probabilities)),
        "fold_mean_auroc": float(np.mean(fold_aurocs)),
        "rank_normalized_pooled_auroc": float(roc_auc_score(labels, rank_scores)),
        "pooled_auprc": float(average_precision_score(labels, probabilities)),
        "rank_normalized_pooled_auprc": float(
            average_precision_score(labels, rank_scores)
        ),
        "fold_median_probability_sd": float(np.std(medians, ddof=1)),
        "same_oof_patients": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit cross-fold probability-scale effects in multi-seed OOF results"
    )
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument(
        "--output",
        default="runs/multiseed_oof_audit.csv",
    )
    args = parser.parse_args()

    rows = []
    reference_patients: np.ndarray | None = None
    for run in args.runs:
        run_dir = Path(run)
        result = audit_run(run_dir)
        predictions = pd.read_csv(run_dir / "oof_predictions.csv")
        patients = np.sort(predictions["Patient_id"].astype(str).to_numpy())
        if reference_patients is None:
            reference_patients = patients
        elif not np.array_equal(reference_patients, patients):
            result["same_oof_patients"] = False
        rows.append(result)

    dataframe = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)

    for metric in (
        "pooled_auroc",
        "fold_mean_auroc",
        "rank_normalized_pooled_auroc",
        "pooled_auprc",
        "rank_normalized_pooled_auprc",
    ):
        values = dataframe[metric].to_numpy(dtype=float)
        print(
            f"{metric}: mean={values.mean():.6f}, "
            f"sd={values.std(ddof=1):.6f}"
        )
    print(dataframe.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
