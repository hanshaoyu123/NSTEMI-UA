#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstemi_ecg.config import load_config, resolve_path
from nstemi_ecg.data import (
    assign_patient_folds,
    build_cohort_dataframe,
    load_metadata,
    split_fit_validation_calibration,
)
from nstemi_ecg.metrics import binary_metrics, find_youden_threshold
from nstemi_ecg.plots import plot_binary_evaluation
from nstemi_ecg.utils import seed_everything, write_json


def build_pipeline(kind: str, columns: list[str]) -> Pipeline:
    if kind == "logistic":
        numeric = ColumnTransformer([("scale", StandardScaler(), columns)], remainder="drop")
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    elif kind == "hist_gradient_boosting":
        numeric = ColumnTransformer([("identity", "passthrough", columns)], remainder="drop")
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=42,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("numeric", numeric), ("classifier", classifier)])


def run_baseline(
    config: dict,
    project_root: Path,
    name: str,
    columns: list[str],
    algorithm: str,
    output_dir: Path,
) -> dict:
    metadata = load_metadata(config, project_root)
    dataframe = build_cohort_dataframe(metadata, config, "train")
    if bool(config["data"].get("use_precomputed_folds", False)):
        dataframe["fold"] = dataframe["cv_fold"].astype(int)
    else:
        dataframe = assign_patient_folds(
            dataframe,
            "_label",
            n_splits=5,
            seed=int(config.get("seed", 42)),
        )
    oof_parts, fold_metrics = [], []
    for fold in sorted(dataframe["fold"].unique()):
        outer_train = dataframe.loc[dataframe["fold"].ne(fold)].reset_index(drop=True)
        outer_validation = dataframe.loc[dataframe["fold"].eq(fold)].reset_index(drop=True)
        fit_df, _, calibration_df = split_fit_validation_calibration(
            outer_train,
            "_label",
            validation_fraction=float(config["training"]["inner_validation_fraction"]),
            calibration_fraction=float(config["training"]["calibration_fraction"]),
            seed=int(config.get("seed", 42)) + int(fold),
        )
        model = build_pipeline(algorithm, columns)
        model.fit(fit_df[columns], fit_df["_label"].astype(int))
        calibration_probabilities = model.predict_proba(calibration_df[columns])[:, 1]
        threshold = find_youden_threshold(
            calibration_df["_label"].astype(int).to_numpy(),
            calibration_probabilities,
        )
        probabilities = model.predict_proba(outer_validation[columns])[:, 1]
        result = binary_metrics(
            outer_validation["_label"].astype(int).to_numpy(),
            probabilities,
            threshold,
        )
        result["fold"] = int(fold)
        fold_metrics.append(result)
        part = outer_validation[
            ["metadata_index", "Patient_id", "record_id", "_label", "fold"]
        ].copy()
        part["probability"] = probabilities
        part["threshold"] = threshold
        part["prediction"] = (probabilities >= threshold).astype(int)
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True)
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(fold_metrics, run_dir / "fold_metrics.json")
    threshold = float(np.median(oof.groupby("fold")["threshold"].first().to_numpy()))
    summary = binary_metrics(oof["_label"].astype(int).to_numpy(), oof["probability"].to_numpy(), threshold)
    plot_binary_evaluation(
        oof["_label"].astype(int).to_numpy(),
        oof["probability"].to_numpy(),
        threshold,
        run_dir / "evaluation",
    )
    write_json({"model": name, "columns": columns, "algorithm": algorithm, "pooled_oof": summary}, run_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Age/sex and workflow-variable baselines")
    parser.add_argument("--config", default="configs/main_nstemi_vs_ua.yaml")
    parser.add_argument("--output-dir", default="runs/tabular_baselines")
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output_dir = resolve_path(project_root, args.output_dir)
    summaries = {}
    summaries["age_sex_logistic"] = run_baseline(
        config,
        project_root,
        "age_sex_logistic",
        ["age", "gender"],
        "logistic",
        output_dir,
    )
    summaries["age_sex_time_logistic"] = run_baseline(
        config,
        project_root,
        "age_sex_time_logistic",
        ["age", "gender", "Time_Interval"],
        "logistic",
        output_dir,
    )
    summaries["age_sex_time_gradient_boosting"] = run_baseline(
        config,
        project_root,
        "age_sex_time_gradient_boosting",
        ["age", "gender", "Time_Interval"],
        "hist_gradient_boosting",
        output_dir,
    )
    write_json(summaries, output_dir / "tabular_baseline_summary.json")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()

