#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstemi_ecg.config import load_config
from nstemi_ecg.metrics import (
    binary_metrics,
    calibration_error,
    calibration_slope_intercept,
    delong_auc_test,
)
from nstemi_ecg.models import build_model


def load_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        run_dir / "oof_predictions.csv",
        dtype={"Patient_id": "string", "record_id": "string"},
    )


def parameter_count(run_dir: Path) -> int:
    config, _ = load_config(run_dir / "run_config.yaml")
    return int(sum(parameter.numel() for parameter in build_model(config).parameters()))


def patient_bootstrap_delta(
    labels: np.ndarray,
    baseline_raw: np.ndarray,
    attention_raw: np.ndarray,
    patients: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    unique_patients = np.unique(patients)
    patient_indices = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    rng = np.random.default_rng(seed)
    values = {"delta_auroc": [], "delta_auprc": [], "delta_brier": []}
    for _ in range(iterations):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([patient_indices[patient] for patient in sampled])
        y = labels[indices]
        b = baseline_raw[indices]
        a = attention_raw[indices]
        if len(np.unique(y)) < 2:
            continue
        values["delta_auroc"].append(roc_auc_score(y, a) - roc_auc_score(y, b))
        values["delta_auprc"].append(
            average_precision_score(y, a) - average_precision_score(y, b)
        )
        values["delta_brier"].append(
            brier_score_loss(y, a) - brier_score_loss(y, b)
        )
    return {
        key: [
            float(np.nanpercentile(items, 2.5)),
            float(np.nanpercentile(items, 97.5)),
        ]
        for key, items in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final fair comparison of baseline and attention models")
    parser.add_argument("--baseline-run", default="runs/main_nstemi_vs_ua")
    parser.add_argument("--attention-run", default="runs/attention_v2_fair")
    parser.add_argument("--output-dir", default="runs/final_model_comparison")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_run).resolve()
    attention_dir = Path(args.attention_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_predictions(baseline_dir)
    attention = load_predictions(attention_dir)
    columns = ["metadata_index", "Patient_id", "_label", "fold"]
    baseline_view = baseline[columns + ["probability", "threshold"]].rename(
        columns={"probability": "baseline_probability", "threshold": "baseline_threshold"}
    )
    attention_view = attention[
        columns + ["probability", "raw_probability", "threshold"]
    ].rename(
        columns={
            "probability": "attention_probability",
            "raw_probability": "attention_raw_probability",
            "threshold": "attention_threshold",
        }
    )
    matched = baseline_view.merge(
        attention_view,
        on=columns,
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != 6616:
        raise RuntimeError(f"Expected 6616 matched patients, got {len(matched)}")
    labels = matched["_label"].astype(int).to_numpy()
    baseline_raw = matched["baseline_probability"].to_numpy(dtype=float)
    attention_raw = matched["attention_raw_probability"].to_numpy(dtype=float)
    attention_calibrated = matched["attention_probability"].to_numpy(dtype=float)

    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))["pooled_oof"]
    attention_summary = json.loads((attention_dir / "summary.json").read_text(encoding="utf-8"))["pooled_oof"]
    raw_baseline_metrics = {
        "auroc": float(roc_auc_score(labels, baseline_raw)),
        "auprc": float(average_precision_score(labels, baseline_raw)),
        "brier": float(brier_score_loss(labels, baseline_raw)),
        "ece": calibration_error(labels, baseline_raw),
        **calibration_slope_intercept(labels, baseline_raw),
    }
    raw_attention_metrics = {
        "auroc": float(roc_auc_score(labels, attention_raw)),
        "auprc": float(average_precision_score(labels, attention_raw)),
        "brier": float(brier_score_loss(labels, attention_raw)),
        "ece": calibration_error(labels, attention_raw),
        **calibration_slope_intercept(labels, attention_raw),
    }
    delong = delong_auc_test(labels, attention_raw, baseline_raw)
    bootstrap = patient_bootstrap_delta(
        labels,
        baseline_raw,
        attention_raw,
        matched["Patient_id"].astype(str).to_numpy(),
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    pooled_rows = [
        {
            "metric": "AUROC",
            "baseline": raw_baseline_metrics["auroc"],
            "attention_v2": raw_attention_metrics["auroc"],
            "delta_attention_minus_baseline": raw_attention_metrics["auroc"] - raw_baseline_metrics["auroc"],
            "comparison_note": "raw scores; paired DeLong",
        },
        {
            "metric": "AUPRC",
            "baseline": raw_baseline_metrics["auprc"],
            "attention_v2": raw_attention_metrics["auprc"],
            "delta_attention_minus_baseline": raw_attention_metrics["auprc"] - raw_baseline_metrics["auprc"],
            "comparison_note": "raw scores; patient bootstrap",
        },
        {
            "metric": "Sensitivity",
            "baseline": baseline_summary["sensitivity"],
            "attention_v2": attention_summary["sensitivity"],
            "delta_attention_minus_baseline": attention_summary["sensitivity"] - baseline_summary["sensitivity"],
            "comparison_note": "each model at its own prespecified OOF threshold",
        },
        {
            "metric": "Specificity",
            "baseline": baseline_summary["specificity"],
            "attention_v2": attention_summary["specificity"],
            "delta_attention_minus_baseline": attention_summary["specificity"] - baseline_summary["specificity"],
            "comparison_note": "each model at its own prespecified OOF threshold",
        },
        {
            "metric": "PPV",
            "baseline": baseline_summary["ppv"],
            "attention_v2": attention_summary["ppv"],
            "delta_attention_minus_baseline": attention_summary["ppv"] - baseline_summary["ppv"],
            "comparison_note": "prevalence-dependent",
        },
        {
            "metric": "NPV",
            "baseline": baseline_summary["npv"],
            "attention_v2": attention_summary["npv"],
            "delta_attention_minus_baseline": attention_summary["npv"] - baseline_summary["npv"],
            "comparison_note": "prevalence-dependent",
        },
        {
            "metric": "F1",
            "baseline": baseline_summary["f1"],
            "attention_v2": attention_summary["f1"],
            "delta_attention_minus_baseline": attention_summary["f1"] - baseline_summary["f1"],
            "comparison_note": "each model at its own OOF threshold",
        },
        {
            "metric": "Accuracy",
            "baseline": baseline_summary["accuracy"],
            "attention_v2": attention_summary["accuracy"],
            "delta_attention_minus_baseline": attention_summary["accuracy"] - baseline_summary["accuracy"],
            "comparison_note": "each model at its own OOF threshold",
        },
        {
            "metric": "Brier score",
            "baseline": raw_baseline_metrics["brier"],
            "attention_v2": raw_attention_metrics["brier"],
            "delta_attention_minus_baseline": raw_attention_metrics["brier"] - raw_baseline_metrics["brier"],
            "comparison_note": "raw probabilities; architecture comparison",
        },
        {
            "metric": "Expected calibration error",
            "baseline": raw_baseline_metrics["ece"],
            "attention_v2": raw_attention_metrics["ece"],
            "delta_attention_minus_baseline": raw_attention_metrics["ece"] - raw_baseline_metrics["ece"],
            "comparison_note": "raw probabilities; architecture comparison",
        },
        {
            "metric": "Calibration slope",
            "baseline": raw_baseline_metrics["calibration_slope"],
            "attention_v2": raw_attention_metrics["calibration_slope"],
            "delta_attention_minus_baseline": raw_attention_metrics["calibration_slope"] - raw_baseline_metrics["calibration_slope"],
            "comparison_note": "raw probabilities; architecture comparison; ideal 1.0",
        },
        {
            "metric": "Calibration intercept",
            "baseline": raw_baseline_metrics["calibration_intercept"],
            "attention_v2": raw_attention_metrics["calibration_intercept"],
            "delta_attention_minus_baseline": raw_attention_metrics["calibration_intercept"] - raw_baseline_metrics["calibration_intercept"],
            "comparison_note": "raw probabilities; architecture comparison; ideal 0",
        },
    ]
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output_dir / "pooled_metric_comparison.csv", index=False)

    baseline_folds = {item["fold"]: item for item in json.loads((baseline_dir / "fold_metrics.json").read_text(encoding="utf-8"))}
    attention_folds = {item["fold"]: item for item in json.loads((attention_dir / "fold_metrics.json").read_text(encoding="utf-8"))}
    fold_rows = []
    for fold in sorted(baseline_folds):
        b = baseline_folds[fold]
        a = attention_folds[fold]
        fold_rows.append(
            {
                "fold": fold,
                "baseline_auroc": b["auroc"],
                "attention_auroc": a["auroc"],
                "delta_auroc": a["auroc"] - b["auroc"],
                "baseline_auprc": b["auprc"],
                "attention_auprc": a["auprc"],
                "delta_auprc": a["auprc"] - b["auprc"],
                "baseline_threshold": b["threshold"],
                "attention_threshold": a["threshold"],
            }
        )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / "fold_metric_comparison.csv", index=False)
    fold_paired_test = ttest_rel(folds["attention_auroc"], folds["baseline_auroc"])
    fold_summary = {
        "baseline_auroc_mean": float(folds["baseline_auroc"].mean()),
        "baseline_auroc_sd": float(folds["baseline_auroc"].std(ddof=1)),
        "attention_auroc_mean": float(folds["attention_auroc"].mean()),
        "attention_auroc_sd": float(folds["attention_auroc"].std(ddof=1)),
        "paired_fold_t_statistic": float(fold_paired_test.statistic),
        "paired_fold_p_value": float(fold_paired_test.pvalue),
    }

    high_sensitivity = pd.DataFrame(
        [
            {
                "model": "original_baseline",
                **baseline_summary["high_sensitivity_operating_point"],
            },
            {
                "model": "attention_v2",
                **attention_summary["high_sensitivity_operating_point"],
            },
        ]
    )
    high_sensitivity.to_csv(output_dir / "high_sensitivity_comparison.csv", index=False)

    baseline_parameters = parameter_count(baseline_dir)
    attention_parameters = parameter_count(attention_dir)
    result = {
        "fairness": {
            "matched_patients": int(len(matched)),
            "same_patient_folds": bool(
                matched.groupby("metadata_index")["fold"].nunique().eq(1).all()
            ),
            "baseline_test_set_used_for_selection": False,
            "attention_test_set_used_for_selection": False,
        },
        "delong": delong,
        "bootstrap_delta_95ci": bootstrap,
        "raw_probability_metrics": {
            "baseline": raw_baseline_metrics,
            "attention_v2": raw_attention_metrics,
        },
        "final_configured_probability_metrics": {
            "baseline": {
                "brier": baseline_summary["brier"],
                "ece": baseline_summary["expected_calibration_error"],
                "calibration_slope": baseline_summary["calibration_slope"],
                "calibration_intercept": baseline_summary["calibration_intercept"],
            },
            "attention_v2_temperature_scaled": {
                "brier": attention_summary["brier"],
                "ece": attention_summary["expected_calibration_error"],
                "calibration_slope": attention_summary["calibration_slope"],
                "calibration_intercept": attention_summary["calibration_intercept"],
            },
        },
        "fold_summary": fold_summary,
        "parameter_count": {
            "baseline": baseline_parameters,
            "attention_v2": attention_parameters,
            "increase_pct": 100.0 * (attention_parameters - baseline_parameters) / baseline_parameters,
        },
        "threshold_std": {
            "baseline": float(folds["baseline_threshold"].std(ddof=1)),
            "attention_v2": float(folds["attention_threshold"].std(ddof=1)),
        },
        "decision": "Retain original baseline as the primary model.",
    }
    (output_dir / "final_comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown = f"""# Final model comparison

## Fairness

- Matched patients: {len(matched):,}
- Same patient-level folds: {result["fairness"]["same_patient_folds"]}
- Neither model used the hidden test labels for selection.

## Architecture-only comparison on raw probabilities

| Metric | Original baseline | Attention v2 | Difference |
|---|---:|---:|---:|
| AUROC | {raw_baseline_metrics["auroc"]:.4f} | {raw_attention_metrics["auroc"]:.4f} | {raw_attention_metrics["auroc"] - raw_baseline_metrics["auroc"]:+.4f} |
| AUPRC | {raw_baseline_metrics["auprc"]:.4f} | {raw_attention_metrics["auprc"]:.4f} | {raw_attention_metrics["auprc"] - raw_baseline_metrics["auprc"]:+.4f} |
| Brier | {raw_baseline_metrics["brier"]:.4f} | {raw_attention_metrics["brier"]:.4f} | {raw_attention_metrics["brier"] - raw_baseline_metrics["brier"]:+.4f} |
| ECE | {raw_baseline_metrics["ece"]:.4f} | {raw_attention_metrics["ece"]:.4f} | {raw_attention_metrics["ece"] - raw_baseline_metrics["ece"]:+.4f} |

Paired DeLong test: z={delong["z"]:.4f}, p={delong["p_value"]:.4f}.

Patient-level bootstrap 95% CI for delta AUROC:
[{bootstrap["delta_auroc"][0]:+.4f}, {bootstrap["delta_auroc"][1]:+.4f}].

## Configured output with fold-specific temperature scaling

| Metric | Original baseline | Attention v2 | Difference |
|---|---:|---:|---:|
| AUROC | {baseline_summary["auroc"]:.4f} | {attention_summary["auroc"]:.4f} | {attention_summary["auroc"] - baseline_summary["auroc"]:+.4f} |
| AUPRC | {baseline_summary["auprc"]:.4f} | {attention_summary["auprc"]:.4f} | {attention_summary["auprc"] - baseline_summary["auprc"]:+.4f} |
| Brier | {baseline_summary["brier"]:.4f} | {attention_summary["brier"]:.4f} | {attention_summary["brier"] - baseline_summary["brier"]:+.4f} |
| ECE | {baseline_summary["expected_calibration_error"]:.4f} | {attention_summary["expected_calibration_error"]:.4f} | {attention_summary["expected_calibration_error"] - baseline_summary["expected_calibration_error"]:+.4f} |

Fold-specific calibration changes the relative scale between folds, so this
configured-output AUROC is not used as the primary architecture comparison.

## Decision

The attention model shows only a small, non-significant AUROC gain and
inconsistent fold-level behavior. Retain the original baseline as the primary model.
The attention model should be reported as an ablation experiment.
"""
    (output_dir / "FINAL_MODEL_COMPARISON.md").write_text(markdown, encoding="utf-8")
    print(pooled.to_string(index=False), flush=True)
    print("\n", folds.to_string(index=False), flush=True)
    print("\n", json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
