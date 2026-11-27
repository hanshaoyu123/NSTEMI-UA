#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstemi_ecg.config import load_config, save_config
from nstemi_ecg.data import build_cohort_dataframe, load_metadata, make_dataloader
from nstemi_ecg.engine import (
    load_checkpoint,
    make_dataset,
    predict_loader,
    prepare_training_partitions,
)
from nstemi_ecg.metrics import (
    binary_metrics,
    bootstrap_ci,
    calibration_error,
    calibration_slope_intercept,
    fit_temperature,
    find_threshold_for_sensitivity,
    find_youden_threshold,
    sigmoid,
)
from nstemi_ecg.plots import plot_binary_evaluation
from nstemi_ecg.utils import choose_device, seed_everything, write_json


def calibrate_checkpoint(
    checkpoint_path: Path,
    calibration_df: pd.DataFrame,
    config: dict,
    project_root: Path,
    device,
    output_checkpoint: Path,
) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    calibration_df = calibration_df.reset_index(drop=True)
    cal_dataset = make_dataset(
        calibration_df,
        checkpoint["config"],
        project_root,
        checkpoint["signal_scaler"],
        checkpoint["clinical_scaler"],
        training=False,
    )
    cal_loader = make_dataloader(
        cal_dataset,
        int(checkpoint["config"]["training"]["eval_batch_size"]),
        False,
        int(checkpoint["config"]["training"].get("num_workers", 0)),
        bool(checkpoint["config"]["training"].get("pin_memory", True)),
    )
    predictions = predict_loader(model, cal_loader, device, checkpoint["config"])
    labels = predictions["labels"].astype(int)
    temperature = fit_temperature(
        predictions["logits"].astype(float),
        labels,
        lower=float(config.get("calibration", {}).get("temperature_min", 0.05)),
        upper=float(config.get("calibration", {}).get("temperature_max", 10.0)),
    )
    calibrated = sigmoid(predictions["logits"].astype(float) / temperature)
    threshold = find_youden_threshold(labels, calibrated)
    high_sensitivity_threshold = find_threshold_for_sensitivity(
        labels,
        calibrated,
        target_sensitivity=float(config["evaluation"].get("target_sensitivity", 0.90)),
    )
    checkpoint["temperature"] = temperature
    checkpoint["threshold"] = threshold
    checkpoint["threshold_high_sensitivity"] = high_sensitivity_threshold
    checkpoint["calibration_method"] = "temperature_scaling"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(checkpoint, output_checkpoint)
    return {
        "temperature": temperature,
        "threshold": threshold,
        "threshold_high_sensitivity": high_sensitivity_threshold,
        "calibration_size": int(len(labels)),
        "raw_brier": float(
            np.mean((predictions["probabilities"] - labels) ** 2)
        ),
        "calibrated_brier": float(np.mean((calibrated - labels) ** 2)),
    }


def predict_outer_validation(
    checkpoint_path: Path,
    validation_df: pd.DataFrame,
    project_root: Path,
    device,
) -> tuple[pd.DataFrame, dict]:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    validation_df = validation_df.reset_index(drop=True)
    dataset = make_dataset(
        validation_df,
        checkpoint["config"],
        project_root,
        checkpoint["signal_scaler"],
        checkpoint["clinical_scaler"],
        training=False,
    )
    loader = make_dataloader(
        dataset,
        int(checkpoint["config"]["training"]["eval_batch_size"]),
        False,
        int(checkpoint["config"]["training"].get("num_workers", 0)),
        bool(checkpoint["config"]["training"].get("pin_memory", True)),
    )
    predictions = predict_loader(model, loader, device, checkpoint["config"])
    raw = predictions["probabilities"]
    calibrated = sigmoid(predictions["logits"].astype(float) / float(checkpoint["temperature"]))
    result = validation_df.copy()
    result["logit"] = predictions["logits"]
    result["raw_probability"] = raw
    result["probability"] = calibrated
    result["temperature"] = float(checkpoint["temperature"])
    result["threshold"] = float(checkpoint["threshold"])
    result["threshold_high_sensitivity"] = float(checkpoint["threshold_high_sensitivity"])
    result["prediction"] = (calibrated >= float(checkpoint["threshold"])).astype(int)
    result["prediction_high_sensitivity"] = (
        calibrated >= float(checkpoint["threshold_high_sensitivity"])
    ).astype(int)
    return result, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc temperature calibration of an existing CV run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config, project_root = load_config(run_dir / "run_config.yaml")
    config["calibration"] = {
        "temperature_scaling": True,
        "temperature_min": 0.05,
        "temperature_max": 10.0,
    }
    save_config(config, output_dir / "run_config.yaml")
    seed_everything(int(config.get("seed", 42)))
    device = choose_device(args.device)

    metadata = load_metadata(config, project_root)
    dataframe = build_cohort_dataframe(metadata, config, "train")
    dataframe["fold"] = dataframe["cv_fold"].astype(int)
    oof_parts = []
    fold_results = []
    calibration_results = []
    for fold in sorted(dataframe["fold"].unique()):
        fold = int(fold)
        outer_train = dataframe.loc[dataframe["fold"].ne(fold)].reset_index(drop=True)
        outer_validation = dataframe.loc[dataframe["fold"].eq(fold)].reset_index(drop=True)
        _, _, calibration_df = prepare_training_partitions(
            outer_train,
            config,
            fold_seed=int(config.get("seed", 42)) + fold,
        )
        calibrated_checkpoint = output_dir / f"fold_{fold}" / "best.pt"
        calibration_info = calibrate_checkpoint(
            run_dir / f"fold_{fold}" / "best.pt",
            calibration_df,
            config,
            project_root,
            device,
            calibrated_checkpoint,
        )
        calibration_info["fold"] = fold
        calibration_results.append(calibration_info)
        validation_predictions, _ = predict_outer_validation(
            calibrated_checkpoint,
            outer_validation,
            project_root,
            device,
        )
        validation_predictions.to_csv(output_dir / f"fold_{fold}" / "validation_predictions.csv", index=False)
        metrics = binary_metrics(
            validation_predictions["_label"].astype(int).to_numpy(),
            validation_predictions["probability"].to_numpy(),
            float(validation_predictions["threshold"].iloc[0]),
        )
        metrics["fold"] = fold
        fold_results.append(metrics)
        write_json(metrics, output_dir / f"fold_{fold}" / "metrics.json")
        oof_parts.append(validation_predictions)
        print(
            f"fold={fold} temperature={calibration_info['temperature']:.4f} "
            f"threshold={calibration_info['threshold']:.4f} AUROC={metrics['auroc']:.4f}",
            flush=True,
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    labels = oof["_label"].astype(int).to_numpy()
    probabilities = oof["probability"].to_numpy(dtype=float)
    thresholds = oof.groupby("fold")["threshold"].first().to_numpy()
    pooled_threshold = float(np.median(thresholds))
    summary = binary_metrics(labels, probabilities, pooled_threshold)
    summary["expected_calibration_error"] = calibration_error(labels, probabilities)
    summary.update(calibration_slope_intercept(labels, probabilities))
    summary["bootstrap_95ci"] = bootstrap_ci(
        labels,
        probabilities,
        oof["Patient_id"].to_numpy(),
        pooled_threshold,
        iterations=int(config["evaluation"].get("bootstrap_iterations", 2000)),
        seed=int(config.get("seed", 42)),
    )
    summary["pooled_threshold"] = pooled_threshold
    summary["fold_thresholds"] = thresholds.tolist()
    high_threshold = float(np.median(oof.groupby("fold")["threshold_high_sensitivity"].first().to_numpy()))
    summary["high_sensitivity_operating_point"] = binary_metrics(
        labels,
        probabilities,
        high_threshold,
    )
    write_json({"pooled_oof": summary, "folds": fold_results}, output_dir / "summary.json")
    pd.DataFrame(calibration_results).to_csv(output_dir / "calibration_results.csv", index=False)
    write_json(calibration_results, output_dir / "calibration_results.json")
    plot_binary_evaluation(
        labels,
        probabilities,
        pooled_threshold,
        output_dir / "evaluation",
        calibration_bins=int(config["evaluation"].get("reliability_bins", 10)),
    )

    final_checkpoint = run_dir / "final_model" / "best.pt"
    if final_checkpoint.is_file():
        _, _, final_calibration = prepare_training_partitions(
            dataframe,
            config,
            fold_seed=int(config.get("seed", 42)) + 104729,
        )
        final_info = calibrate_checkpoint(
            final_checkpoint,
            final_calibration,
            config,
            project_root,
            device,
            output_dir / "final_model" / "best.pt",
        )
        write_json(final_info, output_dir / "final_model" / "calibration.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

