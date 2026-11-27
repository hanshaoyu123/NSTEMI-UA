from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, resolve_path, save_config
from .data import (
    ECGDataset,
    assign_patient_folds,
    build_cohort_dataframe,
    fit_clinical_scaler,
    load_metadata,
    make_dataloader,
)
from .engine import (
    load_checkpoint,
    make_dataset,
    predict_loader,
    prepare_training_partitions,
    train_one_model,
)
from .metrics import (
    binary_metrics,
    bootstrap_ci,
    calibration_slope_intercept,
    calibration_error,
    multiclass_metrics,
)
from .plots import plot_binary_evaluation, plot_fold_metrics
from .utils import choose_device, seed_everything, write_json


def _override_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = copy.deepcopy(config)
    if args.seed is not None:
        result["seed"] = int(args.seed)
    if getattr(args, "epochs", None) is not None:
        result["training"]["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        result["training"]["batch_size"] = int(args.batch_size)
    if getattr(args, "num_workers", None) is not None:
        result["training"]["num_workers"] = int(args.num_workers)
    if getattr(args, "amp", None) is not None:
        result["training"]["amp"] = bool(args.amp)
    if getattr(args, "compile", None) is not None:
        result["training"]["compile"] = bool(args.compile)
    if getattr(args, "no_final", False):
        result["training"]["train_final_model"] = False
    return result


def _cap_dataframe(dataframe: pd.DataFrame, max_records: int | None, seed: int) -> pd.DataFrame:
    if max_records is None or max_records >= len(dataframe):
        return dataframe
    parts = []
    labels = sorted(dataframe["_label"].dropna().unique())
    per_class = max(1, max_records // max(1, len(labels)))
    for label in labels:
        subset = dataframe.loc[dataframe["_label"].eq(label)]
        n = min(per_class, len(subset))
        parts.append(subset.sample(n=n, random_state=seed))
    capped = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return capped.reset_index(drop=True)


def _prepare_training_frame(
    config: dict[str, Any],
    project_root: Path,
    fold_count: int,
) -> pd.DataFrame:
    metadata = load_metadata(config, project_root)
    dataframe = build_cohort_dataframe(metadata, config, "train")
    if bool(config["data"].get("use_precomputed_folds", False)):
        if (dataframe["cv_fold"] < 0).any():
            raise ValueError("Precomputed folds requested but one or more rows have cv_fold=-1")
        dataframe["fold"] = dataframe["cv_fold"].astype(int)
    else:
        dataframe = assign_patient_folds(
            dataframe,
            "_label",
            n_splits=fold_count,
            seed=int(config.get("seed", 42)),
        )
    return dataframe


def _save_oof_columns(
    rows: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    threshold: float,
    high_sensitivity_threshold: float,
    temperature: float,
    fold: int,
    num_classes: int,
) -> pd.DataFrame:
    result = rows.copy()
    result["fold"] = fold
    result["threshold"] = threshold
    result["threshold_high_sensitivity"] = high_sensitivity_threshold
    result["temperature"] = temperature
    if num_classes == 2:
        result["logit"] = predictions["logits"]
        result["raw_probability"] = predictions["probabilities"]
        calibrated_probability = 1.0 / (
            1.0 + np.exp(-np.clip(predictions["logits"].astype(float) / temperature, -50.0, 50.0))
        )
        result["probability"] = calibrated_probability
        result["prediction"] = (calibrated_probability >= threshold).astype(int)
        result["prediction_high_sensitivity"] = (
            calibrated_probability >= high_sensitivity_threshold
        ).astype(int)
    else:
        for class_index in range(num_classes):
            result[f"probability_{class_index}"] = predictions["probabilities"][:, class_index]
        result["prediction"] = predictions["probabilities"].argmax(axis=1)
        result["logit"] = float("nan")
        result["probability"] = float("nan")
    return result


def _evaluate_binary_pooled(
    oof: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    labels = oof["_label"].astype(int).to_numpy()
    probabilities = oof["probability"].to_numpy()
    thresholds = oof.groupby("fold")["threshold"].first().to_numpy()
    pooled_threshold = float(np.median(thresholds))
    metrics = binary_metrics(labels, probabilities, pooled_threshold)
    metrics["expected_calibration_error"] = calibration_error(
        labels,
        probabilities,
        bins=int(config["evaluation"].get("reliability_bins", 10)),
    )
    metrics.update(calibration_slope_intercept(labels, probabilities))
    metrics["bootstrap_95ci"] = bootstrap_ci(
        labels,
        probabilities,
        oof["Patient_id"].to_numpy(),
        pooled_threshold,
        iterations=int(config["evaluation"].get("bootstrap_iterations", 2000)),
        seed=int(config.get("seed", 42)),
    )
    metrics["pooled_threshold"] = pooled_threshold
    metrics["fold_thresholds"] = thresholds.tolist()
    high_threshold = float(
        np.median(oof.groupby("fold")["threshold_high_sensitivity"].first().to_numpy())
    )
    metrics["high_sensitivity_operating_point"] = binary_metrics(
        labels,
        probabilities,
        high_threshold,
    )
    plot_binary_evaluation(
        labels,
        probabilities,
        pooled_threshold,
        output_dir / "evaluation",
        calibration_bins=int(config["evaluation"].get("reliability_bins", 10)),
    )
    return metrics


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train patient-level ECG models with nested calibration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config, project_root = load_config(args.config)
    config = _override_config(config, args)
    seed_everything(int(config.get("seed", 42)))
    device = choose_device(args.device)
    run_dir = resolve_path(project_root, args.output_dir or config["output_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        if not (args.max_train_batches or args.resume):
            raise FileExistsError(f"Run directory already exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "run_config.yaml")
    write_json(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "arguments": {key: str(value) for key, value in vars(args).items()},
        },
        run_dir / "run_metadata.json",
    )

    dataframe = _prepare_training_frame(config, project_root, args.folds)
    dataframe = _cap_dataframe(dataframe, args.max_records, int(config.get("seed", 42)))
    available_folds = sorted(int(value) for value in dataframe["fold"].unique())
    if args.max_folds is not None:
        available_folds = available_folds[: args.max_folds]
    if not available_folds:
        raise RuntimeError("No folds are available after data filtering")
    print(f"Training rows={len(dataframe):,}; folds={available_folds}; device={device}", flush=True)

    oof_parts, fold_metrics = [], []
    thresholds = []
    for fold in available_folds:
        outer_train = dataframe.loc[dataframe["fold"].ne(fold)].reset_index(drop=True)
        outer_validation = dataframe.loc[dataframe["fold"].eq(fold)].reset_index(drop=True)
        fit_df, validation_df, calibration_df = prepare_training_partitions(
            outer_train,
            config,
            fold_seed=int(config.get("seed", 42)) + fold,
        )
        fold_dir = run_dir / f"fold_{fold}"
        if args.resume and (fold_dir / "best.pt").is_file():
            model, checkpoint = load_checkpoint(fold_dir / "best.pt", device)
            print(f"Resuming fold {fold} from {fold_dir / 'best.pt'}", flush=True)
        else:
            model, checkpoint, calibration_predictions = train_one_model(
                fit_df,
                validation_df,
                calibration_df,
                config,
                project_root,
                fold_dir,
                fold,
                device,
                max_train_batches=args.max_train_batches,
            )
            pd.DataFrame(checkpoint["history"]).to_csv(fold_dir / "training_history.csv", index=False)
            calibration_frame = pd.DataFrame(
                {
                    "metadata_index": calibration_predictions["metadata_indices"],
                    "label": calibration_predictions["labels"].astype(int),
                    "logit": calibration_predictions["logits"],
                    "raw_probability": calibration_predictions["probabilities"],
                    "temperature": float(checkpoint["temperature"]),
                }
            )
            calibration_frame["calibrated_probability"] = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        calibration_frame["logit"].to_numpy(dtype=float)
                        / float(checkpoint["temperature"]),
                        -50.0,
                        50.0,
                    )
                )
            )
            calibration_frame.to_csv(fold_dir / "calibration_predictions.csv", index=False)

        signal_scaler = checkpoint["signal_scaler"]
        clinical_scaler = checkpoint["clinical_scaler"]
        validation_dataset = make_dataset(
            outer_validation,
            config,
            project_root,
            signal_scaler,
            clinical_scaler,
            training=False,
        )
        validation_loader = make_dataloader(
            validation_dataset,
            int(config["training"]["eval_batch_size"]),
            False,
            int(config["training"].get("num_workers", 0)),
            bool(config["training"].get("pin_memory", True)),
        )
        predictions = predict_loader(model, validation_loader, device, config)
        temperature = float(checkpoint.get("temperature", 1.0))
        calibrated_probability = 1.0 / (
            1.0 + np.exp(-np.clip(predictions["logits"].astype(float) / temperature, -50.0, 50.0))
        )
        oof_part = _save_oof_columns(
            outer_validation,
            predictions,
            float(checkpoint["threshold"]),
            float(checkpoint.get("threshold_high_sensitivity", checkpoint["threshold"])),
            temperature,
            fold,
            int(config["data"]["num_classes"]),
        )
        oof_part.to_csv(fold_dir / "validation_predictions.csv", index=False)
        oof_parts.append(oof_part)
        thresholds.append(float(checkpoint["threshold"]))
        if int(config["data"]["num_classes"]) == 2:
            fold_result = binary_metrics(
                predictions["labels"].astype(int),
                calibrated_probability,
                float(checkpoint["threshold"]),
            )
        else:
            fold_result = multiclass_metrics(
                predictions["labels"].astype(int),
                predictions["probabilities"],
                int(config["data"]["num_classes"]),
            )
        fold_result["fold"] = fold
        fold_metrics.append(fold_result)
        write_json(fold_result, fold_dir / "metrics.json")

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(fold_metrics, run_dir / "fold_metrics.json")

    if int(config["data"]["num_classes"]) == 2:
        summary = _evaluate_binary_pooled(oof, config, run_dir)
        plot_fold_metrics(fold_metrics, run_dir / "evaluation")
    else:
        labels = oof["_label"].astype(int).to_numpy()
        probability_columns = [
            f"probability_{index}" for index in range(int(config["data"]["num_classes"]))
        ]
        summary = multiclass_metrics(
            labels,
            oof[probability_columns].to_numpy(),
            int(config["data"]["num_classes"]),
        )
    write_json({"pooled_oof": summary, "folds": fold_metrics}, run_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)

    if bool(config["training"].get("train_final_model", True)) and args.max_train_batches is None:
        final_path = run_dir / "final_model" / "best.pt"
        if args.resume and final_path.is_file():
            print(f"Keeping existing final model: {final_path}", flush=True)
        else:
            fit_df, validation_df, calibration_df = prepare_training_partitions(
                dataframe,
                config,
                fold_seed=int(config.get("seed", 42)) + 104729,
            )
            train_one_model(
                fit_df,
                validation_df,
                calibration_df,
                config,
                project_root,
                run_dir / "final_model",
                "final",
                device,
            )
            print(f"Final model saved: {final_path}", flush=True)


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Predict the hidden-label official test set")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--all-test-records",
        action="store_true",
        help="Predict every test ECG row, including multiple records from the same patient.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config, project_root = load_config(run_dir / "run_config.yaml")
    device = choose_device(args.device)
    metadata = load_metadata(config, project_root)
    if args.all_test_records:
        test_df = metadata.loc[metadata["split"].eq("test")].copy().reset_index(drop=True)
        test_df["_label"] = float("nan")
    else:
        test_df = build_cohort_dataframe(metadata, config, "test")
    if args.max_records is not None:
        test_df = test_df.head(args.max_records).copy()

    checkpoint_paths = sorted(run_dir.glob("fold_*/best.pt"))
    final_checkpoint = run_dir / "final_model" / "best.pt"
    if final_checkpoint.is_file():
        checkpoint_paths = [final_checkpoint]
    if not checkpoint_paths:
        raise FileNotFoundError("No fold or final checkpoints found")

    probability_sum = None
    thresholds = []
    high_sensitivity_thresholds = []
    lead_weight_sum = None
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        dataset = make_dataset(
            test_df,
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
        temperature = float(checkpoint.get("temperature", 1.0))
        if predictions["probabilities"].ndim == 2:
            probabilities = predictions["probabilities"]
        else:
            probabilities = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        predictions["logits"].astype(float) / temperature,
                        -50.0,
                        50.0,
                    )
                )
            )
        if probability_sum is None:
            probability_sum = probabilities.astype(np.float64)
            lead_weight_sum = predictions["lead_weights"].astype(np.float64)
        else:
            probability_sum += probabilities.astype(np.float64)
            lead_weight_sum += predictions["lead_weights"].astype(np.float64)
        thresholds.append(float(checkpoint.get("threshold", np.nan)))
        high_sensitivity_thresholds.append(
            float(checkpoint.get("threshold_high_sensitivity", np.nan))
        )

    probabilities = probability_sum / len(checkpoint_paths)
    lead_weights = lead_weight_sum / len(checkpoint_paths)
    threshold = float(np.nanmedian(thresholds))
    high_sensitivity_threshold = float(np.nanmedian(high_sensitivity_thresholds))
    output = pd.DataFrame(
        {
            "Patient_id": test_df["Patient_id"].to_numpy(),
            "ecg_row_record": test_df["ecg_row_record"].to_numpy(),
            "record_id": test_df["record_id"].to_numpy(),
            "split": test_df["split"].to_numpy(),
            "threshold": threshold,
            "threshold_high_sensitivity": high_sensitivity_threshold,
        }
    )
    if int(config["data"]["num_classes"]) == 2:
        output["probability"] = probabilities
        output["prediction"] = (probabilities >= threshold).astype(int)
        output["prediction_high_sensitivity"] = (
            probabilities >= high_sensitivity_threshold
        ).astype(int)
        submission = output[["Patient_id", "ecg_row_record", "prediction"]].copy()
    else:
        for class_index in range(int(config["data"]["num_classes"])):
            output[f"probability_{class_index}"] = probabilities[:, class_index]
        output["prediction"] = probabilities.argmax(axis=1)
        submission = output[["Patient_id", "ecg_row_record", "prediction"]].copy()

    output_path = Path(args.output) if args.output else run_dir / "test_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    submission.to_csv(output_path.with_name(output_path.stem + "_submission.csv"), index=False)
    np.save(output_path.with_suffix(".lead_weights.npy"), lead_weights)
    print(f"Predictions: {output_path}", flush=True)


def _default_perturbations() -> list[dict[str, Any]]:
    return [
        {"name": "clean"},
        {"name": "gaussian_noise", "sigma_mv": 0.02},
        {"name": "gaussian_noise", "sigma_mv": 0.05},
        {"name": "gaussian_noise", "sigma_mv": 0.10},
        {"name": "baseline_wander", "amplitude_mv": 0.10, "frequency_hz": 0.15},
        {"name": "baseline_wander", "amplitude_mv": 0.30, "frequency_hz": 0.30},
        {"name": "amplitude_scale", "factor": 0.80},
        {"name": "amplitude_scale", "factor": 1.20},
        {"name": "time_shift", "seconds": 0.50},
        {"name": "time_shift", "seconds": 1.00},
        {"name": "lead_dropout", "count": 1},
        {"name": "lead_dropout", "count": 3},
        {"name": "limb_lead_dropout"},
        {"name": "chest_lead_dropout"},
        {"name": "powerline_50hz", "amplitude_mv": 0.05},
        {"name": "quantize", "step_mv": 0.01},
    ]


def _scenario_name(spec: dict[str, Any]) -> str:
    name = spec["name"]
    details = [
        f"{key}={value}"
        for key, value in spec.items()
        if key not in {"name", "sampling_rate_hz"}
    ]
    return name + ("__" + "__".join(details) if details else "")


def robustness_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained fold models under signal perturbations")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config, project_root = load_config(run_dir / "run_config.yaml")
    if int(config["data"]["num_classes"]) != 2:
        raise NotImplementedError("Robustness binary metrics are currently implemented for num_classes=2")
    device = choose_device(args.device)
    oof = pd.read_csv(
        run_dir / "oof_predictions.csv",
        dtype={"Patient_id": "string", "record_id": "string"},
    )
    if args.max_records is not None:
        oof = (
            oof.groupby("_label", group_keys=False)
            .head(max(1, args.max_records // max(1, oof["_label"].nunique())))
            .reset_index(drop=True)
        )
    fold_paths = {int(path.parent.name.split("_")[-1]): path for path in run_dir.glob("fold_*/best.pt")}
    if not fold_paths:
        raise FileNotFoundError("No fold checkpoints found")
    fold_thresholds = {
        fold: float(torch.load(path, map_location="cpu", weights_only=False)["threshold"])
        for fold, path in fold_paths.items()
    }
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    for spec in _default_perturbations():
        scenario = _scenario_name(spec)
        all_probabilities, all_labels, all_groups, all_folds = [], [], [], []
        for fold, fold_rows in oof.groupby("fold"):
            fold = int(fold)
            if fold not in fold_paths:
                continue
            model, checkpoint = load_checkpoint(fold_paths[fold], device)
            dataset = make_dataset(
                fold_rows.reset_index(drop=True),
                checkpoint["config"],
                project_root,
                checkpoint["signal_scaler"],
                checkpoint["clinical_scaler"],
                training=False,
                perturbation=spec,
            )
            loader = make_dataloader(
                dataset,
                int(checkpoint["config"]["training"]["eval_batch_size"]),
                False,
                int(checkpoint["config"]["training"].get("num_workers", 0)),
                bool(checkpoint["config"]["training"].get("pin_memory", True)),
            )
            predictions = predict_loader(model, loader, device, checkpoint["config"])
            temperature = float(checkpoint.get("temperature", 1.0))
            all_probabilities.append(
                1.0
                / (
                    1.0
                    + np.exp(
                        -np.clip(
                            predictions["logits"].astype(float) / temperature,
                            -50.0,
                            50.0,
                        )
                    )
                )
            )
            all_labels.append(fold_rows["_label"].astype(int).to_numpy())
            all_groups.append(fold_rows["Patient_id"].to_numpy())
            all_folds.append(np.full(len(fold_rows), fold, dtype=int))
        probabilities = np.concatenate(all_probabilities)
        labels = np.concatenate(all_labels)
        groups = np.concatenate(all_groups)
        folds = np.concatenate(all_folds)
        threshold = float(
            np.nanmedian([fold_thresholds[int(fold)] for fold in np.unique(folds)])
        )
        metrics = binary_metrics(labels, probabilities, threshold)
        metrics["scenario"] = scenario
        metrics["threshold"] = threshold
        detail_rows.append(metrics)
        print(f"{scenario}: AUROC={metrics['auroc']:.4f}, sensitivity={metrics['sensitivity']:.4f}", flush=True)

    details = pd.DataFrame(detail_rows)
    details.to_csv(output_dir / "robustness_metrics.csv", index=False)
    write_json(details.to_dict(orient="records"), output_dir / "robustness_summary.json")

    fig, axis = plt.subplots(figsize=(12, 4.8))
    positions = np.arange(len(details))
    axis.plot(positions, details["auroc"], marker="o", color="#B94A48")
    axis.set_xticks(positions)
    axis.set_xticklabels(details["scenario"], rotation=55, ha="right")
    axis.set_ylim(max(0.0, details["auroc"].min() - 0.1), 1.0)
    axis.set_ylabel("AUROC")
    axis.set_title("Signal perturbation stress test")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_auroc.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "robustness_auroc.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Robustness output: {output_dir}", flush=True)


def _integrated_gradients(
    model: torch.nn.Module,
    signal: torch.Tensor,
    clinical: torch.Tensor | None,
    steps: int = 24,
) -> np.ndarray:
    model.eval()
    baseline = torch.zeros_like(signal)
    alphas = torch.linspace(0.0, 1.0, steps, device=signal.device).view(steps, 1, 1, 1)
    interpolated = (baseline.unsqueeze(0) + alphas * (signal - baseline).unsqueeze(0)).squeeze(1)
    interpolated.requires_grad_(True)
    clinical_batch = None
    if clinical is not None and clinical.numel():
        clinical_batch = clinical.unsqueeze(0).expand(steps, -1)
    logits = model(interpolated, clinical_batch)["logits"]
    if logits.ndim == 1:
        objective = logits.sum() if float(logits.mean().detach()) >= 0 else -logits.sum()
    else:
        predicted_class = int(logits.mean(dim=0).argmax().item())
        objective = logits[:, predicted_class].sum()
    model.zero_grad(set_to_none=True)
    gradients = torch.autograd.grad(objective, interpolated)[0]
    average_gradients = gradients.mean(dim=0)
    importance = ((signal - baseline) * average_gradients).abs().detach().cpu().numpy()[0]
    return importance


def interpret_main() -> None:
    parser = argparse.ArgumentParser(description="Compute integrated-gradients ECG explanations")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--ig-steps", type=int, default=24)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config, project_root = load_config(run_dir / "run_config.yaml")
    device = choose_device(args.device)
    oof = pd.read_csv(
        run_dir / "oof_predictions.csv",
        dtype={"Patient_id": "string", "record_id": "string"},
    )
    if len(oof) > args.max_records:
        parts = []
        per_class = max(1, args.max_records // max(1, oof["_label"].nunique()))
        for _, subset in oof.groupby("_label"):
            parts.append(subset.head(per_class))
        oof = pd.concat(parts, ignore_index=True)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_paths = {int(path.parent.name.split("_")[-1]): path for path in run_dir.glob("fold_*/best.pt")}
    importance_sum = None
    lead_rows = []
    count = 0
    for fold, rows in oof.groupby("fold"):
        fold = int(fold)
        if fold not in fold_paths:
            continue
        model, checkpoint = load_checkpoint(fold_paths[fold], device)
        dataset = make_dataset(
            rows.reset_index(drop=True),
            checkpoint["config"],
            project_root,
            checkpoint["signal_scaler"],
            checkpoint["clinical_scaler"],
            training=False,
        )
        for row_index in range(len(dataset)):
            item = dataset[row_index]
            signal = item["signal"].unsqueeze(0).to(device)
            clinical = item["clinical"].unsqueeze(0).to(device) if item["clinical"].numel() else None
            importance = _integrated_gradients(model, signal, clinical, steps=args.ig_steps)
            normalized = importance / max(float(importance.max()), 1.0e-8)
            importance_sum = normalized if importance_sum is None else importance_sum + normalized
            lead_importance = normalized.sum(axis=1)
            row = rows.iloc[row_index]
            for lead_index, value in enumerate(lead_importance, start=1):
                lead_rows.append(
                    {
                        "record_id": row["record_id"],
                        "label": int(row["_label"]),
                        "lead": f"L{lead_index}",
                        "importance": float(value),
                    }
                )
            count += 1
    if importance_sum is None:
        raise RuntimeError("No explanations were generated")
    average_importance = importance_sum / count
    pd.DataFrame(lead_rows).to_csv(output_dir / "lead_importance.csv", index=False)

    fig, axis = plt.subplots(figsize=(13, 4.6))
    image = axis.imshow(
        average_importance,
        aspect="auto",
        cmap="magma",
        extent=[0, int(config["data"]["input_length"]) / float(config["data"]["sampling_rate_hz"]), 12, 0],
    )
    axis.set(
        xlabel="time (s)",
        ylabel="lead index",
        title=f"Average integrated-gradients importance (n={count})",
        yticks=np.arange(12) + 0.5,
        yticklabels=[f"L{index}" for index in range(1, 13)],
    )
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "average_saliency_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "average_saliency_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    np.save(output_dir / "average_importance.npy", average_importance)
    print(f"Interpretability output: {output_dir}", flush=True)
