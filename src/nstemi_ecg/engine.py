from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import (
    ECGDataset,
    fit_clinical_scaler,
    fit_signal_scaler,
    make_dataloader,
    split_fit_validation_calibration,
)
from .metrics import (
    binary_metrics,
    fit_temperature,
    find_threshold_for_sensitivity,
    find_youden_threshold,
    sigmoid,
    softmax,
)
from .models import build_model
from .utils import choose_device, seed_everything


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0
        self.should_stop = False

    def update(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["signal"].to(device, non_blocking=True),
        batch["clinical"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
        batch["metadata_index"].to(device, non_blocking=True),
    )


def build_loss(config: dict[str, Any], labels: np.ndarray, device: torch.device) -> nn.Module:
    num_classes = int(config["data"]["num_classes"])
    use_class_weight = bool(config["training"].get("class_weight", True))
    if num_classes == 2:
        positives = float(np.sum(labels == 1))
        negatives = float(np.sum(labels == 0))
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device) if use_class_weight else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    if use_class_weight:
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.sum() * num_classes
    else:
        weights = np.ones(num_classes, dtype=np.float32)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.95))),
        eps=float(training.get("eps", 1.0e-10)),
    )
    total_steps = max(1, int(training["epochs"]) * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * float(training.get("warmup_fraction", 0.05))))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    configure: dict[str, Any],
) -> dict[str, np.ndarray]:
    model.eval()
    logits_all, labels_all, indices_all, weights_all = [], [], [], []
    use_amp = bool(configure["training"].get("amp", True)) and device.type == "cuda"
    for batch in loader:
        signal, clinical, labels, metadata_indices = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(signal, clinical if clinical.numel() else None)
        logits_all.append(output["logits"].float().cpu().numpy())
        labels_all.append(labels.float().cpu().numpy())
        indices_all.append(metadata_indices.cpu().numpy())
        weights_all.append(output["lead_weights"].float().cpu().numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    metadata_indices = np.concatenate(indices_all)
    lead_weights = np.concatenate(weights_all)
    if int(configure["data"]["num_classes"]) == 2:
        probabilities = sigmoid(logits)
    else:
        probabilities = softmax(logits)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "labels": labels,
        "metadata_indices": metadata_indices,
        "lead_weights": lead_weights,
    }


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    for batch in loader:
        signal, clinical, labels, _ = _move_batch(batch, device)
        output = model(signal, clinical if clinical.numel() else None)
        logits = output["logits"]
        target = labels.float() if int(config["data"]["num_classes"]) == 2 else labels.long()
        loss = criterion(logits, target)
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
    return total_loss / max(1, total)


def train_one_model(
    fit_df,
    validation_df,
    calibration_df,
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    fold: int | str,
    device: torch.device,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> tuple[nn.Module, dict[str, Any], dict[str, np.ndarray]]:
    seed_everything(int(config.get("seed", 42)) + (fold if isinstance(fold, int) else 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_scaler = fit_signal_scaler(fit_df, config, project_root)
    clinical_scaler = fit_clinical_scaler(fit_df, config)
    training = config["training"]

    train_dataset = ECGDataset(
        fit_df,
        config,
        project_root,
        training=True,
        signal_scaler=signal_scaler,
        clinical_scaler=clinical_scaler,
    )
    validation_dataset = ECGDataset(
        validation_df,
        config,
        project_root,
        training=False,
        signal_scaler=signal_scaler,
        clinical_scaler=clinical_scaler,
    )
    calibration_dataset = ECGDataset(
        calibration_df,
        config,
        project_root,
        training=False,
        signal_scaler=signal_scaler,
        clinical_scaler=clinical_scaler,
    )
    train_loader = make_dataloader(
        train_dataset,
        int(training["batch_size"]),
        True,
        int(training.get("num_workers", 0)),
        bool(training.get("pin_memory", True)),
    )
    validation_loader = make_dataloader(
        validation_dataset,
        int(training["eval_batch_size"]),
        False,
        int(training.get("num_workers", 0)),
        bool(training.get("pin_memory", True)),
    )
    calibration_loader = make_dataloader(
        calibration_dataset,
        int(training["eval_batch_size"]),
        False,
        int(training.get("num_workers", 0)),
        bool(training.get("pin_memory", True)),
    )

    model = build_model(config).to(device)
    if bool(training.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    criterion = build_loss(config, fit_df["_label"].to_numpy(), device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, len(train_loader))
    early_stopping = EarlyStopping(
        int(training.get("early_stopping_patience", 8)),
        float(training.get("min_delta", 5.0e-4)),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = -np.inf
    history = []
    start = time.time()

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            signal, clinical, labels, _ = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(signal, clinical if clinical.numel() else None)
                target = labels.float() if int(config["data"]["num_classes"]) == 2 else labels.long()
                loss = criterion(output["logits"], target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training.get("gradient_clip_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        validation_predictions = predict_loader(model, validation_loader, device, config)
        train_loss = running_loss / max(1, seen)
        if int(config["data"]["num_classes"]) == 2:
            try:
                validation_auc = roc_auc_score(
                    validation_predictions["labels"].astype(int),
                    validation_predictions["probabilities"],
                )
            except ValueError:
                validation_auc = float("nan")
        else:
            from sklearn.metrics import roc_auc_score as multiclass_auc

            try:
                validation_auc = multiclass_auc(
                    validation_predictions["labels"].astype(int),
                    validation_predictions["probabilities"],
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                validation_auc = float("nan")
        validation_loss = evaluate_loss(model, validation_loader, criterion, device, config)
        improved = early_stopping.update(float(validation_auc))
        if improved:
            best_score = float(validation_auc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_auroc": float(validation_auc),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"fold={fold} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_auroc={validation_auc:.4f}",
            flush=True,
        )
        if early_stopping.should_stop:
            print(f"fold={fold} early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    calibration_predictions = predict_loader(model, calibration_loader, device, config)
    temperature = 1.0
    if bool(config.get("calibration", {}).get("temperature_scaling", False)):
        temperature = fit_temperature(
            calibration_predictions["logits"].astype(float),
            calibration_predictions["labels"].astype(int),
            lower=float(config["calibration"].get("temperature_min", 0.05)),
            upper=float(config["calibration"].get("temperature_max", 10.0)),
        )
    calibration_probabilities = sigmoid(
        calibration_predictions["logits"].astype(float) / temperature
    )
    if int(config["data"]["num_classes"]) == 2:
        threshold = find_youden_threshold(
            calibration_predictions["labels"].astype(int),
            calibration_probabilities,
        )
        high_sensitivity_threshold = find_threshold_for_sensitivity(
            calibration_predictions["labels"].astype(int),
            calibration_probabilities,
            target_sensitivity=float(
                config["evaluation"].get("target_sensitivity", 0.90)
            ),
        )
    else:
        threshold = float("nan")
        high_sensitivity_threshold = float("nan")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "fold": fold,
        "threshold": threshold,
        "threshold_high_sensitivity": high_sensitivity_threshold,
        "temperature": temperature,
        "signal_scaler": signal_scaler,
        "clinical_scaler": clinical_scaler,
        "best_validation_auroc": best_score,
        "history": history,
    }
    torch.save(checkpoint, output_dir / "best.pt")
    return model, checkpoint, calibration_predictions


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def prepare_training_partitions(
    dataframe,
    config: dict[str, Any],
    fold_seed: int,
):
    training = config["training"]
    return split_fit_validation_calibration(
        dataframe,
        label_column="_label",
        validation_fraction=float(training["inner_validation_fraction"]),
        calibration_fraction=float(training["calibration_fraction"]),
        seed=fold_seed,
    )


def make_dataset(
    dataframe,
    config: dict[str, Any],
    project_root: Path,
    signal_scaler: dict[str, Any],
    clinical_scaler: dict[str, Any],
    training: bool = False,
    perturbation: dict[str, Any] | None = None,
) -> ECGDataset:
    return ECGDataset(
        dataframe,
        config,
        project_root,
        training=training,
        perturbation=perturbation,
        signal_scaler=signal_scaler,
        clinical_scaler=clinical_scaler,
    )
