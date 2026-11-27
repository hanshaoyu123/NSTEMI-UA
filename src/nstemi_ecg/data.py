from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import resolve_path
from .transforms import augment_ecg, perturb_ecg


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _parse_header(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    first = lines[0].split()
    signal_count = int(first[1])
    sampling_rate = float(first[2])
    lines = lines[1 : signal_count + 1]
    gains, baselines, formats, names = [], [], [], []
    for line in lines:
        parts = line.split()
        gain_text, _ = parts[2].split("/", maxsplit=1)
        if "(" in gain_text:
            gain_text, baseline_text = gain_text.split("(", maxsplit=1)
            baseline = int(baseline_text.rstrip(")"))
        else:
            baseline = int(parts[4])
        gains.append(float(gain_text))
        baselines.append(baseline)
        formats.append(int(parts[1].split("x")[0]))
        names.append(parts[-1])
    return {
        "signal_count": signal_count,
        "sampling_rate": sampling_rate,
        "gains": np.asarray(gains, dtype=np.float32),
        "baselines": np.asarray(baselines, dtype=np.float32),
        "formats": formats,
        "names": names,
    }


def _read_wfdb_format16(path: Path) -> np.ndarray:
    header = _parse_header(path.with_suffix(".hea"))
    if header["signal_count"] != len(LEADS) or any(fmt != 16 for fmt in header["formats"]):
        raise ValueError(f"Unsupported WFDB record: {path}")
    raw = np.fromfile(path, dtype="<i2")
    frames = raw.size // header["signal_count"]
    raw = raw[: frames * header["signal_count"]].reshape(frames, header["signal_count"]).T
    return ((raw.astype(np.float32) - header["baselines"][:, None]) / header["gains"][:, None])


def _bandpass(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1, padlen=min(300, signal.shape[-1] - 1)).astype(np.float32)


class ECGDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, Any],
        project_root: Path,
        training: bool = False,
        perturbation: dict[str, Any] | None = None,
        signal_scaler: dict[str, Any] | None = None,
        clinical_scaler: dict[str, Any] | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.config = config
        self.project_root = project_root
        self.training = training
        self.perturbation = perturbation or {"name": "clean"}
        self.signal_scaler = signal_scaler
        self.clinical_scaler = clinical_scaler
        self.memmaps: dict[Path, np.ndarray] = {}
        self.data_config = config["data"]
        self.use_clinical = bool(self.data_config.get("use_clinical", False))
        self.clinical_columns = list(self.data_config.get("clinical_columns", []))
        self.label_column = self.data_config["label_column"]
        self.num_classes = int(self.data_config["num_classes"])

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_cached_ecg(self, row: pd.Series) -> np.ndarray:
        split = row["split"]
        template = self.data_config["signal_file_template"]
        path = resolve_path(self.project_root, self.data_config["signals_dir"]) / template.format(split=split)
        if path not in self.memmaps:
            if not path.is_file():
                raise FileNotFoundError(f"Signal memmap not found: {path}")
            self.memmaps[path] = np.load(path, mmap_mode="r")
        return np.asarray(self.memmaps[path][int(row["source_row_index"])], dtype=np.float32)

    def _load_raw_ecg(self, row: pd.Series) -> np.ndarray:
        raw_path = resolve_path(self.project_root, self.data_config["raw_data_dir"]) / row["ecg_row_record"]
        signal = _read_wfdb_format16(raw_path)
        if bool(self.data_config.get("raw_filter", True)):
            signal = _bandpass(signal, int(self.data_config.get("sampling_rate_hz", 500)))
        return np.clip(signal, -5.0, 5.0).astype(np.float32)

    def _load_median_raw(self, row: pd.Series) -> np.ndarray:
        path = resolve_path(self.project_root, self.data_config["med_data_dir"]) / row["ecg_med_record"]
        header_path = resolve_path(self.project_root, self.data_config["raw_data_dir"]) / f"{row['record_id']}.hea"
        header = _parse_header(header_path)
        raw = np.fromfile(path, dtype="<i2")
        frames = raw.size // len(LEADS)
        raw = raw[: frames * len(LEADS)].reshape(frames, len(LEADS)).T.astype(np.float32)
        signal = (raw - header["baselines"][:, None]) / header["gains"][:, None]
        if bool(self.data_config.get("raw_filter", True)):
            signal = _bandpass(signal, int(self.data_config.get("sampling_rate_hz", 500)))
        return np.clip(signal, -5.0, 5.0).astype(np.float32)

    def load_signal(self, row: pd.Series) -> np.ndarray:
        kind = self.data_config.get("signal_kind", "ecg")
        if kind == "ecg":
            signal = self._load_cached_ecg(row)
        elif kind == "median":
            signal = self._load_cached_ecg(row)
        elif kind == "raw":
            signal = self._load_raw_ecg(row)
        elif kind == "median_raw":
            signal = self._load_median_raw(row)
        else:
            raise ValueError(f"Unknown signal kind: {kind}")
        target_length = int(self.data_config["input_length"])
        if signal.shape[1] != target_length:
            fixed = np.zeros((signal.shape[0], target_length), dtype=np.float32)
            width = min(target_length, signal.shape[1])
            fixed[:, :width] = signal[:, :width]
            signal = fixed
        return signal

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        signal = self.load_signal(row)
        if self.training:
            signal = augment_ecg(
                signal,
                self.config.get("augmentation", {}),
                seed=int(torch.initial_seed() % (2**31) + index),
            )
        if self.perturbation.get("name", "clean") != "clean":
            signal = perturb_ecg(
                signal,
                self.perturbation,
                seed=int(self.config.get("seed", 42) + index * 1009),
            )
        if self.signal_scaler is not None:
            mean = np.asarray(self.signal_scaler["mean"], dtype=np.float32)[:, None]
            std = np.asarray(self.signal_scaler["std"], dtype=np.float32)[:, None]
            signal = (signal - mean) / std

        label_value = float(row[self.label_column]) if pd.notna(row[self.label_column]) else float("nan")
        clinical = np.zeros(0, dtype=np.float32)
        if self.use_clinical:
            clinical = row[self.clinical_columns].to_numpy(dtype=np.float32)
            if self.clinical_scaler is not None:
                clinical = (clinical - self.clinical_scaler["mean"]) / self.clinical_scaler["std"]

        return {
            "signal": torch.from_numpy(signal.astype(np.float32)),
            "clinical": torch.from_numpy(clinical.astype(np.float32)),
            "label": torch.tensor(label_value, dtype=torch.float32),
            "metadata_index": torch.tensor(int(row["metadata_index"]), dtype=torch.long),
        }


def load_metadata(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    path = resolve_path(project_root, config["data"]["metadata_path"])
    string_columns = {
        "Patient_id",
        "ecg_row_record",
        "ecg_med_record",
        "record_id",
        "med_record_id",
        "pair_key",
    }
    dataframe = pd.read_csv(path, dtype={column: "string" for column in string_columns})
    dataframe["metadata_index"] = np.arange(len(dataframe), dtype=np.int64)
    for column in dataframe.columns:
        if column in {
            "primary_cohort",
            "primary_cohort_all_records",
            "median_primary_cohort",
            "all_comers_cohort",
            "stemi_vs_nstemi_cohort",
            "three_class_cohort",
            "test_inference_cohort",
            "selected_one_per_patient",
        }:
            dataframe[column] = dataframe[column].fillna(False).astype(bool)
    return dataframe


def build_cohort_dataframe(
    metadata: pd.DataFrame,
    config: dict[str, Any],
    split: str,
) -> pd.DataFrame:
    data_config = config["data"]
    if split == "train":
        mask = metadata["split"].eq("train") & metadata[data_config["cohort_flag"]]
    elif split == "test":
        mask = metadata[data_config["test_flag"]]
    else:
        raise ValueError(f"split must be train or test, got {split}")
    dataframe = metadata.loc[mask].copy()

    if split == "train":
        label_column = data_config["label_column"]
        dataframe = dataframe.loc[dataframe[label_column].notna()].copy()
        dataframe["_label"] = (
            dataframe[label_column].astype(int) + int(data_config.get("label_shift", 0))
        )
        expected = set(range(int(data_config["num_classes"])))
        observed = set(dataframe["_label"].unique())
        if not observed.issubset(expected):
            raise ValueError(f"Unexpected labels after shift: {sorted(observed)}")
    else:
        dataframe["_label"] = np.nan

    if bool(data_config.get("strict_one_record_per_patient", True)):
        repeated = dataframe.groupby("Patient_id").size()
        if (repeated > 1).any():
            examples = repeated[repeated > 1].index[:10].tolist()
            raise ValueError(
                "Cohort contains multiple records per patient. Select one record per patient "
                f"or disable strict_one_record_per_patient. Examples: {examples}"
            )
    return dataframe.reset_index(drop=True)


def assign_patient_folds(
    dataframe: pd.DataFrame,
    label_column: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    result = dataframe.copy()
    if "cv_fold" in result.columns and result["cv_fold"].ge(0).all():
        result["fold"] = result["cv_fold"].astype(int)
        return result
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result["fold"] = -1
    for fold, (_, validation_index) in enumerate(
        splitter.split(result, y=result[label_column], groups=result["Patient_id"])
    ):
        result.loc[result.index[validation_index], "fold"] = fold
    return result


def split_fit_validation_calibration(
    dataframe: pd.DataFrame,
    label_column: str,
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("Validation and calibration fractions must be positive")
    if validation_fraction + calibration_fraction >= 0.5:
        raise ValueError("Validation + calibration fractions must be below 0.5")

    holdout_fraction = validation_fraction + calibration_fraction
    if dataframe["Patient_id"].nunique() == len(dataframe):
        stratify = dataframe[label_column].astype(int) if dataframe[label_column].nunique() > 1 else None
        fit_df, holdout_df = train_test_split(
            dataframe,
            test_size=holdout_fraction,
            random_state=seed,
            stratify=stratify,
        )
    else:
        patient_labels = (
            dataframe.groupby("Patient_id")[label_column]
            .max()
            .astype(int)
            .rename("patient_label")
            .reset_index()
        )
        patient_stratify = patient_labels["patient_label"] if patient_labels["patient_label"].nunique() > 1 else None
        fit_patients, holdout_patients = train_test_split(
            patient_labels,
            test_size=holdout_fraction,
            random_state=seed,
            stratify=patient_stratify,
        )
        fit_df = dataframe.loc[dataframe["Patient_id"].isin(fit_patients["Patient_id"])].copy()
        holdout_df = dataframe.loc[dataframe["Patient_id"].isin(holdout_patients["Patient_id"])].copy()

    calibration_share = calibration_fraction / holdout_fraction
    if holdout_df["Patient_id"].nunique() == len(holdout_df):
        holdout_stratify = (
            holdout_df[label_column].astype(int) if holdout_df[label_column].nunique() > 1 else None
        )
        validation_df, calibration_df = train_test_split(
            holdout_df,
            test_size=calibration_share,
            random_state=seed,
            stratify=holdout_stratify,
        )
    else:
        holdout_patient_labels = (
            holdout_df.groupby("Patient_id")[label_column]
            .max()
            .astype(int)
            .rename("patient_label")
            .reset_index()
        )
        holdout_stratify = (
            holdout_patient_labels["patient_label"]
            if holdout_patient_labels["patient_label"].nunique() > 1
            else None
        )
        validation_patients, calibration_patients = train_test_split(
            holdout_patient_labels,
            test_size=calibration_share,
            random_state=seed,
            stratify=holdout_stratify,
        )
        validation_df = holdout_df.loc[
            holdout_df["Patient_id"].isin(validation_patients["Patient_id"])
        ].copy()
        calibration_df = holdout_df.loc[
            holdout_df["Patient_id"].isin(calibration_patients["Patient_id"])
        ].copy()
    return fit_df.copy(), validation_df.copy(), calibration_df.copy()


def fit_signal_scaler(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    max_rows = int(config["data"].get("normalization_max_rows", 512))
    row_indices = np.arange(len(dataframe))
    if len(row_indices) > max_rows:
        rng = np.random.default_rng(int(config.get("seed", 42)))
        row_indices = np.sort(rng.choice(row_indices, size=max_rows, replace=False))
    dataset = ECGDataset(dataframe.iloc[row_indices], config, project_root, training=False)
    lead_count = int(config["data"]["lead_count"])
    total = np.zeros(lead_count, dtype=np.float64)
    total_sq = np.zeros(lead_count, dtype=np.float64)
    count = 0
    for index in range(len(dataset)):
        signal = dataset.load_signal(dataset.dataframe.iloc[index]).astype(np.float64)
        total += signal.sum(axis=1)
        total_sq += np.square(signal).sum(axis=1)
        count += signal.shape[1]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1.0e-8)
    std = np.sqrt(variance)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "rows_used": int(len(dataset)),
    }


def fit_clinical_scaler(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    columns = list(config["data"].get("clinical_columns", []))
    if not columns:
        return {"mean": [], "std": []}
    values = dataframe[columns].to_numpy(dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def make_dataloader(
    dataset: ECGDataset,
    batch_size: int,
    training: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": training,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory) and torch.cuda.is_available(),
        "drop_last": training and len(dataset) > batch_size,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
