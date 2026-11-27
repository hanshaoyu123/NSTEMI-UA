#!/usr/bin/env python
"""Preprocess the ACS 12-lead ECG dataset for NSTEMI model development.

The script intentionally avoids WFDB as a runtime dependency. It parses the
headers and channel-interleaved format-16 files directly, validates the file
layout, performs quality control, filters both the 10-second ECG and the
one-second median beat, and writes model-ready memmaps plus patient-level
metadata and cross-validation folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_HZ = 500
RAW_FRAMES = 10 * FS_HZ
MED_FRAMES = 1 * FS_HZ
BANDPASS_HZ = (0.5, 40.0)
CLIP_MV = 5.0
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    parser.add_argument(
        "--skip-signals",
        action="store_true",
        help="Only rebuild metadata and QC; do not write filtered signal memmaps.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wfdb_header(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Invalid WFDB header: {path}")
    signal_count = int(first[1])
    sampling_rate = float(first[2])
    samples_per_signal = int(first[3])
    signal_lines = [line.split() for line in lines[1 : signal_count + 1]]
    formats = [int(parts[1].split("x")[0]) for parts in signal_lines]
    gains = []
    baselines = []
    units = []
    names = []
    for parts in signal_lines:
        adc = parts[2]
        baseline = int(parts[4])
        gain_text, unit_text = adc.split("/", maxsplit=1)
        baseline_text = gain_text.split("(", maxsplit=1)
        gain = float(baseline_text[0])
        if len(baseline_text) == 2:
            baseline = int(baseline_text[1].rstrip(")"))
        gains.append(gain)
        baselines.append(baseline)
        units.append(unit_text)
        names.append(parts[-1])
    return {
        "signal_count": signal_count,
        "sampling_rate": sampling_rate,
        "samples_per_signal": samples_per_signal,
        "formats": formats,
        "gains": np.asarray(gains, dtype=np.float64),
        "baselines": np.asarray(baselines, dtype=np.float64),
        "units": units,
        "lead_names": names,
    }


def read_format16(path: Path, header: dict) -> tuple[np.ndarray, int]:
    signal_count = header["signal_count"]
    raw = np.fromfile(path, dtype="<i2")
    frames = raw.size // signal_count
    usable = frames * signal_count
    if usable != raw.size:
        raw = raw[:usable]
    raw = raw.reshape(frames, signal_count).T.astype(np.float32)
    mv = (raw - header["baselines"][:, None]) / header["gains"][:, None]
    return mv, frames


def bandpass_ecg(signal: np.ndarray) -> np.ndarray:
    sos = butter(4, BANDPASS_HZ, btype="bandpass", fs=FS_HZ, output="sos")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        filtered = sosfiltfilt(sos, signal, axis=-1, padlen=min(300, signal.shape[-1] - 1))
    return np.clip(filtered, -CLIP_MV, CLIP_MV).astype(np.float32)


def lead_quality(signal_mv: np.ndarray, lead: str) -> list[dict]:
    rows = []
    for index, name in enumerate(lead):
        x = signal_mv[index]
        differences = np.diff(x)
        finite = np.isfinite(x)
        endpoint = max(1, int(0.5 * FS_HZ))
        baseline_drift = float(np.nanmedian(x[-endpoint:]) - np.nanmedian(x[:endpoint]))
        rows.append(
            {
                "lead": name,
                "min_mv": float(np.nanmin(x)),
                "max_mv": float(np.nanmax(x)),
                "mean_mv": float(np.nanmean(x)),
                "std_mv": float(np.nanstd(x)),
                "ptp_mv": float(np.nanmax(x) - np.nanmin(x)),
                "zero_diff_fraction": float(np.mean(differences == 0)),
                "saturation_fraction": float(np.mean(np.abs(x) >= 4.5)),
                "abs_gt_5mv_fraction": float(np.mean(np.abs(x) > CLIP_MV)),
                "hf_noise_proxy_mv": float(np.nanmedian(np.abs(differences))),
                "baseline_drift_mv": baseline_drift,
                "nonfinite_fraction": float(1.0 - np.mean(finite)),
                "flat_lead": bool(np.nanmax(x) - np.nanmin(x) < 0.005 or np.mean(differences == 0) > 0.98),
                "saturated_lead": bool(np.mean(np.abs(x) >= 4.5) > 0.001),
                "amplitude_outlier_lead": bool(np.mean(np.abs(x) > CLIP_MV) > 0.001),
                "noise_outlier_lead": bool(np.nanmedian(np.abs(differences)) > 0.10),
            }
        )
    return rows


def classify_label(row: pd.Series) -> str:
    if int(row["NSTEMI"]) == 1:
        return "NSTEMI"
    if int(row["STEMI"]) == 1:
        return "STEMI"
    if int(row["UA"]) == 1:
        return "UA"
    if int(row["AMI"]) == 1:
        return "AMI_UNCLASSIFIED"
    return "NON_ACS"


def prepare_metadata(data_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "CSV" / "train.csv")
    test = pd.read_csv(data_dir / "CSV" / "test.csv")
    train["split"] = "train"
    test["split"] = "test"
    for column in train.columns.intersection(test.columns):
        if column in {"Patient_id", "ecg_row_record", "ecg_med_record", "split"}:
            continue
        train[column] = pd.to_numeric(train[column], errors="coerce")
        test[column] = pd.to_numeric(test[column], errors="coerce")
    metadata = pd.concat([train, test], ignore_index=True, sort=False)
    metadata["source_row_index"] = metadata.groupby("split").cumcount()
    metadata["record_id"] = metadata["ecg_row_record"].str.replace(r"\.dat$", "", regex=True)
    metadata["med_record_id"] = metadata["ecg_med_record"].str.replace(r"\.med$", "", regex=True)
    metadata["group"] = "UNKNOWN_TEST"
    labeled = metadata["split"].eq("train")
    metadata.loc[labeled, "group"] = metadata.loc[labeled].apply(classify_label, axis=1)
    metadata["primary_label"] = -1
    metadata.loc[metadata["group"].eq("NSTEMI"), "primary_label"] = 1
    metadata.loc[metadata["group"].eq("UA"), "primary_label"] = 0
    metadata["all_comers_label"] = -1
    metadata.loc[labeled, "all_comers_label"] = metadata.loc[labeled, "group"].eq("NSTEMI").astype(int)
    metadata["stemi_vs_nstemi_label"] = -1
    metadata.loc[metadata["group"].eq("NSTEMI"), "stemi_vs_nstemi_label"] = 1
    metadata.loc[metadata["group"].eq("STEMI"), "stemi_vs_nstemi_label"] = 0
    metadata["three_class_label"] = -1
    metadata.loc[metadata["group"].eq("NSTEMI"), "three_class_label"] = 1
    metadata.loc[metadata["group"].eq("UA"), "three_class_label"] = 2
    metadata.loc[metadata["group"].eq("STEMI"), "three_class_label"] = 3
    metadata["time_interval_days"] = metadata["Time_Interval"] / (24 * 60)
    metadata["pair_key"] = metadata["Patient_id"] + "|" + metadata["record_id"]
    metadata["raw_exists"] = metadata.apply(
        lambda row: (data_dir / "row_data" / row["ecg_row_record"]).is_file(), axis=1
    )
    metadata["header_exists"] = metadata.apply(
        lambda row: (data_dir / "row_data" / f"{row['record_id']}.hea").is_file(), axis=1
    )
    metadata["med_exists"] = metadata.apply(
        lambda row: (data_dir / "med_data" / row["ecg_med_record"]).is_file(), axis=1
    )
    return metadata


def allocate_memmap(path: Path, rows: int, frames: int) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(rows, len(LEADS), frames),
    )


def run_qc(
    data_dir: Path,
    metadata: pd.DataFrame,
    output_dir: Path,
    skip_signals: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals_dir = output_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    record_rows: list[dict] = []
    lead_rows: list[dict] = []

    raw_outputs: dict[str, np.memmap | None] = {}
    med_outputs: dict[str, np.memmap | None] = {}
    if not skip_signals:
        for split in ("train", "test"):
            n_rows = int((metadata["split"] == split).sum())
            raw_outputs[split] = allocate_memmap(
                signals_dir / f"ecg_{split}_filtered_mV_12x5000_float16.npy",
                n_rows,
                RAW_FRAMES,
            )
            med_outputs[split] = allocate_memmap(
                signals_dir / f"median_{split}_filtered_mV_12x500_float16.npy",
                n_rows,
                MED_FRAMES,
            )

    for counter, row in enumerate(metadata.itertuples(index=False), start=1):
        raw_path = data_dir / "row_data" / row.ecg_row_record
        header_path = data_dir / "row_data" / f"{row.record_id}.hea"
        med_path = data_dir / "med_data" / row.ecg_med_record
        raw_reasons: list[str] = []
        median_reasons: list[str] = []

        if not raw_path.is_file() or not header_path.is_file() or not med_path.is_file():
            raw_reasons.append("missing_file")
            median_reasons.append("missing_file")
            record_rows.append(
                {
                    "record_id": row.record_id,
                    "split": row.split,
                    "source_row_index": row.source_row_index,
                    "qc_status": "FAIL",
                    "qc_reasons": "missing_file",
                    "median_qc_status": "FAIL",
                    "median_qc_reasons": "missing_file",
                    "raw_frames_actual": 0,
                    "med_frames_actual": 0,
                }
            )
            continue

        header = parse_wfdb_header(header_path)
        raw_mv, raw_frames = read_format16(raw_path, header)
        med_raw = np.fromfile(med_path, dtype="<i2")
        med_frames = med_raw.size // len(LEADS)
        med_mv = med_raw[: med_frames * len(LEADS)].reshape(med_frames, len(LEADS)).T.astype(np.float32)
        med_mv = (med_mv - header["baselines"][:, None]) / header["gains"][:, None]

        expected_raw_shape = (
            header["signal_count"] == len(LEADS)
            and math.isclose(header["sampling_rate"], FS_HZ)
            and header["samples_per_signal"] == RAW_FRAMES
            and header["formats"] == [16] * len(LEADS)
            and header["lead_names"] == LEADS
            and header["units"] == ["mV"] * len(LEADS)
        )
        if not expected_raw_shape:
            raw_reasons.append("header_mismatch")
        if raw_frames != RAW_FRAMES:
            raw_reasons.append(f"raw_frames_{raw_frames}")
        if med_frames != MED_FRAMES:
            median_reasons.append(f"med_frames_{med_frames}")

        current_leads = lead_quality(raw_mv, LEADS)
        for lead_row in current_leads:
            lead_row.update(
                {
                    "record_id": row.record_id,
                    "split": row.split,
                    "source_row_index": row.source_row_index,
                }
            )
            lead_rows.append(lead_row)

        flat_count = sum(lead["flat_lead"] for lead in current_leads)
        saturation_count = sum(lead["saturated_lead"] for lead in current_leads)
        outlier_count = sum(lead["amplitude_outlier_lead"] for lead in current_leads)
        noisy_count = sum(lead["noise_outlier_lead"] for lead in current_leads)
        all_flat = flat_count == len(LEADS)
        max_abs = float(np.max(np.abs(raw_mv)))
        median_max_abs = float(np.max(np.abs(med_mv))) if med_mv.size else np.inf

        if raw_frames < int(9.0 * FS_HZ):
            raw_reasons.append("duration_below_9s")
        if all_flat:
            raw_reasons.append("all_leads_flat")
        if flat_count >= 3:
            raw_reasons.append("three_or_more_flat_leads")
        if max_abs > 15.0:
            raw_reasons.append("extreme_amplitude")
        raw_fail_reasons = {
            reason
            for reason in raw_reasons
            if reason.startswith(
                ("missing_file", "header_", "duration_", "all_leads_flat", "three_or_more", "extreme_")
            )
        }
        if raw_fail_reasons:
            status = "FAIL"
        elif raw_reasons or saturation_count or outlier_count or noisy_count:
            status = "WARN"
        else:
            status = "PASS"

        if med_frames != MED_FRAMES or median_max_abs > 15.0 or not med_mv.size:
            median_reasons.append("median_waveform_invalid")
        median_status = "FAIL" if median_reasons else "PASS"

        if not skip_signals:
            filtered = bandpass_ecg(raw_mv)
            filtered_padded = np.zeros((len(LEADS), RAW_FRAMES), dtype=np.float32)
            filtered_padded[:, : filtered.shape[1]] = filtered
            raw_outputs[row.split][row.source_row_index] = filtered_padded.astype(np.float16)

            filtered_med = bandpass_ecg(med_mv)
            med_padded = np.zeros((len(LEADS), MED_FRAMES), dtype=np.float32)
            filtered_med = filtered_med[:, :MED_FRAMES]
            med_padded[:, : filtered_med.shape[1]] = filtered_med
            med_outputs[row.split][row.source_row_index] = med_padded.astype(np.float16)

        record_rows.append(
            {
                "record_id": row.record_id,
                "split": row.split,
                "source_row_index": row.source_row_index,
                "header_signal_count": header["signal_count"],
                "header_sampling_rate": header["sampling_rate"],
                "header_samples_per_signal": header["samples_per_signal"],
                "header_format": ",".join(map(str, header["formats"])),
                "header_leads": "|".join(header["lead_names"]),
                "raw_file_bytes": raw_path.stat().st_size,
                "raw_frames_actual": raw_frames,
                "raw_duration_s": raw_frames / FS_HZ,
                "med_file_bytes": med_path.stat().st_size,
                "med_frames_actual": med_frames,
                "raw_abs_max_mv": max_abs,
                "raw_global_std_mv": float(np.mean([lead["std_mv"] for lead in current_leads])),
                "max_zero_diff_fraction": float(
                    max(lead["zero_diff_fraction"] for lead in current_leads)
                ),
                "max_hf_noise_proxy_mv": float(
                    max(lead["hf_noise_proxy_mv"] for lead in current_leads)
                ),
                "max_abs_baseline_drift_mv": float(
                    max(abs(lead["baseline_drift_mv"]) for lead in current_leads)
                ),
                "flat_lead_count": flat_count,
                "saturated_lead_count": saturation_count,
                "amplitude_outlier_lead_count": outlier_count,
                "noise_outlier_lead_count": noisy_count,
                "qc_status": status,
                "qc_reasons": ";".join(raw_reasons),
                "median_max_abs_mv": median_max_abs,
                "median_qc_status": median_status,
                "median_qc_reasons": ";".join(median_reasons),
                "raw_sha256": sha256_file(raw_path),
                "med_sha256": sha256_file(med_path),
            }
        )
        if counter % 1000 == 0:
            print(f"QC processed {counter}/{len(metadata)} records", flush=True)

    if not skip_signals:
        for split in ("train", "test"):
            raw_outputs[split].flush()
            med_outputs[split].flush()

    qc_records = pd.DataFrame(record_rows)
    qc_leads = pd.DataFrame(lead_rows)
    qc_records["raw_duplicate_count"] = qc_records.groupby("raw_sha256")["record_id"].transform("size")
    qc_records["raw_duplicate_group"] = np.where(
        qc_records["raw_duplicate_count"] > 1,
        qc_records.groupby("raw_sha256").ngroup(),
        -1,
    )
    qc_records["med_duplicate_count"] = qc_records.groupby("med_sha256")["record_id"].transform("size")
    qc_records["med_duplicate_group"] = np.where(
        qc_records["med_duplicate_count"] > 1,
        qc_records.groupby("med_sha256").ngroup(),
        -1,
    )
    return qc_records, qc_leads


def finalize_metadata(metadata: pd.DataFrame, qc_records: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.drop(columns=["raw_exists", "header_exists", "med_exists"], errors="ignore")
    metadata = metadata.merge(
        qc_records.drop(columns=["split", "source_row_index"], errors="ignore"),
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    metadata["qc_status"] = metadata["qc_status"].fillna("FAIL")

    ordered = metadata.copy()
    ordered["_qc_sort"] = ordered["qc_status"].map({"PASS": 0, "WARN": 1, "FAIL": 2}).fillna(2)
    ordered = ordered.sort_values(
        ["Patient_id", "_qc_sort", "Time_Interval", "split", "source_row_index"]
    )
    ordered["patient_record_rank"] = ordered.groupby("Patient_id").cumcount() + 1
    selected_keys = set(ordered.loc[ordered["patient_record_rank"].eq(1), "pair_key"])
    metadata["selected_one_per_patient"] = metadata["pair_key"].isin(selected_keys)
    metadata["primary_cohort"] = (
        metadata["primary_label"].isin([0, 1])
        & metadata["qc_status"].ne("FAIL")
        & metadata["selected_one_per_patient"]
    )
    metadata["primary_cohort_all_records"] = (
        metadata["primary_label"].isin([0, 1]) & metadata["qc_status"].ne("FAIL")
    )
    metadata["median_primary_cohort"] = (
        metadata["primary_cohort"] & metadata["median_qc_status"].eq("PASS")
    )
    metadata["test_inference_cohort"] = (
        metadata["split"].eq("test")
        & metadata["selected_one_per_patient"]
        & metadata["qc_status"].ne("FAIL")
    )
    metadata["all_comers_cohort"] = (
        metadata["split"].eq("train")
        & metadata["all_comers_label"].isin([0, 1])
        & metadata["qc_status"].ne("FAIL")
        & metadata["selected_one_per_patient"]
    )
    metadata["stemi_vs_nstemi_cohort"] = (
        metadata["stemi_vs_nstemi_label"].isin([0, 1])
        & metadata["qc_status"].ne("FAIL")
        & metadata["selected_one_per_patient"]
    )
    metadata["three_class_cohort"] = (
        metadata["three_class_label"].isin([1, 2, 3])
        & metadata["qc_status"].ne("FAIL")
        & metadata["selected_one_per_patient"]
    )
    metadata["cv_fold"] = -1

    primary_train = metadata.loc[metadata["split"].eq("train") & metadata["primary_cohort"]].copy()
    if len(primary_train):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for fold, (_, validation_index) in enumerate(
            splitter.split(
                primary_train,
                y=primary_train["primary_label"],
                groups=primary_train["Patient_id"],
            )
        ):
            metadata.loc[primary_train.index[validation_index], "cv_fold"] = fold
    return metadata


def save_figures(metadata: pd.DataFrame, qc_leads: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    labels = metadata.loc[metadata["split"].eq("train"), "group"].value_counts()
    labels = labels.reindex(["NSTEMI", "STEMI", "UA", "NON_ACS", "AMI_UNCLASSIFIED"]).fillna(0)
    ax = labels.plot(kind="bar", color=["#C84545", "#E58A3A", "#D5B23F", "#6C8EAD", "#8A7AA0"])
    ax.set_title("Training label distribution")
    ax.set_xlabel("")
    ax.set_ylabel("ECG records")
    for patch, value in zip(ax.patches, labels):
        ax.text(patch.get_x() + patch.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "label_distribution.png")
    plt.close(ax.figure)

    primary = metadata.loc[metadata["primary_cohort"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for label, color, name in [(0, "#D5B23F", "UA"), (1, "#C84545", "NSTEMI")]:
        subset = primary.loc[primary["primary_label"].eq(label)]
        axes[0].hist(subset["age"].dropna(), bins=25, alpha=0.55, label=name, color=color)
        axes[1].hist(
            subset["time_interval_days"].dropna(),
            bins=np.linspace(0, 7, 15),
            alpha=0.55,
            label=name,
            color=color,
        )
    axes[0].set(title="Age distribution", xlabel="Age (years)", ylabel="Patients")
    axes[1].set(title="ECG-to-angiography interval", xlabel="Days", ylabel="Patients")
    sex_counts = primary.groupby(["primary_label", "gender"]).size().unstack(fill_value=0)
    sex_counts.index = ["UA", "NSTEMI"]
    sex_counts.columns = ["Female", "Male"]
    sex_counts.plot(kind="bar", ax=axes[2], color=["#B48EAD", "#5E81AC"])
    axes[2].set(title="Sex distribution", xlabel="", ylabel="Patients")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "primary_cohort_characteristics.png")
    plt.close(fig)

    qc_counts = metadata["qc_status"].value_counts().reindex(["PASS", "WARN", "FAIL"]).fillna(0)
    ax = qc_counts.plot(kind="bar", color=["#5B8C5A", "#D9A441", "#B94A48"])
    ax.set(title="Signal-level quality control", xlabel="", ylabel="ECG records")
    for patch, value in zip(ax.patches, qc_counts):
        ax.text(patch.get_x() + patch.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "qc_status.png")
    plt.close(ax.figure)

    merged = qc_leads.merge(
        metadata[["record_id", "split", "qc_status"]],
        on=["record_id", "split"],
        how="left",
        suffixes=("", "_record"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(merged["hf_noise_proxy_mv"].clip(upper=0.25), bins=60, color="#5E81AC")
    axes[0].set(title="High-frequency noise proxy", xlabel="Median |first difference| (mV)", ylabel="Lead-records")
    axes[1].hist(merged["max_mv"].clip(-5, 5), bins=60, color="#A3BE8C")
    axes[1].set(title="Positive amplitude", xlabel="Maximum amplitude (mV)", ylabel="Lead-records")
    axes[2].hist(merged["baseline_drift_mv"].clip(-1, 1), bins=60, color="#B48EAD")
    axes[2].set(title="Baseline drift proxy", xlabel="Endpoint median difference (mV)", ylabel="Lead-records")
    fig.tight_layout()
    fig.savefig(figures / "qc_metric_distributions.png")
    plt.close(fig)

    examples = metadata.loc[metadata["split"].eq("train") & metadata["primary_cohort"]]
    selected = examples.groupby("group", sort=True).head(1)
    signal_cache = {
        "train": np.load(
            output_dir / "signals" / "ecg_train_filtered_mV_12x5000_float16.npy",
            mmap_mode="r",
        )
    }
    time = np.arange(RAW_FRAMES) / FS_HZ
    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 8), sharex=True)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for axis, (_, row) in zip(axes, selected.iterrows()):
        signal = np.asarray(signal_cache["train"][int(row["source_row_index"])], dtype=np.float32)
        offset = np.arange(len(LEADS))[::-1, None] * 2.5
        axis.plot(time, (signal + offset).T, linewidth=0.55)
        axis.set_yticks(np.arange(len(LEADS))[::-1] * 2.5)
        axis.set_yticklabels(LEADS[::-1])
        axis.set_title(f"{row['record_id']} | {row['group']} | filtered 0.5-40 Hz")
        axis.set_ylabel("mV + offset")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(figures / "representative_filtered_ecgs.png")
    plt.close(fig)


def compute_normalization(metadata: pd.DataFrame, output_dir: Path) -> dict:
    path = output_dir / "signals" / "ecg_train_filtered_mV_12x5000_float16.npy"
    if not path.is_file():
        return {}
    array = np.load(path, mmap_mode="r")
    rows = metadata.loc[
        metadata["split"].eq("train") & metadata["primary_cohort"], "source_row_index"
    ].to_numpy()
    count = 0
    total = np.zeros(len(LEADS), dtype=np.float64)
    total_sq = np.zeros(len(LEADS), dtype=np.float64)
    for start in range(0, len(rows), 256):
        batch = np.asarray(array[rows[start : start + 256]], dtype=np.float64)
        count += batch.shape[0] * batch.shape[2]
        total += batch.sum(axis=(0, 2))
        total_sq += np.square(batch).sum(axis=(0, 2))
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return {
        "scope": "primary one-per-patient training cohort; filtered 10-s ECG",
        "rows": int(len(rows)),
        "samples_per_lead": int(count // len(LEADS)),
        "lead_order": LEADS,
        "mean_mv": dict(zip(LEADS, map(float, mean))),
        "std_mv": dict(zip(LEADS, map(float, np.sqrt(variance)))),
        "note": "For cross-validation, recompute these statistics within each training fold.",
    }


def write_summary(
    metadata: pd.DataFrame,
    qc_leads: pd.DataFrame,
    normalization: dict,
    output_dir: Path,
) -> None:
    summary = {
        "total_records": int(len(metadata)),
        "train_records": int(metadata["split"].eq("train").sum()),
        "test_records": int(metadata["split"].eq("test").sum()),
        "unique_patients_total": int(metadata["Patient_id"].nunique()),
        "unique_patients_train": int(metadata.loc[metadata["split"].eq("train"), "Patient_id"].nunique()),
        "unique_patients_test": int(metadata.loc[metadata["split"].eq("test"), "Patient_id"].nunique()),
        "group_counts_train": {
            key: int(value)
            for key, value in metadata.loc[metadata["split"].eq("train"), "group"].value_counts().items()
        },
        "qc_counts": {
            key: int(value)
            for key, value in metadata["qc_status"].value_counts().items()
        },
        "primary_cohort_train": {
            "records_one_per_patient": int(
                (metadata["split"].eq("train") & metadata["primary_cohort"]).sum()
            ),
            "NSTEMI": int(
                (
                    metadata["split"].eq("train")
                    & metadata["primary_cohort"]
                    & metadata["primary_label"].eq(1)
                ).sum()
            ),
            "UA": int(
                (
                    metadata["split"].eq("train")
                    & metadata["primary_cohort"]
                    & metadata["primary_label"].eq(0)
                ).sum()
            ),
            "all_records": int(
                (metadata["split"].eq("train") & metadata["primary_cohort_all_records"]).sum()
            ),
        },
        "test_records_selected_one_per_patient": int(
            (metadata["split"].eq("test") & metadata["selected_one_per_patient"]).sum()
        ),
        "exact_duplicate_raw_groups": int(
            metadata.loc[
                metadata["raw_duplicate_count"].fillna(0) > 1, "raw_duplicate_group"
            ].nunique()
        ),
        "cross_split_exact_duplicate_groups": int(
            metadata.loc[metadata["raw_duplicate_count"].fillna(0) > 1]
            .groupby("raw_duplicate_group")["split"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "exact_duplicate_median_groups": int(
            metadata.loc[
                metadata["med_duplicate_count"].fillna(0) > 1, "med_duplicate_group"
            ].nunique()
        ),
        "cross_split_exact_duplicate_median_groups": int(
            metadata.loc[metadata["med_duplicate_count"].fillna(0) > 1]
            .groupby("med_duplicate_group")["split"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "lead_qc_rows": int(len(qc_leads)),
        "median_qc_counts": {
            key: int(value)
            for key, value in metadata["median_qc_status"].value_counts().items()
        },
        "normalization": normalization,
        "filtering": {
            "type": "fourth-order Butterworth zero-phase bandpass",
            "band_hz": list(BANDPASS_HZ),
            "clip_mv": CLIP_MV,
            "stored_dtype": "float16",
            "raw_duration_s": RAW_FRAMES / FS_HZ,
            "median_duration_s": MED_FRAMES / FS_HZ,
        },
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CSV metadata and validating record identifiers...", flush=True)
    metadata = prepare_metadata(data_dir)
    print(f"Metadata rows: {len(metadata):,}", flush=True)

    print("Running signal validation and preprocessing...", flush=True)
    qc_records, qc_leads = run_qc(data_dir, metadata, output_dir, args.skip_signals)
    metadata = finalize_metadata(metadata, qc_records)

    metadata = metadata.sort_values(["split", "source_row_index"]).reset_index(drop=True)
    metadata.to_csv(metadata_dir / "metadata_preprocessed.csv", index=False)
    metadata.loc[metadata["split"].eq("train")].to_csv(
        metadata_dir / "train_preprocessed.csv", index=False
    )
    metadata.loc[metadata["split"].eq("test")].to_csv(
        metadata_dir / "test_preprocessed.csv", index=False
    )
    qc_records.to_csv(metadata_dir / "qc_records.csv", index=False)
    qc_leads.to_csv(metadata_dir / "qc_leads.csv", index=False)

    cohort_labels = {
        "primary_one_per_patient": "primary_cohort",
        "primary_median_waveform": "median_primary_cohort",
        "primary_all_records": "primary_cohort_all_records",
        "all_comers": "all_comers_cohort",
        "NSTEMI_vs_STEMI": "stemi_vs_nstemi_cohort",
        "three_class": "three_class_cohort",
        "test_inference_one_per_patient": "test_inference_cohort",
    }
    cohort_rows = []
    for cohort_name, flag_column in cohort_labels.items():
        for split in ("train", "test"):
            subset = metadata.loc[metadata["split"].eq(split) & metadata[flag_column]]
            cohort_rows.append(
                {
                    "cohort": cohort_name,
                    "split": split,
                    "records": int(len(subset)),
                    "patients": int(subset["Patient_id"].nunique()),
                    "NSTEMI": int(subset["primary_label"].eq(1).sum()),
                    "UA": int(subset["primary_label"].eq(0).sum()),
                    "STEMI": int(subset["group"].eq("STEMI").sum()),
                    "labels_available": bool(split == "train"),
                    "cv_folds_defined": int(subset["cv_fold"].ge(0).sum()),
                }
            )
    pd.DataFrame(cohort_rows).to_csv(metadata_dir / "cohort_summary.csv", index=False)

    label_audit = (
        metadata.loc[metadata["split"].eq("train"), "group"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="records")
    )
    label_audit.to_csv(metadata_dir / "label_audit.csv", index=False)

    normalization = {}
    if not args.skip_signals:
        print("Computing training-only normalization statistics...", flush=True)
        normalization = compute_normalization(metadata, output_dir)
        (metadata_dir / "normalization_train_primary_global.json").write_text(
            json.dumps(normalization, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Writing quality-control and cohort figures...", flush=True)
        save_figures(metadata, qc_leads, output_dir)

    write_summary(metadata, qc_leads, normalization, output_dir)
    print(f"Preprocessing complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
