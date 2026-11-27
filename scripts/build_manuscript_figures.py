#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstemi_ecg.metrics import decision_curve


COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "gray": "#777777",
    "light_gray": "#DDDDDD",
    "black": "#222222",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=COLORS["black"],
        linewidth=0.7,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=COLORS["black"],
            linewidth=0.8,
        )
    )


def build_figure_technical_roadmap(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.4, 5.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    stages = [
        (
            0.035,
            "1  COHORT",
            "Cohort construction",
            "19,955 records / 18,909 patients\n"
            "7,197 NSTEMI or UA records\n"
            "6,749 patients after waveform QC\n"
            "One ECG per patient: 6,616\n"
            "NSTEMI 1,133 | UA 5,483",
            "#DCEFF7",
        ),
        (
            0.27,
            "2  VALIDATION",
            "Patient-level nested design",
            "Outer: five patient-grouped folds\n"
            "Inner: fit 80%\n"
            "Early-stop validation 10%\n"
            "Independent calibration 10%\n"
            "Stability: seeds 42-46",
            "#E8F4E8",
        ),
        (
            0.505,
            "3  MODELING",
            "Multiscale ECG learning",
            "0.5-40 Hz filtering\n"
            "Fold-specific normalization\n"
            "Multiscale residual encoder\n"
            "Lead attention fusion\n"
            "Temperature scaling",
            "#FFF2D9",
        ),
        (
            0.74,
            "4  EVALUATION",
            "Discrimination and reliability",
            "AUROC and AUPRC\n"
            "Calibration and DCA\n"
            "Clinical and architecture baselines\n"
            "Subgroups and perturbations\n"
            "Integrated gradients",
            "#F2EAF7",
        ),
    ]
    width, top, height = 0.225, 0.68, 0.24
    for x, stage, title, details, color in stages:
        add_box(axis, (x, top), width, height, "", color)
        axis.text(
            x + width / 2,
            top + height - 0.025,
            stage,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=COLORS["teal"],
        )
        axis.text(
            x + width / 2,
            top + height - 0.065,
            title,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        axis.text(
            x + width / 2,
            top + height - 0.105,
            details,
            ha="center",
            va="top",
            fontsize=7.7,
            linespacing=1.35,
        )
    for left, right in zip(stages[:-1], stages[1:]):
        add_arrow(
            axis,
            (left[0] + width + 0.004, top + height / 2),
            (right[0] - 0.008, top + height / 2),
        )

    add_box(
        axis,
        (0.055, 0.34),
        0.89,
        0.15,
        "PRIMARY ANALYSIS\n"
        "Pre-angiography 12-lead ECG -> NSTEMI versus UA\n"
        "Locked metric: pooled out-of-fold AUROC | all seeds reported",
        "#DCEFF7",
    )
    add_arrow(axis, (0.50, top), (0.50, 0.49))

    add_box(
        axis,
        (0.055, 0.12),
        0.89,
        0.13,
        "SECONDARY AND STRESS-TEST FRAME\n"
        "Clinical baselines | attention ablation | subgroup forest | "
        "signal perturbations | lead-level integrated gradients",
        "#FFF2D9",
    )
    add_arrow(axis, (0.50, 0.34), (0.50, 0.25))

    add_box(
        axis,
        (0.055, 0.015),
        0.89,
        0.065,
        "EVIDENCE BOUNDARY  |  retrospective, same-centre dataset; "
        "official subtype holdout labels unavailable; no independent external validation",
        "#F8E3DE",
    )
    add_arrow(axis, (0.50, 0.12), (0.50, 0.08))
    save_figure(fig, output_dir, "figure1_technical_roadmap")


def build_figure_1(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.8, 5.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    add_box(
        axis,
        (0.04, 0.84),
        0.88,
        0.11,
        "Released ACS ECG dataset\n19,955 records | 18,909 patients",
        "#DCEFF7",
    )
    add_box(
        axis,
        (0.04, 0.59),
        0.60,
        0.12,
        "NSTEMI or UA after raw-waveform QC\n7,197 records | 6,749 patients",
        "#E8F4E8",
    )
    add_box(
        axis,
        (0.72, 0.59),
        0.24,
        0.12,
        "Official holdout\n1,995 records | 1,891 patients",
        "#F2EAF7",
    )
    add_box(
        axis,
        (0.04, 0.36),
        0.60,
        0.12,
        "One ECG per patient\nquality rank, then shortest ECG-to-angiography interval",
        "#FFF2D9",
    )
    add_box(
        axis,
        (0.04, 0.12),
        0.60,
        0.13,
        "Primary analysis cohort\n6,616 patients: NSTEMI 1,133 | UA 5,483",
        "#E8F4E8",
    )
    add_box(
        axis,
        (0.72, 0.12),
        0.24,
        0.13,
        "Transport analysis\n1,839 after\none-per-patient selection",
        "#F2EAF7",
    )
    add_arrow(axis, (0.34, 0.84), (0.34, 0.71))
    add_arrow(axis, (0.81, 0.84), (0.84, 0.71))
    add_arrow(axis, (0.34, 0.59), (0.34, 0.48))
    add_arrow(axis, (0.34, 0.36), (0.34, 0.25))
    add_arrow(axis, (0.84, 0.59), (0.84, 0.25))
    save_figure(fig, output_dir, "figure3_study_flow")


def build_figure_2(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 3.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.01, "12 x 5,000\nECG input", "#DCEFF7"),
        (0.18, "Per-lead shared\n1D encoder", "#E8F4E8"),
        (0.35, "4 residual stages\n3/5/7 kernels + SE", "#FFF2D9"),
        (0.52, "Average + max\npooling", "#F8E3DE"),
        (0.69, "12-lead\nattention", "#F2EAF7"),
        (0.86, "NSTEMI\nlogit", "#E8F4E8"),
    ]
    for index, (x, text, color) in enumerate(boxes):
        width = 0.13 if index < 5 else 0.07
        add_box(axis, (x, 0.38), width, 0.25, text, color)
        if index < len(boxes) - 1:
            add_arrow(axis, (x + width, 0.505), (boxes[index + 1][0] - 0.01, 0.505))
    axis.text(
        0.5,
        0.18,
        "5,606,042 parameters | weighted BCE | AdamW | mixed precision",
        ha="center",
        va="center",
        fontsize=8,
    )
    save_figure(fig, output_dir, "figure2_model_architecture")


def build_figure_3(oof_path: Path, output_dir: Path) -> None:
    predictions = pd.read_csv(oof_path)
    labels = predictions["_label"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    threshold = float(predictions.groupby("fold")["threshold"].first().median())
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    threshold_curve, model_nb, treat_all = decision_curve(labels, probabilities)
    fraction_positive, mean_predicted = calibration_curve(
        labels,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3))
    axis = axes[0, 0]
    axis.plot(fpr, tpr, color=COLORS["blue"], linewidth=1.8)
    axis.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=0.8)
    axis.set(xlabel="1 - specificity", ylabel="Sensitivity")
    axis.set_title(
        f"ROC curve (AUROC {roc_auc_score(labels, probabilities):.3f})",
        fontsize=9,
    )

    axis = axes[0, 1]
    axis.plot(recall, precision, color=COLORS["teal"], linewidth=1.8)
    axis.axhline(labels.mean(), color=COLORS["gray"], linestyle="--", linewidth=0.8)
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.set_title(
        f"Precision-recall (AUPRC {average_precision_score(labels, probabilities):.3f})",
        fontsize=9,
    )

    axis = axes[1, 0]
    axis.plot(
        mean_predicted,
        fraction_positive,
        color=COLORS["orange"],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )
    axis.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=0.8)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed proportion")
    axis.set_title("Calibration", fontsize=9)

    axis = axes[1, 1]
    mask = threshold_curve <= 0.30
    axis.plot(
        threshold_curve[mask],
        model_nb[mask],
        color=COLORS["red"],
        linewidth=1.8,
        label="Model",
    )
    axis.plot(
        threshold_curve[mask],
        treat_all[mask],
        color=COLORS["black"],
        linestyle="--",
        linewidth=0.8,
        label="Treat all",
    )
    axis.axhline(0, color=COLORS["gray"], linestyle=":", linewidth=0.8, label="Treat none")
    axis.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        xlim=(0.01, 0.30),
        ylim=(-0.20, 0.18),
    )
    axis.set_title("Decision curve", fontsize=9)
    axis.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "figure4_primary_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure4_primary_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def build_figure_4(audit_path: Path, output_dir: Path) -> None:
    audit = pd.read_csv(audit_path).sort_values("seed")
    metrics = [
        ("pooled_auroc", "Raw pooled", COLORS["red"], "o"),
        ("fold_mean_auroc", "Mean fold", COLORS["blue"], "s"),
        ("rank_normalized_pooled_auroc", "Rank-normalized", COLORS["teal"], "^"),
    ]
    fig, axis = plt.subplots(figsize=(6.4, 3.9))
    for column, label, color, marker in metrics:
        axis.plot(
            audit["seed"],
            audit[column],
            marker=marker,
            color=color,
            linewidth=1.2,
            markersize=5,
            label=label,
        )
    axis.set(
        xlabel="Random seed",
        ylabel="Pooled OOF AUROC",
        xticks=audit["seed"].tolist(),
        ylim=(0.70, 0.80),
    )
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    axis.grid(axis="y", color=COLORS["light_gray"], linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure5_multiseed_stability")


def bootstrap_auc_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(roc_auc_score(sampled_labels, probabilities[indices]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def subgroup_masks(frame: pd.DataFrame) -> dict[str, pd.Series | str]:
    days = frame["Time_Interval"] / (24 * 60)
    return {
        "Overall": "overall",
        "Female": frame["gender"].eq(0),
        "Male": frame["gender"].eq(1),
        "Age <65": frame["age"].lt(65),
        "Age >=65": frame["age"].ge(65),
        "ECG-to-angiography <1 day": days.lt(1),
        "1-3 days": days.ge(1) & days.le(3),
        ">3 days": days.gt(3),
        "No VF/VT": frame["VF_VT"].eq(0),
        "VF/VT": frame["VF_VT"].eq(1),
        "Non-paced": frame["Paced"].eq(0),
        "Paced": frame["Paced"].eq(1),
        "No prior PCI": frame["Prior_PCI"].eq(0),
        "Prior PCI": frame["Prior_PCI"].eq(1),
        "QC PASS": frame["qc_status"].eq("PASS"),
        "QC WARN": frame["qc_status"].eq("WARN"),
    }


def build_figure_5(oof_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(oof_path)
    rows = []
    for name, mask in subgroup_masks(frame).items():
        subset = frame if isinstance(mask, str) else frame.loc[mask]
        labels = subset["_label"].to_numpy(dtype=int)
        probabilities = subset["probability"].to_numpy(dtype=float)
        if len(subset) < 30 or len(np.unique(labels)) < 2:
            continue
        low, high = bootstrap_auc_ci(labels, probabilities)
        rows.append(
            {
                "subgroup": name,
                "n": len(subset),
                "auroc": roc_auc_score(labels, probabilities),
                "ci_low": low,
                "ci_high": high,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "subgroup_auc_ci.csv", index=False)
    plotting = result.loc[result["subgroup"].ne("Overall")].iloc[::-1]
    overall = float(result.loc[result["subgroup"].eq("Overall"), "auroc"].iloc[0])
    fig, axis = plt.subplots(figsize=(6.6, 5.2))
    positions = np.arange(len(plotting))
    errors = np.vstack(
        [
            plotting["auroc"] - plotting["ci_low"],
            plotting["ci_high"] - plotting["auroc"],
        ]
    )
    axis.errorbar(
        plotting["auroc"],
        positions,
        xerr=errors,
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        capsize=2,
        markersize=4,
        linewidth=0.8,
    )
    axis.axvline(overall, color=COLORS["red"], linestyle="--", linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{name} (n={n})" for name, n in zip(plotting["subgroup"], plotting["n"])]
    )
    axis.set_xlabel("AUROC")
    axis.set_xlim(0.50, 0.90)
    axis.grid(axis="x", color=COLORS["light_gray"], linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure6_subgroups")


def readable_scenario(value: str) -> str:
    replacements = {
        "gaussian_noise__sigma_mv=": "Gaussian noise, sigma=",
        "baseline_wander__amplitude_mv=": "Baseline wander, amplitude=",
        "amplitude_scale__factor=": "Amplitude x",
        "time_shift__seconds=": "Time shift, ",
        "lead_dropout__count=": "Lead dropout, n=",
        "powerline_50hz__amplitude_mv=": "50-Hz interference, ",
        "quantize__step_mv=": "Quantization, ",
    }
    if value == "clean":
        return "Clean"
    if value == "limb_lead_dropout":
        return "All limb leads removed"
    if value == "chest_lead_dropout":
        return "All chest leads removed"
    output = value
    for old, new in replacements.items():
        output = output.replace(old, new)
    if output.startswith("Gaussian noise, sigma="):
        return output + " mV"
    if output.startswith("Baseline wander, amplitude="):
        amplitude, frequency = output.removeprefix(
            "Baseline wander, amplitude="
        ).split("__frequency_hz=")
        return f"Baseline wander, {amplitude} mV / {frequency} Hz"
    if output.startswith("Time shift, "):
        return output + " s"
    if output.startswith("50-Hz interference, "):
        return output + " mV"
    if output.startswith("Quantization, "):
        return output + " mV"
    return output


def build_figure_6(robustness_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(robustness_path)
    data["label"] = data["scenario"].map(readable_scenario)
    clean = float(data.loc[data["scenario"].eq("clean"), "auroc"].iloc[0])
    data = data.sort_values("auroc")
    fig, axis = plt.subplots(figsize=(6.5, 5.8))
    positions = np.arange(len(data))
    axis.hlines(
        positions,
        clean,
        data["auroc"],
        color=COLORS["light_gray"],
        linewidth=1.0,
    )
    colors = np.where(data["scenario"].eq("clean"), COLORS["teal"], COLORS["blue"])
    axis.scatter(data["auroc"], positions, color=colors, s=28, zorder=3)
    axis.axvline(clean, color=COLORS["red"], linestyle="--", linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(data["label"])
    axis.set_xlabel("AUROC")
    axis.set_xlim(0.55, 0.80)
    axis.grid(axis="x", color=COLORS["light_gray"], linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure7_signal_robustness")


def build_figure_s1(importance_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(importance_path)
    lead_names = {
        "L1": "I",
        "L2": "II",
        "L3": "III",
        "L4": "aVR",
        "L5": "aVL",
        "L6": "aVF",
        "L7": "V1",
        "L8": "V2",
        "L9": "V3",
        "L10": "V4",
        "L11": "V5",
        "L12": "V6",
    }
    data["importance"] = data["importance"].clip(lower=0)
    totals = data.groupby("record_id")["importance"].transform("sum")
    data["normalized_importance"] = np.where(
        totals.gt(0),
        data["importance"] / totals * 100.0,
        np.nan,
    )
    grouped = (
        data.groupby("lead")["normalized_importance"]
        .agg(["mean", "std"])
        .reindex([f"L{index}" for index in range(1, 13)])
    )
    grouped.index = [lead_names[value] for value in grouped.index]
    fig, axis = plt.subplots(figsize=(6.2, 4.1))
    positions = np.arange(len(grouped))
    axis.barh(
        positions,
        grouped["mean"],
        xerr=grouped["std"],
        color=COLORS["blue"],
        edgecolor=COLORS["black"],
        linewidth=0.4,
        capsize=2,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(grouped.index)
    axis.invert_yaxis()
    axis.set_xlabel("Normalized integrated-gradient importance (%)")
    axis.set_ylabel("ECG lead")
    axis.grid(axis="x", color=COLORS["light_gray"], linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_dir, "figureS1_lead_importance")


def main() -> None:
    configure_style()
    output_dir = Path("paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = Path("runs/main_nstemi_vs_ua_calibrated/oof_predictions.csv")
    audit_path = Path("runs/multiseed_oof_audit.csv")
    robustness_path = Path("runs/main_nstemi_vs_ua_calibrated/robustness/robustness_metrics.csv")

    build_figure_technical_roadmap(output_dir)
    build_figure_1(output_dir)
    build_figure_2(output_dir)
    build_figure_3(oof_path, output_dir)
    build_figure_4(audit_path, output_dir)
    build_figure_5(oof_path, output_dir)
    build_figure_6(robustness_path, output_dir)
    build_figure_s1(
        Path(
            "runs/main_nstemi_vs_ua_calibrated/"
            "interpretability/lead_importance.csv"
        ),
        output_dir,
    )
    print(f"Saved manuscript figures in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
