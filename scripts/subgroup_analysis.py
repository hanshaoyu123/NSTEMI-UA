#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstemi_ecg.metrics import binary_metrics


def subgroup_definitions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    days = frame["Time_Interval"] / (24 * 60)
    return {
        "Overall": pd.Series(True, index=frame.index),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-level OOF subgroup analysis")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-n", type=int, default=30)
    args = parser.parse_args()
    prediction_path = Path(args.predictions).resolve()
    frame = pd.read_csv(
        prediction_path,
        dtype={"Patient_id": "string", "record_id": "string"},
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else prediction_path.parent / "subgroups"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(frame.groupby("fold")["threshold"].first().median())
    rows = []
    for name, mask in subgroup_definitions(frame).items():
        subset = frame.loc[mask]
        if len(subset) < args.min_n or subset["_label"].nunique() < 2:
            continue
        metrics = binary_metrics(
            subset["_label"].astype(int).to_numpy(),
            subset["probability"].to_numpy(),
            threshold,
        )
        rows.append(
            {
                "subgroup": name,
                "n": len(subset),
                "positive": int(subset["_label"].sum()),
                "auroc": metrics["auroc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "subgroup_metrics.csv", index=False)

    plot_frame = result.loc[result["subgroup"].ne("Overall")].iloc[::-1]
    fig, axis = plt.subplots(figsize=(6.5, max(4.0, 0.32 * len(plot_frame))))
    positions = np.arange(len(plot_frame))
    axis.scatter(plot_frame["auroc"], positions, color="#B94A48", zorder=3)
    overall = result.loc[result["subgroup"].eq("Overall"), "auroc"]
    if len(overall):
        axis.axvline(float(overall.iloc[0]), color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{name} (n={n})" for name, n in zip(plot_frame["subgroup"], plot_frame["n"])])
    axis.set_xlim(0.4, 1.0)
    axis.set_xlabel("AUROC")
    axis.set_title("Patient-level subgroup performance")
    fig.tight_layout()
    fig.savefig(output_dir / "subgroup_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "subgroup_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

