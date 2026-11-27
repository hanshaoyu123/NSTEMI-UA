#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize patient-level multi-seed runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/main_nstemi_vs_ua",
            "runs/main_nstemi_vs_ua_seed43",
            "runs/main_nstemi_vs_ua_seed44",
        ],
    )
    parser.add_argument("--output", default="runs/main_multiseed_summary.csv")
    args = parser.parse_args()
    rows = []
    reference_indices = None
    for run in args.runs:
        run_dir = Path(run)
        summary_path = run_dir / "summary.json"
        oof_path = run_dir / "oof_predictions.csv"
        if not summary_path.is_file() or not oof_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pooled = summary["pooled_oof"]
        folds = pd.DataFrame(summary["folds"])
        run_config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
        oof = pd.read_csv(oof_path, usecols=["metadata_index", "_label"])
        indices = np.sort(oof["metadata_index"].to_numpy())
        if reference_indices is None:
            reference_indices = indices
        same_patients = np.array_equal(indices, reference_indices)
        rows.append(
            {
                "run": run_dir.name,
                "seed": int(run_config.get("seed", -1)),
                "auroc": pooled["auroc"],
                "auprc": pooled["auprc"],
                "sensitivity": pooled["sensitivity"],
                "specificity": pooled["specificity"],
                "f1": pooled["f1"],
                "brier": pooled["brier"],
                "fold_auroc_mean": folds["auroc"].mean(),
                "fold_auroc_sd": folds["auroc"].std(ddof=1),
                "same_oof_patients": same_patients,
            }
        )
    if not rows:
        raise FileNotFoundError("No completed runs were found")
    dataframe = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
