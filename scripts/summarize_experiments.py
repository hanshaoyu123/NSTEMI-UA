#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_summary(run_dir: Path) -> dict | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("pooled_oof", payload)
    return {
        "experiment": run_dir.name,
        "path": str(run_dir),
        "n": metrics.get("n"),
        "positive": metrics.get("positive"),
        "auroc": metrics.get("auroc", metrics.get("macro_auroc")),
        "auprc": metrics.get("auprc"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "ppv": metrics.get("ppv"),
        "npv": metrics.get("npv"),
        "f1": metrics.get("f1", metrics.get("macro_f1")),
        "brier": metrics.get("brier"),
        "accuracy": metrics.get("accuracy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment summaries into one table")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="runs/experiment_comparison.csv")
    args = parser.parse_args()
    root = Path(args.runs_root).resolve()
    rows = []
    summary_paths = sorted(root.rglob("summary.json"))
    for summary_path in summary_paths:
        row = flatten_summary(summary_path.parent)
        if row is not None:
            rows.append(row)
    for summary_path in sorted(root.rglob("tabular_baseline_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for name, metrics in payload.items():
            rows.append(
                {
                    "experiment": name,
                    "path": str(summary_path.parent),
                    "n": metrics.get("n"),
                    "positive": metrics.get("positive"),
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "sensitivity": metrics.get("sensitivity"),
                    "specificity": metrics.get("specificity"),
                    "ppv": metrics.get("ppv"),
                    "npv": metrics.get("npv"),
                    "f1": metrics.get("f1"),
                    "brier": metrics.get("brier"),
                    "accuracy": metrics.get("accuracy"),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No summary.json files found under {root}")
    dataframe = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["experiment", "n", "auroc", "auprc"])
        .sort_values("auroc", ascending=False, na_position="last")
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
