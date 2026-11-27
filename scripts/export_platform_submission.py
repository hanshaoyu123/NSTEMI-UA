#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the official platform label-column submission")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--label", default="NSTEMI")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions,
        dtype={"Patient_id": "string", "ecg_row_record": "string"},
    )
    metadata = pd.read_csv(
        args.metadata,
        dtype={
            "Patient_id": "string",
            "ecg_row_record": "string",
            "ecg_med_record": "string",
        },
    )
    merge_columns = ["Patient_id", "ecg_row_record"]
    merged = metadata[merge_columns + ["ecg_med_record", "gender", "age", "Time_Interval"]].merge(
        predictions[merge_columns + ["prediction"]],
        on=merge_columns,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(predictions):
        raise RuntimeError(
            f"Prediction/metadata row mismatch: {len(predictions)} predictions, {len(merged)} matched"
        )
    if not merged["prediction"].isin([0, 1]).all():
        raise RuntimeError("Prediction column must contain only binary 0/1 values")
    merged = merged.rename(columns={"prediction": args.label})
    output_columns = [
        "Patient_id",
        "ecg_row_record",
        "ecg_med_record",
        "gender",
        "age",
        "Time_Interval",
        args.label,
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged[output_columns].to_csv(output, index=False)
    print(f"Platform submission: {output}")
    print(f"Rows: {len(merged)}; label column: {args.label}; positives: {int(merged[args.label].sum())}")


if __name__ == "__main__":
    main()

