#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstemi_ecg.metrics import delong_auc_test
from nstemi_ecg.utils import write_json


def load_predictions(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        dtype={"Patient_id": "string", "record_id": "string"},
    )
    if "metadata_index" not in dataframe.columns:
        raise ValueError(f"{path} does not contain metadata_index")
    if "probability" not in dataframe.columns:
        raise ValueError(f"{path} does not contain probability")
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired DeLong comparison of two OOF prediction files")
    parser.add_argument("--a", required=True, help="First model oof_predictions.csv")
    parser.add_argument("--b", required=True, help="Second model oof_predictions.csv")
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--output", default="runs/model_comparison.json")
    args = parser.parse_args()
    a = load_predictions(Path(args.a))
    b = load_predictions(Path(args.b))
    columns = ["metadata_index", "_label", "probability"]
    merged = a[columns].merge(
        b[columns],
        on=["metadata_index", "_label"],
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise RuntimeError("No matched OOF predictions")
    result = delong_auc_test(
        merged["_label"].to_numpy(),
        merged["probability_a"].to_numpy(),
        merged["probability_b"].to_numpy(),
    )
    output = {
        "name_a": args.name_a,
        "name_b": args.name_b,
        "matched_patients": int(len(merged)),
        **result,
    }
    write_json(output, Path(args.output))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

