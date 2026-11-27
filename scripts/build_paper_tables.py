#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def median_iqr(values: pd.Series, decimals: int = 2) -> str:
    values = values.dropna().astype(float)
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return f"{median:.{decimals}f} ({q1:.{decimals}f}-{q3:.{decimals}f})"


def count_percent(values: pd.Series) -> str:
    values = values.dropna().astype(int)
    return f"{int(values.sum()):,} ({values.mean() * 100:.1f}%)"


def continuous_row(
    dataframe: pd.DataFrame,
    column: str,
    label: str,
    decimals: int = 2,
) -> dict[str, str | float]:
    positive = dataframe.loc[dataframe["primary_label"].eq(1), column]
    negative = dataframe.loc[dataframe["primary_label"].eq(0), column]
    p_value = mannwhitneyu(positive, negative, alternative="two-sided").pvalue
    return {
        "characteristic": label,
        "overall": median_iqr(dataframe[column], decimals),
        "ua": median_iqr(negative, decimals),
        "nstemi": median_iqr(positive, decimals),
        "p_value": f"{p_value:.4f}",
    }


def binary_row(
    dataframe: pd.DataFrame,
    column: str,
    label: str,
) -> dict[str, str | float]:
    positive = dataframe.loc[dataframe["primary_label"].eq(1), column]
    negative = dataframe.loc[dataframe["primary_label"].eq(0), column]
    table = np.array(
        [
            [int(positive.sum()), int(len(positive) - positive.sum())],
            [int(negative.sum()), int(len(negative) - negative.sum())],
        ]
    )
    p_value = chi2_contingency(table, correction=False).pvalue
    return {
        "characteristic": label,
        "overall": count_percent(dataframe[column]),
        "ua": count_percent(negative),
        "nstemi": count_percent(positive),
        "p_value": f"{p_value:.4f}",
    }


def build_table_1(metadata_path: Path, output_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    cohort = metadata.loc[metadata["primary_cohort"].astype(bool)].copy()
    cohort["female"] = cohort["gender"].eq(0).astype(int)
    cohort["vf_vt"] = cohort["VF_VT"].fillna(0).astype(int)
    cohort["paced"] = cohort["Paced"].fillna(0).astype(int)
    cohort["prior_pci"] = cohort["Prior_PCI"].fillna(0).astype(int)
    cohort["qc_warn"] = cohort["qc_status"].eq("WARN").astype(int)

    rows = [
        {
            "characteristic": "Patients, n",
            "overall": f"{len(cohort):,}",
            "ua": f"{int(cohort['primary_label'].eq(0).sum()):,}",
            "nstemi": f"{int(cohort['primary_label'].eq(1).sum()):,}",
            "p_value": "",
        },
        continuous_row(cohort, "age", "Age, years, median (IQR)", 0),
        binary_row(cohort, "female", "Female sex, n (%)"),
        continuous_row(
            cohort,
            "time_interval_days",
            "ECG-to-angiography interval, days, median (IQR)",
            2,
        ),
        binary_row(cohort, "vf_vt", "VF/VT, n (%)"),
        binary_row(cohort, "paced", "Paced rhythm, n (%)"),
        binary_row(cohort, "prior_pci", "Prior PCI, n (%)"),
        binary_row(cohort, "qc_warn", "ECG quality WARN, n (%)"),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "table1_baseline.csv", index=False)
    (output_dir / "table1_baseline.md").write_text(
        markdown_table(table),
        encoding="utf-8",
    )
    return table


def markdown_table(dataframe: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(dataframe.columns) + " |",
        "|" + "|".join(["---"] * len(dataframe.columns)) + "|",
    ]
    for row in dataframe.astype(str).itertuples(index=False):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build_table_2(
    multiseed_path: Path,
    audit_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    multiseed = pd.read_csv(multiseed_path)
    audit = pd.read_csv(audit_path)
    audit_columns = audit[
        [
            "seed",
            "rank_normalized_pooled_auroc",
            "fold_median_probability_sd",
        ]
    ]
    table = multiseed.merge(audit_columns, on="seed", how="inner", validate="one_to_one")
    table = table[
        [
            "seed",
            "auroc",
            "auprc",
            "sensitivity",
            "specificity",
            "brier",
            "fold_auroc_mean",
            "rank_normalized_pooled_auroc",
            "fold_median_probability_sd",
        ]
    ].sort_values("seed")
    numeric_columns = [column for column in table.columns if column != "seed"]
    mean = {"seed": "Mean"}
    sd = {"seed": "SD"}
    for column in numeric_columns:
        mean[column] = float(table[column].mean())
        sd[column] = float(table[column].std(ddof=1))
    table = pd.concat([table, pd.DataFrame([mean, sd])], ignore_index=True)

    csv_table = table.copy()
    for column in numeric_columns:
        csv_table[column] = pd.to_numeric(csv_table[column], errors="coerce").round(6)
    csv_table.to_csv(output_dir / "table2_multiseed.csv", index=False)

    display = table.copy()
    for column in numeric_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    (output_dir / "table2_multiseed.md").write_text(
        markdown_table(display),
        encoding="utf-8",
    )
    return table


def build_table_3(
    comparison_path: Path,
    tabular_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = []
    baseline_models = [
        ("Age + sex, logistic", "clinical baseline", "age_sex_logistic"),
        ("Age + sex + interval, logistic", "clinical baseline", "age_sex_time_logistic"),
        (
            "Age + sex + interval, gradient boosting",
            "clinical baseline",
            "age_sex_time_gradient_boosting",
        ),
    ]
    for label, role, directory in baseline_models:
        summary = json.loads(
            (tabular_root / directory / "summary.json").read_text(encoding="utf-8")
        )["pooled_oof"]
        comparison_file = tabular_root.parent / (
            f"baseline_comparison_main_vs_{directory}.json"
        )
        comparison_result = (
            json.loads(comparison_file.read_text(encoding="utf-8"))
            if comparison_file.is_file()
            else {}
        )
        rows.append(
            {
                "model": label,
                "role": role,
                "auroc": summary["auroc"],
                "auprc": summary["auprc"],
                "brier": summary["brier"],
                "ece": "",
                "p_vs_primary": comparison_result.get("p_value", ""),
            }
        )

    raw = comparison["raw_probability_metrics"]
    rows.extend(
        [
            {
                "model": "Multiscale residual ECG",
                "role": "primary model, raw probability",
                "auroc": raw["baseline"]["auroc"],
                "auprc": raw["baseline"]["auprc"],
                "brier": raw["baseline"]["brier"],
                "ece": raw["baseline"]["ece"],
                "p_vs_primary": "",
            },
            {
                "model": "Attention ECG",
                "role": "architecture ablation",
                "auroc": raw["attention_v2"]["auroc"],
                "auprc": raw["attention_v2"]["auprc"],
                "brier": raw["attention_v2"]["brier"],
                "ece": raw["attention_v2"]["ece"],
                "p_vs_primary": comparison["delong"]["p_value"],
            },
        ]
    )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "table3_baselines_ablation.csv", index=False)
    display = table.copy()
    for column in ["auroc", "auprc", "brier", "ece"]:
        display[column] = display[column].map(
            lambda value: "" if value == "" else f"{float(value):.4f}"
        )
    display["p_vs_primary"] = table["p_vs_primary"].map(
        lambda value: (
            ""
            if value == ""
            else "<0.001"
            if float(value) < 0.001
            else f"{float(value):.3f}"
        )
    )
    (output_dir / "table3_baselines_ablation.md").write_text(
        markdown_table(display),
        encoding="utf-8",
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready analysis tables")
    parser.add_argument(
        "--metadata",
        default="processed/metadata/metadata_preprocessed.csv",
    )
    parser.add_argument(
        "--multiseed",
        default="runs/main_multiseed_summary.csv",
    )
    parser.add_argument(
        "--audit",
        default="runs/multiseed_oof_audit.csv",
    )
    parser.add_argument(
        "--comparison",
        default="runs/final_model_comparison/final_comparison.json",
    )
    parser.add_argument(
        "--tabular-root",
        default="runs/tabular_baselines",
    )
    parser.add_argument("--output-dir", default="paper_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_1 = build_table_1(Path(args.metadata), output_dir)
    table_2 = build_table_2(
        Path(args.multiseed),
        Path(args.audit),
        output_dir,
    )
    table_3 = build_table_3(
        Path(args.comparison),
        Path(args.tabular_root),
        output_dir,
    )
    print(table_1.to_string(index=False))
    print(table_2.to_string(index=False))
    print(table_3.to_string(index=False))
    print(f"Saved tables in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
