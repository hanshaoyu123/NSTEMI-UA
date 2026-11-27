# NSTEMI ECG analysis code

This repository contains the analysis code for a patient-level deep-learning
study that distinguishes NSTEMI from unstable angina using pre-angiography
12-lead ECGs. It is prepared for the code-availability statement of a
manuscript submitted to *Biomedical Signal Processing and Control*.

The release contains source code, experiment configurations, and unit tests.
It intentionally excludes the public dataset, derived signal arrays, cached
metadata, model checkpoints, prediction files, figures, tables, manuscripts,
and temporary files.

## Data

The source dataset is publicly available from Figshare:

<https://doi.org/10.6084/m9.figshare.29925314>

Download the dataset and arrange it as described in
[`data/README.md`](data/README.md). Data are not redistributed in this
repository. Users should follow the dataset's license and access conditions.

## Environment

- Python 3.11 or newer
- PyTorch 2.4 or newer; CPU execution is supported
- The submitted analysis was run with Python 3.14.6, PyTorch 2.14.0,
  NumPy 2.4.6, pandas 3.0.3, SciPy 1.18.0, and scikit-learn 1.9.0
- A CUDA GPU is recommended for full five-fold training

Create an isolated environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

For a CUDA installation, install the PyTorch build appropriate for the local
CUDA driver before installing `requirements.txt`, following
<https://pytorch.org/get-started/locally/>.

Run the unit tests after installation:

```bash
python -m pip install pytest
python -m pytest -q
```

## Preprocessing

The preprocessing script validates the WFDB files, performs lead-level quality
control, applies the prespecified 0.5-40 Hz zero-phase band-pass filter, selects
one quality-ranked ECG per patient, and creates patient-grouped five-fold
assignments.

```bash
python scripts/preprocess_acs.py \
  --data-dir data \
  --output-dir processed
```

The filtered signal arrays are large; approximately 2.6 GB is required before
other analysis outputs are generated. The arrays are memory-mapped during
training.

## Primary analysis

Seed 42 is the prespecified primary seed. It was not selected using held-out
test performance.

```bash
python scripts/train_cv.py \
  --config configs/main_nstemi_vs_ua.yaml \
  --seed 42
```

The primary configuration trains patient-grouped outer folds with inner
fit/early-stopping/calibration partitions. Temperature scaling and operating
thresholds are then fitted using the existing outer-fold checkpoints:

```bash
python scripts/calibrate_existing_cv.py \
  --run-dir runs/main_nstemi_vs_ua \
  --output-dir runs/main_nstemi_vs_ua_calibrated
```

Key outputs are:

- `runs/main_nstemi_vs_ua_calibrated/oof_predictions.csv`
- `runs/main_nstemi_vs_ua_calibrated/summary.json`
- `runs/main_nstemi_vs_ua_calibrated/fold_metrics.json`
- `runs/main_nstemi_vs_ua_calibrated/final_model/best.pt`

Predict the hidden-label, same-centre holdout records with:

```bash
python scripts/predict_test.py \
  --run-dir runs/main_nstemi_vs_ua_calibrated
```

The holdout labels are not present in the public data. Local test AUROC cannot
therefore be computed; predictions are provided for the official evaluation
platform. This holdout is not an external validation cohort.

## Multi-seed stability

Run the additional prespecified seeds without changing the primary analysis:

```bash
for seed in 43 44 45 46; do
  python scripts/train_cv.py \
    --config configs/main_nstemi_vs_ua.yaml \
    --output-dir "runs/main_nstemi_vs_ua_seed${seed}" \
    --seed "${seed}"
done

python scripts/summarize_seeds.py \
  --runs \
    runs/main_nstemi_vs_ua \
    runs/main_nstemi_vs_ua_seed43 \
    runs/main_nstemi_vs_ua_seed44 \
    runs/main_nstemi_vs_ua_seed45 \
    runs/main_nstemi_vs_ua_seed46 \
  --output runs/main_multiseed_summary.csv

python scripts/audit_multiseed_oof.py \
  --output runs/multiseed_oof_audit.csv
```

The audit reports both conventional pooled OOF metrics and fold-rank-normalized
metrics, which help quantify cross-fold probability-scale effects.

## Robustness and interpretability

After the calibrated primary run has completed:

```bash
python scripts/run_robustness.py \
  --run-dir runs/main_nstemi_vs_ua_calibrated

python scripts/interpret_ecg.py \
  --run-dir runs/main_nstemi_vs_ua_calibrated \
  --max-records 32
```

Robustness outputs are written under
`runs/main_nstemi_vs_ua_calibrated/robustness/` and integrated-gradient outputs
under `runs/main_nstemi_vs_ua_calibrated/interpretability/`.

## Baselines and auxiliary analyses

The main configurations used in the study are:

| Config | Purpose |
|---|---|
| `configs/main_nstemi_vs_ua.yaml` | Primary NSTEMI vs UA analysis |
| `configs/baseline_simple_cnn.yaml` | Shallow CNN baseline |
| `configs/attention_v2_fair.yaml` | Attention architecture ablation |
| `configs/aux_all_comers.yaml` | NSTEMI vs non-NSTEMI |
| `configs/aux_nstemi_vs_stemi.yaml` | NSTEMI vs STEMI |
| `configs/aux_three_class.yaml` | NSTEMI/UA/STEMI classification |
| `configs/aux_median_waveform.yaml` | Median-beat representation ablation |
| `configs/aux_clinical_fusion.yaml` | ECG plus age/sex fusion |
| `configs/robust_all_records.yaml` | Multiple records per patient |
| `configs/robust_raw_unfiltered.yaml` | Unfiltered raw WFDB input |
| `configs/all_comers_calibrated.yaml` | Calibrated all-comers analysis |

Run non-ECG clinical baselines with:

```bash
python scripts/tabular_baselines.py \
  --config configs/main_nstemi_vs_ua.yaml
```

Construct the final paired model comparison with:

```bash
python scripts/final_model_comparison.py \
  --baseline-run runs/main_nstemi_vs_ua_calibrated \
  --attention-run runs/attention_v2_fair \
  --output-dir runs/final_model_comparison
```

Use `scripts/compare_models.py` for a paired DeLong comparison between any two
OOF prediction files and `scripts/subgroup_analysis.py` for prespecified
subgroup metrics.

## Rebuilding manuscript tables and figures

After all analyses above have completed:

```bash
python scripts/build_paper_tables.py \
  --output-dir paper_tables

python scripts/build_manuscript_figures.py
```

Figure-generation defaults expect the calibrated primary run, the multi-seed
audit, robustness results, and interpretability outputs at the run paths listed
above. The script writes publication-sized PNG and PDF figures.

## Reproducibility boundaries

- Splits and normalization are patient-grouped; normalization is fitted inside
  each training partition.
- The hidden test labels are never used for model, hyperparameter, or threshold
  selection.
- The same-centre holdout is described as a holdout, not external validation.
- Saliency maps are descriptive and do not establish pathological mechanisms.
- Full bitwise reproducibility of GPU training is not guaranteed across CUDA,
  PyTorch, or hardware versions. The analysis lock and reported results should
  be interpreted together with the software versions above.

Use `python scripts/<script>.py --help` for the complete command-line options.
