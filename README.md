# Data layout

Download the public ECG dataset from:

<https://doi.org/10.6084/m9.figshare.29925314>

Do not commit downloaded or derived data to this repository. After downloading,
arrange the files as follows:

```text
data/
  CSV/
    train.csv
    test.csv
  row_data/
    00001.dat
    00001.hea
    ...
  med_data/
    00001.med
    ...
```

The file names must match the `ecg_row_record` and `ecg_med_record` values in
the two CSV files. Record identifiers in the CSV may include the `.dat` or
`.med` suffix; the repository scripts handle that format.

Then generate the model-ready arrays and metadata:

```bash
python scripts/preprocess_acs.py \
  --data-dir data \
  --output-dir processed
```

The `processed/` directory is also excluded from version control because it
contains large signal arrays and record-level derived metadata.

The official test labels are hidden. Do not attempt to infer or fabricate test
labels, and do not use test-set feedback to revise the primary model or
threshold.
