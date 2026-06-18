# data/raw/

The dataset is **not** committed to this repository (and must never be).

## Download

Get the archive from the Google Drive link in the assignment:

https://drive.google.com/file/d/1N07AVjwrKL2fNJNBfSlDef2ynhR7nlTq/view?usp=sharing

Unzip it and place these three files **directly in this folder**:

```
data/raw/development.csv      # training data (features + label)
data/raw/evaluation.csv       # features only (+ id column) — to predict
data/raw/sample_submission.csv
```

## Notes

- `development.csv` has the target column (named `label` or
  `default.payment.next.month`); `evaluation.csv` does not.
- The submission uses the evaluation `id` column as the `Id` field.
- This folder is gitignored (`data/raw/*.csv`) — do not commit the data.
