# Credit-Card Default Detection

Predict whether a credit-card client will **default** on next month's payment, a
binary classification task on the UCI *"default of credit card clients"* dataset
(30,000 clients, split into a 24,000-client development set and a 6,000-client
evaluation set). The classes are imbalanced (~22% defaulters), so the model is
optimised for **macro-F1** rather than accuracy.

The deployed model is a class-balanced random forest on a set of repayment-semantic
engineered features, with the decision threshold tuned for macro-F1. It is defined in
`config.yaml` (the `chosen` block) and trained from scratch by `main.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the three dataset files (`development.csv`, `evaluation.csv` and
`sample_submission.csv`) in `data/raw/`. The dataset is not committed; the download
link is in the assignment PDF under `docs/`.

## Run

```bash
python main.py
```

This loads the data, trains the chosen pipeline from scratch (about 10 s; well under
the 150 s budget), predicts on the evaluation set, and writes the submission to:

```
outputs/submissions/submission.csv      # header: Id,Predicted
```

## Reproducing the report numbers

```bash
python experiments/tune_baseline.py
```

prints the baseline and final-model cross-validation metrics and saves the
confusion-matrix figures to `outputs/figures/`.

## Layout

```
main.py                  Entry point: load -> train chosen model -> predict -> write submission
config.yaml              Tunable knobs; the deployed model is the `chosen` block
src/                     data loading, preprocessing, model factory, evaluation, submission
experiments/             Model/feature/encoding/tuning experiments (not needed to reproduce the submission)
notebooks/01_eda.ipynb   Exploratory data analysis
report/                  The IEEE report (report.pdf, report.tex)
```

## Optional experiment dependencies

The core pipeline (`main.py`) needs only the packages in `requirements.txt`. Two
experiment scripts use extra libraries that are intentionally **not** required:
`experiments/boosting_experiments.py` needs `lightgbm` (plus an OpenMP runtime), and
`experiments/resampling_experiments.py` needs `imbalanced-learn`. Install them
separately to re-run those sweeps.
