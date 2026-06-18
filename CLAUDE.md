# CLAUDE.md

LLM-context guide for this repository. Keep it accurate when the code changes.

## Project

Predict whether a credit-card client will **default** on next month's payment —
a **binary classification** task on the UCI "default of credit card clients"
dataset (30,000 clients). Class `0 = no default`, `1 = default`. The data is
**imbalanced** (non-default dominates), so the optimisation metric is
**macro-F1**, not accuracy. PoliTo DSML Lab project, Summer 2026.

## Dataset

Columns: `ID`, `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`,
`PAY_0`, `PAY_2`..`PAY_6`, `BILL_AMT1`..`BILL_AMT6`, `PAY_AMT1`..`PAY_AMT6`,
and the target.

- **Categorical (nominal):** `SEX`, `EDUCATION`, `MARRIAGE` → one-hot encoded.
- **Numeric / ordinal:** everything else, incl. `PAY_*` (kept as ordinal numeric).
- **Target:** `0` / `1`.

### Data quirks (handled in `src/preprocessing.py`)

- `EDUCATION` has undocumented `{0, 5, 6}` → folded into `4` ("other").
- `MARRIAGE` has undocumented `{0}` → folded into `3` ("other").
- `PAY_*` may contain `-2` and `0` in the real data (spec documents only `-1`, `1..9`).
- **Label column** is detected robustly: `label` *or* `default.payment.next.month`
  (case-insensitive) — see `config.detect_label_col`.
- **ID handling:** `ID` is dropped from features; the evaluation `id` column is kept
  only to build the submission — see `config.detect_id_col`.

## Repo map

```
main.py                  Entry point: load -> preprocess -> train chosen model -> predict -> write submission
src/config.py            SEED, paths, column groups, robust label/id detection
src/data.py              Loaders, split_xy, get_features, get_eval_ids, schema checks
src/preprocessing.py     CodeFolder (undocumented codes) + ColumnTransformer (impute/scale/one-hot)
src/models.py            Candidate models + the single chosen pipeline (HGB + macro-F1 threshold tuning)
src/evaluate.py          StratifiedKFold CV: macro-F1, per-class metrics, ROC-AUC, confusion-matrix figures
src/submission.py        write_submission / validate_submission (strict Id,Predicted format)
experiments/tune_baseline.py   Model comparison + HGB grid search (tuning lives here, NOT in main.py)
notebooks/01_eda.ipynb   EDA starter (class balance, distributions, quirks, missingness)
data/raw/                Dataset (gitignored; see its README for the download link)
outputs/submissions/     submission.csv (gitignored)
outputs/figures/         Saved figures, e.g. confusion matrices (gitignored)
report/                  IEEE report placeholder + rules (no .tex committed)
```

## Conventions

- **Seed:** single `SEED = 42` in `config.py`; `config.set_seed()` + every
  estimator gets `random_state=SEED`. No uncontrolled randomness.
- **Submission header is exactly `Id,Predicted`** — one row per evaluation row,
  `Id` = evaluation `id` column. Enforced by `submission.validate_submission`.
- **No data leakage:** all transforms are fit inside the sklearn `Pipeline`, so
  they see training folds only. Code-folding is a stateless mapping (no fitting).
- **No external data** (assignment rule).
- **Tuning lives outside `main.py`** (`experiments/`); `main.py` runs only the one
  "best" config from `src/models.py`.
- **Build at runtime:** no pre-computed artifacts / pre-trained models; `main.py`
  trains from scratch and must finish in **≤ 150 s**.

## Common commands

```bash
# activate the provided environment
source .venv/bin/activate

# end-to-end: train + write outputs/submissions/submission.csv
python main.py

# model comparison + HGB tuning (generates report numbers + figures)
python experiments/tune_baseline.py

# EDA
jupyter notebook notebooks/01_eda.ipynb

# regenerate the submission = just re-run
python main.py
```

(Requires the three CSVs in `data/raw/` first — see `data/raw/README.md`.)

## Do NOT

- ❌ Commit the dataset (`data/raw/*.csv` is gitignored — keep it that way).
- ❌ Let evaluation data influence any fitted transform/model (no leakage).
- ❌ Change the submission header — it must stay exactly `Id,Predicted`.
- ❌ Put hyperparameter tuning / model search in `main.py` (keep it in `experiments/`).
- ❌ Rely on pre-computed artifacts or pre-trained models — build everything at runtime.
