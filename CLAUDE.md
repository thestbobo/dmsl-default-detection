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
config.yaml              Project source of truth for knobs: seed, CV, threshold grid, column groups, chosen config, experiment registry
main.py                  Entry point: load -> preprocess -> train chosen model -> predict -> write submission
src/config.py            Loads config.yaml; exposes SEED, paths, column groups, threshold grid, chosen config, label/id detection
src/data.py              Loaders, split_xy, get_features, get_eval_ids, schema checks
src/preprocessing.py     CodeFolder + engineered features (pay/util/payratio/bill) + encoding variants (pay_remap/scale/log1p/bill_neg) + ColumnTransformer (impute/scale/one-hot)
src/models.py            Estimator factory (make_estimator) + the chosen pipeline: single HGB OR a soft-vote ensemble (chosen.ensemble), both + macro-F1 threshold tuning
experiments/feature_experiments.py  Feature-engineering sweep (P1): per-config OOF macro-F1 + paired-CV validation
experiments/encoding_experiments.py Encoding/preprocessing sweep (P2): screen+paired-CV on BOTH hgb and logreg
experiments/model_experiments.py    Model bake-off + soft-vote ensemble preview (P3->P4): screen, diversity-vs-HGB, paired-CV
experiments/imbalance_experiments.py Imbalance/class-weight sweep (P5): HGB weights + RF balanced, paired-CV
experiments/threshold_experiments.py Threshold transfer-risk audit (P6): honest leave-one-fold-out cut vs fold-avg/fixed strategies
experiments/boosting_experiments.py  LightGBM bake-off (P7, OPTIONAL DEP): screen + paired-CV vs rf_balanced champion (not deployed)
experiments/stacking_experiments.py  Stacking + logistic meta-learner (P8): screen + paired-CV vs rf_balanced champion (not deployed)
experiments/resampling_experiments.py SMOTE/SMOTE+Tomek/etc. (P10, OPTIONAL DEP imblearn): leakage-safe imblearn Pipeline, vs champion (not deployed)
experiments/target_encoding_experiments.py Target encoding via sklearn.TargetEncoder (P11, zero-dep): vs champion (not deployed)
experiments/anomaly_experiments.py   Isolation-Forest anomaly feature (P9, zero-dep): vs champion (not deployed)
experiments/calibration_experiments.py Probability calibration (P12, zero-dep): proves monotonic no-op for thresholded macro-F1 (not deployed)
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

- **Config source of truth:** tunable knobs (seed, CV, threshold grid, column
  groups, the chosen production config, the experiment registry) live in
  `config.yaml`; `src/config.py` loads it. Change the deployed model via the
  `chosen:` block in `config.yaml` — not `main.py`, not `src/models.py`.
- **Seed:** single `SEED = 42` (in `config.yaml`, exposed as `config.SEED`);
  `config.set_seed()` + every estimator gets `random_state=SEED`. No uncontrolled
  randomness.
- **Submission header is exactly `Id,Predicted`** — one row per evaluation row,
  `Id` = evaluation `id` column. Enforced by `submission.validate_submission`.
- **No data leakage:** all transforms are fit inside the sklearn `Pipeline`, so
  they see training folds only. Code-folding is a stateless mapping (no fitting).
- **No external data** (assignment rule).
- **Tuning lives outside `main.py`** (`experiments/`); `main.py` runs only the one
  "best" config (the `chosen:` block of `config.yaml`, assembled in `src/models.py`).
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

# feature-engineering sweep (P1) / encoding sweep (P2, on hgb + logreg)
python experiments/feature_experiments.py
python experiments/encoding_experiments.py

# model bake-off + soft-vote ensemble preview (P3 -> P4)
python experiments/model_experiments.py

# imbalance / class-weight sweep (P5)
python experiments/imbalance_experiments.py

# threshold transfer-risk audit (P6)
python experiments/threshold_experiments.py

# LightGBM bake-off (P7, optional dep: pip install lightgbm + libomp on macOS)
python experiments/boosting_experiments.py

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

## Current leaderboard state (2026-06-21)

- **Best LB is now 0.719** (E28 / P14): `rf_balanced + paysem_util_payratio` engineered
  features + macro-F1 threshold tuning — **DEPLOYED** (`outputs/submissions/submission.csv`,
  byte-identical to `submission_31.csv`). `main.py` reproduces it in **9.9 s**, threshold
  **0.611**, 1,217/6,000 (20.3%) predicted defaults. This **broke the old 0.713 "ceiling"**,
  which was a dev-CV illusion (OOF is decoupled from this LB — see L13).
- **P14 is the live lever: PAY-semantic feature engineering on RF.** The PAY_* codes are
  not a monotone scale (−2 no-consumption, −1 paid-in-full, 0 revolving-credit = riskier,
  1..9 late); the new `paysem` family keeps those states separate (+ latest-month state +
  PAY_0≥2 flag) and `coverx` adds recent coverage/trend. FE-on-RF LB results:
  `+paysem_util_payratio` **0.719**, `+paysem_coverx_payratio` 0.717, `+paysem` 0.716,
  `+payratio` 0.716, `+coverx_payratio` 0.716, `+util` 0.714. Deploy via
  `chosen.feature_groups` (now applied to the single-member `chosen.ensemble=[rf_balanced]`
  path; see `src/models._member_pipeline`). Scripts: `experiments/feature_families_submit.py`
  (FE combos on the champion, argv-selectable) + `experiments/feature_retune_submit.py`
  (`--groups <name>` re-tunes RF on a feature set, L3).
- **Decision rule that changed (L13):** on this data, OOF under-reads the LB in both
  directions — the *lowest*-OOF combo (`paysem_util_payratio`, OOF 0.7107) is the *best* LB.
  So **generate FE-on-RF candidates and let the LB rank them; do not pre-filter on OOF.**
  L5/L6 ("the tree already has the feature"; "FE is a small lever") held for **HGB**, not RF.
- Final-pick candidates: **s31 (0.719)** + **s30/s23 (0.717/0.716)**; S0/S4 (0.712/0.713)
  are superseded. Open: re-tune on the winning set found no OOF gain (subs 43–45, LB pending);
  CatBoost (review item 3, new dep) is the next candidate if FE stalls.
- (Historical, pre-P14) Previous best was **0.713**: S4 / P5 `rf_balanced`; **0.712**: S0
  baseline HGB. P4 soft-vote `hgb + logreg_clean + rf` scored **0.710** (reverted).
- P5 adjacent RF-balanced variants (`leaf10`, `leaf30`, `leaf50`,
  `balanced_subsample`, `max_features=0.5`) did **not** beat `rf_balanced` in CV;
  `balanced_subsample` tied the paired mean delta but had a slightly lower screen
  score. Treat P5 as exhausted unless burning a leaderboard slot on a tiny tie-break.
- **P6 threshold-transfer is done — not a lever.** Honest leave-one-fold-out transfer
  penalty is only **+0.0012** (rf_balanced 0.7115→0.7103) and the cut is tight/stable
  (0.615 ± 0.012); no fold-averaged or flat-fixed threshold beats the deployed per-fit
  `TunedThresholdClassifierCV`. This rules out the threshold as the CV↔LB wobble source.
- **P7 LightGBM is done — not justified, not deployed.** 5 hand specs + a 40-config
  randomized search; the best tuned LightGBM only *ties* `rf_balanced` (dmean **+0.0004**,
  wins 3/5), far below the +0.005 bar a new dependency needs. `lightgbm` is **not** in
  `requirements.txt`; a lazy `kind: lgbm` hook + `boosting_experiments.py` stay as repro
  infra (need `pip install lightgbm` + libomp to re-run).
- **P8–P12 (NotebookLM research round) are done — no lever, nothing deployed.** A
  literature search proposed 5 ideas; all PARKed after a CV-first screen vs `rf_balanced`:
  **stacking** (best +0.0005 paired, 4/5 — noise like the P4 soft-vote), **resampling**
  (SMOTE / SMOTE+Tomek / SMOTE+ENN / Borderline / under-sample — **all negative**, the
  worst of the round; `imblearn` **not** added), **Isolation-Forest anomaly feature**
  (−0.0001), **target encoding** (−0.0001; via zero-dep `sklearn.TargetEncoder`), and
  **calibration** (a structural no-op — a single threshold on `predict_proba` is rank-based
  and calibration is monotonic). See L12. Two reusable rules: don't calibrate for a
  thresholded metric; for tree ensembles + macro-F1 prefer `class_weight` + threshold
  tuning over resampling.
- **Where we stand — the 0.713 "ceiling" (L11/L12) was broken by P14 FE-on-RF (L13).**
  The earlier "ceiling" was reached by screening on dev OOF, which is decoupled from this LB.
  Respecting the PAY_* code semantics (the `paysem` family) lifted the LB to **0.719**.
  Remaining effort: confirm subs 32–45 on the LB, optionally CatBoost (item 3) /
  seed-averaging (item 5), the **IEEE report**, and the **final 2-submission pick**
  (s31 0.719 + the best runner-up). The pre-P14 conclusion ("don't keep swapping learners")
  still holds — the lever is *features on RF*, not new learners.
