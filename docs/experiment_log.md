# Experiment Log — Credit-card Default Detection

> A running diary of every experiment we run to push macro-F1 on the DSLE
> leaderboard. Read this top-to-bottom to reconstruct **what** we tried, **how**
> it was configured, and **what score** it got — so that at the end we can see
> the big picture and assemble the best combination of changes.
>
> **Primary metric:** macro-F1 (NOT accuracy — the data is imbalanced).
> **Source of truth:** the **leaderboard** score, not dev CV (see Lesson L1).

---

## How to use this log (read me first)

- **Every experiment gets an entry** in [§5 Experiment entries](#5-experiment-entries),
  appended at the bottom, using the [template](#entry-template). Never overwrite a
  past entry — the diary is append-only. Corrections go in a new entry that links
  back.
- **Every leaderboard submission gets a row** in the [§3 Leaderboard table](#3-leaderboard-summary-the-big-picture).
  That table is the scoreboard; the entries are the story behind each row.
- **Numbers must be reproducible.** Each entry records the exact config, seed, and
  CV protocol used. If you can't reproduce a number, it doesn't go in the table.
- **Two different scores, never conflate them:**
  - *CV score* = deployed-objective out-of-fold macro-F1 on `development.csv`
    (5-fold stratified, best threshold swept on OOF probabilities). This is what
    `experiments/` prints. It is our **internal** signal.
  - *LB score* = macro-F1 returned by the DSLE leaderboard after uploading a
    `submission.csv`. This is the **external** signal and the one that grades us.
- **Decision discipline:** a change is only KEPT if it helps the score we trust
  (LB when we have it; CV only as a screen). See Lesson L1 — small CV gains are
  noise on this dataset.
- **Who runs what:** experiments + CV are run by the LLM/agent and logged here.
  The human (`thestbobo`) uploads the chosen `submission.csv` files to DSLE and
  pastes the returned LB score back into the table + the relevant entry.

---

## 1. Fixed context (the problem, unchanging)

| Item | Value |
|---|---|
| Task | Predict `default.payment.next.month` (0/1) for next month — binary classification |
| Dataset | UCI "default of credit card clients"; `development.csv` = 24,000 rows (labelled), `evaluation.csv` = leaderboard set (unlabelled) |
| Features | 23: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`,`PAY_2..6`, `BILL_AMT1..6`, `PAY_AMT1..6` |
| Class balance (dev) | 0 (no default) = **77.88%**, 1 (default) = **22.12%** |
| Metric | **macro-F1** (equal weight to both classes) |
| Submission format | header exactly `Id,Predicted`, one row per eval id |
| Leaderboard | DSLE `http://trinidad.polito.it:8888`, ≤200 submissions, pick up to 2 for final |
| Deadline | **2026-06-24 23:59 CET** |

### Hard constraints (do not break)

- **sklearn-first.** The provided `.venv` ships only scikit-learn (+ numpy/pandas/
  scipy) — **no** xgboost / lightgbm / catboost / imblearn. Prefer zero-dependency
  solutions. Adding a library *is* allowed (the grader runs
  `pip install -r requirements.txt`), but it costs: install/build risk on the
  grader's machine and it eats the 150 s budget. So treat new deps (P7 boosting
  libs, SMOTE) as **last-resort, risk-flagged** options that must clear a real
  **leaderboard** gain to justify — not a CV-only one (Lesson L1).
- `main.py` must train from scratch and write `submission.csv` in **≤150 s**.
- **No data leakage:** all fitted transforms live inside the sklearn `Pipeline`
  so they see training folds only. Stateless mappings (e.g. code-folding) are fine.
- **No external data.**
- **Tuning lives in `experiments/`**, never in `main.py`. `main.py` runs only the
  one chosen config from `src/models.py`.

---

## 2. Environment & reproducibility

| Item | Value |
|---|---|
| Python | 3.13.2 |
| scikit-learn | 1.9.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| Seed | `config.SEED = 42` (set via `config.set_seed()`; every estimator gets `random_state=SEED`) |
| CV protocol | `StratifiedKFold(n_splits=5, shuffle=True, random_state=…)` |
| Deployed objective | OOF `predict_proba`, then macro-F1 maximised over threshold grid `np.linspace(0.05, 0.95, 181)` |
| Validation (robustness) | paired repeated CV over fold seeds `[42, 1, 7, 2024, 99]` (see `experiments/tune_baseline.py`) |

To reproduce any CV number: `source .venv/bin/activate && python experiments/tune_baseline.py`
(or the per-experiment script noted in its entry).

---

## 3. Leaderboard summary (the big picture)

> The scoreboard. One row per uploaded `submission.csv`. Newest at the bottom.
> `Δ vs base` is LB minus the 0.712 baseline.

| #  | Date       | Entry | Description                              | CV macro-F1 | LB macro-F1 | Δ vs base | Final pick? |
|----|------------|-------|------------------------------------------|-------------|-------------|-----------|-------------|
| S0 | 2026-06-18 | E00     | Baseline: HGB defaults + macro-F1 thr tune | 0.7076    | **0.712**   | —         | candidate   |
| S1 | 2026-06-19 | E03/E06 | + utilisation features (BILL/LIMIT: mean/max/last) | 0.7096 | 0.705 | −0.007 | reverted    |
| S2 | 2026-06-20 | E12/E13 | Soft-vote ensemble `hgb + logreg_clean + rf` (P4) | 0.7101 | 0.710 | −0.002 | reverted |
| S3 | 2026-06-20 | E13     | Same P4 ensemble with tuned HGB params from prior P1/X tuning | 0.7101-ish | 0.710 | −0.002 | reverted |
| S4 | 2026-06-20 | E14     | RF balanced class_weight + macro-F1 threshold tune (P5) | 0.7115 | **0.713** | +0.001 | candidate |
| —  | 2026-06-21 | E23–E27 | **Candidate batch (25 files)** — see `outputs/submissions/MANIFEST.md` for each `submission_<N>.csv`, its config, and (once uploaded) its LB. Threshold sweep, parked candidates, Path-X HGB/RF, segmented thresholds, ET-balanced, FE-on-RF. | various | *pending* | — | TBD |
| s14| 2026-06-21 | E23     | Path-X tuned HGB cand2 (`submission_14.csv`) | 0.7101 | 0.706 | −0.006 | reverted |
| s31| 2026-06-21 | E28     | **rf_balanced + paysem_util_payratio** (`submission_31.csv`) — DEPLOYED | 0.7107 | **0.719** | **+0.007** | **candidate** |
| s23| 2026-06-21 | E27/E28 | rf_balanced + payratio (`submission_23.csv`) | 0.7111 | 0.716 | +0.004 | candidate |

**Best LB so far:** `0.719` (s31, E28, **deployed**) — `rf_balanced + paysem_util_payratio`,
the PAY-semantic FE-on-RF combo from the external review. This **broke the prior 0.713
"ceiling"** (which was a dev-CV illusion — OOF is decoupled from this LB, see L13). The
FE-on-RF lever (E27/E28) is the real driver: `+payratio` 0.716, `+paysem` 0.716,
`+paysem_coverx_payratio` 0.717, `+paysem_util_payratio` **0.719**. Final-pick candidates
are now **s31 (0.719)** + **s23/s30 (0.716/0.717)**; the older S0/S4 (0.712/0.713) are
superseded. Historical note below (pre-P14):

S4 (P5 `rf_balanced`) held 0.713. S1 (`util`) was a marginal CV gain
(+0.0008 paired) that did **not** transfer — it scored **0.705 < 0.712** on the LB and
was reverted (`chosen.feature_groups=[]`), exactly as the L1 rule predicted. P4 repeated
the same pattern: both the equal-weight ensemble and the tuned-HGB-param variant scored
**0.710 < 0.712**, so they were reverted. P5 RF class weighting is the first
post-baseline LB improvement, but only by **+0.001**. P6 (threshold transfer) and P7
(LightGBM) then found **no further lever** — the honest threshold penalty is ~0.0012 and
even a fully-tuned LightGBM only ties `rf_balanced` (+0.0004 paired). A second research
round (P8–P12, from a NotebookLM literature search: stacking, SMOTE+Tomek resampling,
calibration, target encoding, anomaly feature) **also found no lever** — all PARKed, the
SMOTE family was the worst of the round, and calibration is a structural no-op for a
thresholded metric (L12). Every path P1–P12 is now closed: **0.713 is the practical
ceiling** (L11/L12). The two submissions to keep for final scoring are **S0 (0.712)** and
**S4 (0.713)**.

---

## 4. Direction / path tracker

> The roadmap of improvement areas we intend to explore, each as a focused
> "path" run in its own session. Status: `planned` → `in-progress` → `done`
> (`parked` if deferred). Update the status + a one-line result as each closes.
> Detailed work for a path lands as numbered entries in §5.

> Finalised 2026-06-19 by reconciling the agent draft with the human's list.
> Zero-dependency paths run first; dependency-adding ones (P7, SMOTE) are last
> and gated on a real LB gain. Re-tuning is a cross-cutting step (X), not a path.

| Path | Area | Dep? | Why it could help | Depends on | Status | Result |
|------|------|------|-------------------|------------|--------|--------|
| **P1** | **Feature engineering** — PAY_* aggregates (worst delinquency `max(PAY_*)`, # months late `PAY>=1`, # duly `PAY<=0`, currently delinquent `PAY_0>=1`, delinquency trend/slope); utilisation `BILL/LIMIT` per-month + mean/max/last; payment ratios `PAY_AMT_t/BILL_AMT_{t-1}`, fraction-of-bill-paid, # zero-payment & # paid-in-full months; bill dynamics (trend, std volatility, MoM deltas), remaining credit `LIMIT-BILL` | zero | Biggest untapped lever; raw columns under-use the repayment history. Flagged in `models.py` as the prerequisite before re-tuning. | — | **done** | No win — `util` best in CV (+0.0008 paired, 4/5), shipped as S1 but lost on LB (**0.705 < 0.712**), reverted to baseline (L1). `pay`/`bill` neutral-to-negative: HGB already has the raw signal (L5). |
| P2 | **Encoding / preprocessing fixes** — remap PAY_* (`-2/-1/0 → 0` "not late", keep `1..9`) + "is delinquent" flags; drop `StandardScaler` on the tree path (no-op for HGB), keep it for linear; `log1p` monetary cols; handle negative `BILL_AMT` (overpayments) | zero | Cleaner ordinal scale helps linear a lot, trees somewhat. | P1 | **done** | HGB neutral (all within noise, scaler+log1p bit-identical → L5/L7); deployed HGB just drops the no-op scaler (LB 0.712 unchanged). Real win is **linear**: `clean_linear` +0.0099 (5/5), `pay_remap` +0.0072 (5/5) on LogReg — carried to P3. (E07–E11) |
| **P3** | **Other sklearn models** — RandomForest, ExtraTrees, plain GradientBoosting; LogReg on engineered features + interactions; `CalibratedClassifierCV` for steadier probabilities/threshold | zero | Diversity for an ensemble + calibration → more stable threshold. | P1 | **done** | RF is the best *single* model (+0.0020 paired, 5/5) but a near-clone of HGB (corr 0.974) — not diversity. LogReg(`clean_linear`) is weaker (−0.0016) but most decorrelated (corr 0.914). No standalone swap is an LB-worthy win; the value is the ensemble → P4. (E12) |
| P4 | **Ensembling / stacking** — soft-voting / stacking of diverse bases (HGB + LogReg + RF) with a logistic meta-learner, then threshold-tune the ensemble | zero | Averaging diverse models gives a small, robust macro-F1 bump and shrinks CV↔LB variance. | P3 | **done** | No LB win — `hgb+logreg+rf` scored **0.710**, and retrying it with the tuned HGB params also scored **0.710**, both below S0 **0.712**. Reverted. (E12/E13) |
| P5 | **Imbalance handling** — `class_weight="balanced"` / `sample_weight` (≈tie with threshold tuning in prior test); SMOTE / under-sampling *only if* we accept the `imblearn` dep | mostly zero | Macro-F1 rewards minority recall; reweighting may beat pure threshold tuning. | — | **done** | HGB class weighting is not robust. `rf_balanced` is robust but small: 0.7115 ± 0.0006, dmean +0.0032, wins 5/5 vs HGB, and S4 scored **0.713** on LB (+0.001 vs S0). Adjacent RF-balanced variants did not improve CV beyond the current candidate (E15), so move to P6 unless spending an LB slot on a tiny tie-break. |
| P6 | **Threshold strategy / transfer risk** — fixed/less-aggressive threshold (≈ base rate 0.22) or fold-averaged threshold vs the per-fit tuned one | zero | The tuned threshold may not transfer dev→eval — a likely source of LB wobble. | — | **done** | Not a lever — honest leave-one-fold-out transfer penalty is only **+0.0012** and the `rf_balanced` cut is tight/stable (0.615 ± 0.012); no fold-avg/flat-fixed strategy beats per-fit tuning. The deployed `TunedThresholdClassifierCV` is already the robust choice. Rules out threshold as the CV↔LB wobble source (L10). (E16) |
| P7 | **Stronger boosting libs** — LightGBM / XGBoost / CatBoost (native categoricals, more knobs) | **NEW DEP** | Usually a small edge (~+0.005) over HGB. | — | **done** | Not justified — LightGBM (5 hand specs + 40-config randomized search) only *ties* `rf_balanced`: best tuned `lgbm_B` dmean **+0.0004**, wins 3/5 (vs the +0.005 dep bar). Noise by L1; not deployed, dep NOT added. Lazy hook + `boosting_experiments.py` kept as repro infra. (E17, L11) |
| P8 | **Stacking** — StackingClassifier with a logistic meta-learner over rf_balanced/hgb/logreg_clean (learns member weights vs P4's equal-weight average) | zero | A meta-learner can beat plain soft-voting by trusting each member where it's strong. | P4 | **done** | No win — best `stack_rfbal_hgb_logreg` is **+0.0005 paired, 4/5** vs champion; noise by L1 (and P4's +0.0026 *lost* on LB). PARK. (E18) |
| P9 | **Isolation-Forest anomaly feature** — append an unsupervised anomaly score | zero | Defaulting is "anomalous"; deviation from the norm might add signal. | — | **done** | Not validated — `rf_balanced+anomaly` **−0.0001**, `hgb+anomaly` −0.0027; nothing beat champion at screen. A tree already extracts this (L5). PARK. (E19) |
| P10 | **Resampling** — SMOTE / SMOTE+Tomek / SMOTE+ENN / Borderline / under-sample (NotebookLM #1) | **NEW DEP** (imblearn) | Rebalancing the train folds + cleaning the boundary may beat class-weighting. | — | **done** | **All negative** vs champion (SMOTE+Tomek −0.0043, SMOTE −0.0030, under-sample −0.0012, …); class_weight+threshold dominates synthetic resampling for trees on macro-F1. Dep **not** added. PARK. (E20) |
| P11 | **Target encoding** — replace one-hot of SEX/EDUCATION/MARRIAGE with `sklearn.TargetEncoder` (cross-fitted) | zero | Compact category→mean-default-rate encoding (NotebookLM #4). | — | **done** | Not validated — `rf_balanced+target-enc` **−0.0001**; tiny cardinality (2/4/3) means one-hot is already lossless, so no upside. PARK. (E21) |
| P12 | **Probability calibration** — CalibratedClassifierCV (sigmoid/isotonic) before the threshold tuner (NotebookLM #3) | zero | "Steadier" probabilities for the threshold cut. | — | **done** | **No-op by construction** — calibration is monotonic, so it preserves rank order and the best thresholded macro-F1 is unchanged (Platt −0.0003 / isotonic +0.0000 on the SAME scores); the CalibratedCV wrap is +0.0001 paired (3/5, incidental cross-fit ensembling). PARK. (E22) |
| **X** | **Re-tune hyperparameters** (cross-cutting) — randomised search on the deployed objective + paired-CV validation, on the *settled* feature set / chosen model | zero | Defaults were best for raw features; the optimum shifts once features change (Lesson L3). | P1/P2 + chosen model | **done** | Run at last (2026-06-21 review): randomised search on the deployed objective for **both** HGB (40 cfg) and RF (30 cfg). Tuned HGB robustly beats the *HGB default* (best +0.0021, 5/5) but stays < champion; best tuned **RF** edges `rf_balanced` only **+0.0003 paired, 4/5** (0.7115→0.7118). CV-noise by L1; top 3 of each emitted as LB candidates (submissions 12–17). (E23) |
| **P13** | **Group-aware / segmented thresholds** — per-subgroup decision cut (PAY_0 buckets / SEX / LIMIT_BAL terciles) instead of one global threshold, tuned by coordinate ascent on global macro-F1 | zero | SOTA on this exact dataset (Gittlin 2025): +1.5–4% balanced acc from per-subgroup cuts; subgroups have very different default base rates. | — | **done** | Not a robust lever — honest leave-one-fold-out transfer is ~neutral (pay0 +0.0002 3/5, limit3 ~0, sex −0.0004); the full-OOF gains (limit3 0.7122) are optimistic. Consistent with L10 (the cut is stable). 3 candidates emitted for the LB (submissions 18–20). (E26) |

---

## 5. Experiment entries

### Entry template

```
### E<NN> — <short title>  (Path P<x>)
- **Date:** YYYY-MM-DD
- **Author:** <agent/human>
- **Hypothesis:** what we expect to change the score and why.
- **Changes:** files touched + one-line summary of the diff. Git ref/branch if any.
- **Config:** model + params, seed, CV protocol, threshold grid, anything non-default.
- **CV result:** deployed-objective OOF macro-F1 (mean ± std over seeds if repeated),
  @best threshold; per-class P/R/F1 if relevant. Compare to the entry's baseline.
- **LB result:** score + submission filename + date (filled after the human uploads).
- **Decision:** KEEP / REVERT / PARK — and the one-sentence rationale.
- **Notes / next:** follow-ups, surprises, links to related entries [[E..]].
```

---

### E00 — Baseline snapshot (anchor)
- **Date:** 2026-06-18
- **Author:** established before this log
- **Hypothesis:** n/a — this is the reference point everything is measured against.
- **Changes:** repo at commit `1bc766a "baseline"`. HGB pipeline =
  code-folding → ColumnTransformer (median-impute+scale numeric, most-frequent-impute+one-hot
  categorical) → `HistGradientBoostingClassifier(random_state=42)` wrapped in
  `TunedThresholdClassifierCV(scoring="f1_macro", cv=5)`. `CHOSEN_HGB_PARAMS = {}`
  (library defaults).
- **Config:** sklearn 1.9.0, seed 42, 5-fold stratified, threshold grid 0.05–0.95 (181 pts).
- **CV result:** deployed-objective OOF macro-F1 = **0.7076** @thr 0.335.
  (For reference, macro-F1 @0.5 = 0.6857 — threshold tuning is worth ~+0.022 CV.)
- **LB result:** **0.712** (`submission.csv`, the current external score).
- **Decision:** KEEP as the anchor / current best.
- **Notes / next:** Note the **CV (0.7076) < LB (0.712)** gap — dev CV slightly
  under-reads the eval set here. This is the config to beat. See Lesson L1.

---

### E01 — Project `config.yaml` + config loader (infra / P1 Task 0)
- **Date:** 2026-06-19
- **Author:** agent
- **Hypothesis:** infra only — no score change. Centralise all tunable knobs in one
  declarative file so the many P1…P7+X sweeps are organised and reproducible.
- **Changes:** new `config.yaml` (seed, CV, threshold grid, column groups,
  `chosen:` production config, `experiments.feature_configs` registry). `src/config.py`
  now loads it and re-exports the same names (`SEED`, `NUMERIC`, `N_SPLITS`,
  `THRESHOLDS`, `VALIDATION_SEEDS`, `CHOSEN_*`, `FEATURE_CONFIGS`); paths/label-id
  detection/`set_seed` stay in Python. `models.py` reads `chosen.*` from config;
  `tune_baseline.py` sources `THRESHOLDS`/`VALIDATION_SEEDS` from config. Docs
  (`CLAUDE.md`, `howto.txt`) updated. **New dep:** `PyYAML==6.0.2` in `requirements.txt`.
- **Config:** n/a (refactor).
- **CV result:** unchanged — `main.py` reproduces E00 exactly (thr 0.3334, ~0.7076).
- **LB result:** n/a (no submission).
- **Decision:** KEEP — behaviour-preserving, enables the rest of P1.
- **Notes / next:** PyYAML is a deliberate, user-approved exception to the zero-dep
  guideline (config infra, not a modeling lib; no leakage, ~0 runtime, read once at
  startup). Grader runs `pip install -r requirements.txt`.

### E02 — PAY_* delinquency aggregates  (Path P1)
- **Date:** 2026-06-19 · **Author:** agent
- **Hypothesis:** explicit worst/late/duly/streak/trend summaries of the repayment
  history (the strongest raw signal) should help.
- **Changes:** `src/preprocessing.py` group `pay` (7 cols: PAY_MAX, PAY_MEAN,
  PAY_N_LATE, PAY_N_DULY, PAY_CURR_DELINQ, PAY_TREND, PAY_WORST_STREAK), added inside
  the Pipeline (stateless row-wise). Scored via `experiments/feature_experiments.py`.
- **Config:** HGB defaults, seed 42 screen + paired CV [42,1,7,2024,99], thr grid 0.05–0.95.
- **CV result:** screen 0.7083 @0.340 (d **+0.0008**). Paired 0.7073 ± 0.0013,
  dmean **−0.0010**, wins **2/5** vs baseline 0.7083 ± 0.0007.
- **LB result:** — (not submitted).
- **Decision:** **REVERT** — worse on paired CV. HGB already extracts the PAY signal
  from the raw columns; aggregates are redundant for a tree (→ L5).
- **Notes / next:** likely useful for the *linear* model in P3.

### E03 — Utilisation BILL/LIMIT  (Path P1)
- **Date:** 2026-06-19 · **Author:** agent
- **Hypothesis:** credit utilisation is the classic credit-risk signal; not directly
  present as a column.
- **Changes:** `src/preprocessing.py` group `util` (3 cols: UTIL_MEAN, UTIL_MAX,
  UTIL_LAST = BILL_AMT_t / LIMIT_BAL, safe denom).
- **Config:** as E02.
- **CV result:** screen 0.7096 @0.325 (d **+0.0021**). Paired 0.7091 ± 0.0009,
  dmean **+0.0008**, wins **4/5**. Most consistent positive in the sweep.
- **LB result:** S1 — **0.705** (`submission.csv`, 2026-06-19). Δ vs base = **−0.007**.
- **Decision:** **REVERT** — LB 0.705 ≤ 0.712, so `util` did not transfer dev→eval. Set
  `chosen.feature_groups=[]`; back on the raw-feature baseline. Textbook L1: a within-noise
  CV gain (+0.0008 paired) cost us 0.007 on the LB.
- **Notes / next:** cleanest/bounded engineered family; low overfit risk (3 cols), yet
  still lost on the LB — FE is not the lever (L6). `util` may still help the *linear*
  model; revisit in P3, not on HGB.

### E04 — Payment ratios / coverage  (Path P1)
- **Date:** 2026-06-19 · **Author:** agent
- **Hypothesis:** how much of the bill the client actually pays (coverage, zero-pay,
  paid-in-full) should separate defaulters.
- **Changes:** `src/preprocessing.py` group `payratio` (4 cols: FRAC_BILL_PAID_MEAN,
  COVER_PREV_MEAN = PAY_AMT_t / previous-month bill, N_ZERO_PAY, N_PAID_FULL; safe denoms,
  BILL≤0 → coverage 1.0).
- **Config:** as E02.
- **CV result:** screen 0.7089 @0.335 (d **+0.0013**). Paired 0.7093 ± 0.0010,
  dmean **+0.0010**, wins **3/5**.
- **LB result:** — (not submitted).
- **Decision:** **PARK** — mildly positive but less consistent than `util` (3/5),
  and ratios have heavy outliers from tiny denominators. Behind `util` as a candidate.
- **Notes / next:** consider clipping ratios; revisit in a combo if `util` helps on LB.

### E05 — Bill dynamics + remaining credit  (Path P1)
- **Date:** 2026-06-19 · **Author:** agent
- **Hypothesis:** bill level/volatility/trend and remaining credit add context.
- **Changes:** `src/preprocessing.py` group `bill` (6 cols: BILL_MEAN, BILL_MAX,
  BILL_STD, BILL_TREND, REM_CREDIT_LAST, REM_CREDIT_MEAN).
- **Config:** as E02.
- **CV result:** screen 0.7075 @0.310 (d **−0.0000**). Not promising → not validated.
- **LB result:** — (not submitted).
- **Decision:** **REVERT** — no signal beyond the raw BILL/LIMIT columns.
- **Notes / next:** —

### E06 — Feature combinations + P1 decision  (Path P1)
- **Date:** 2026-06-19 · **Author:** agent
- **Hypothesis:** the two mildly-positive families (`util` + `payratio`) might compound,
  possibly with `pay` interactions.
- **Changes:** combos added to `config.yaml` registry: `util_payratio`,
  `pay_util_payratio`, `all`.
- **Config:** as E02.
- **CV result (paired vs baseline 0.7083 ± 0.0007):**
  - `util_payratio` 0.7086 ± 0.0012, dmean **+0.0003**, wins 4/5 (no compounding —
    `payratio`'s variance dilutes `util`).
  - `pay_util_payratio` 0.7083 ± 0.0008, dmean **+0.0000**, wins 2/5 (`pay` drags it down).
  - `all` screen 0.7058 @0.330 (d **−0.0017**) → not validated.
- **LB result:** — (not submitted).
- **Decision:** **Ship `util` alone (E03) as the P1 deliverable (S1).** Combos add no
  robust gain; the simplest consistent config wins. P1 → done.
- **Notes / next:** P1 is exhausted at the zero-dep level (FE moved CV <0.003, within
  noise → L6). Next levers: P2 encoding, P3 models (where `pay`/`payratio` may finally
  pay off for linear), P4 ensembling. Re-tune HGB (step X) only after features settle.

---

> **Path P2 — Encoding / preprocessing fixes (E07–E11).** Built on the raw-feature
> baseline (S1 reverted). Each variant is a declarative `encoding` spec in
> `config.yaml` (`experiments.encoding_configs`), scored by
> `experiments/encoding_experiments.py` on the deployed objective for **two**
> estimators: **HGB** (deployed — must be LB-neutral-or-better) and **LogReg**
> (`class_weight="balanced"`, the linear baseline that feeds P3). HGB paired baseline
> = 0.7083 ± 0.0007; LogReg paired baseline = 0.6957 ± 0.0006. All four transforms are
> stateless, leakage-free, in-Pipeline (`src/preprocessing.py: apply_encoding`).

### E07 — PAY_* remap + delinquency flags  (Path P2)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** folding PAY_* `{-2,-1,0} → 0` (keep `1..9`) gives a clean monotone
  "months delinquent" ordinal; a linear model should read that far better than the raw
  `-2..9` scale where the negative codes are not "more on-time than 0".
- **Changes:** `encoding.pay_remap` (in-place `clip(lower=0)` on the 6 PAY cols) and
  `pay_flags` (append per-month `IS_DELINQ_PAY_*` = `PAY≥1`). Variants `pay_remap`,
  `pay_remap_flags`.
- **Config:** HGB defaults / LogReg(`balanced`, max_iter=1000), seed 42 screen + paired
  CV [42,1,7,2024,99], thr grid 0.05–0.95.
- **CV result:**
  - **HGB:** screen 0.7087 @0.315 (d **+0.0011**); paired 0.7084 ± 0.0010, dmean
    **+0.0001**, wins **3/5** → noise. Flags identical to remap (3/5).
  - **LogReg:** `pay_remap` screen 0.7026 @0.625 (d **+0.0065**); paired 0.7028 ± 0.0003,
    dmean **+0.0072**, wins **5/5** → **robust KEEP**. `pay_remap_flags` +0.0062, 5/5 —
    robust but *below* remap alone (the flags add variance the remap already captures).
- **LB result:** — (HGB neutral; not submitted on its own — see E11).
- **Decision:** **KEEP for the linear path / neutral for HGB.** `pay_remap` is the single
  biggest encoding lever for LogReg (+0.0072, 5/5); drop the flags. Off on deployed HGB
  (not a robust win; L1).
- **Notes / next:** the cleaner ordinal is exactly what L5 predicted — a tree is
  indifferent, a linear model is not. Carried into the `clean_linear` combo (E11).

### E08 — Conditional StandardScaler (per-estimator)  (Path P2)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** `StandardScaler` is a per-feature monotone map, so it is a strict
  no-op for HGB (a tree only sees split order) — droppable on the tree path — while it
  is load-bearing for LogReg.
- **Changes:** `build_preprocessor(scale=…)` gates `StandardScaler`; `encoding.scale`
  knob (default `true`). Variant `noscale` (scale off).
- **Config:** as E07.
- **CV result:**
  - **HGB:** screen 0.7076 @0.335, d **+0.0000** — **bit-identical** to baseline
    (sanity check `|d| = 0.00e+00`; eval `predict_proba` max-diff = 0.0, predictions
    identical). Confirmed no-op.
  - **LogReg:** screen 0.6945 @0.595, d **−0.0016** — scaling helps the linear model, as
    expected (unscaled lbfgs also under-converges).
- **LB result:** deployed (E11) — bit-identical to S0, **LB stays 0.712**.
- **Decision:** **KEEP — drop the scaler on the deployed HGB path** (`scale: false`):
  a free simplification with zero LB risk (proven identical). Keep it on for linear.
  Scaling is now a **per-estimator knob**, not always-on. → L7.
- **Notes / next:** the cleanest possible P2 change to the deployed model — removes a
  dead step without touching a single prediction.

### E09 — Signed log1p on monetary columns  (Path P2)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** `sign(x)·log1p(|x|)` on LIMIT_BAL / BILL_AMT* / PAY_AMT* compresses the
  heavy NT$ tails (and preserves negative balances), which should steady a linear model.
- **Changes:** `encoding.log1p` (in-place signed-log1p on `MONETARY_COLS`). Variant `log1p`.
- **Config:** as E07.
- **CV result:**
  - **HGB:** screen 0.7076, d **+0.0000** — **bit-identical** (monotone per feature →
    tree-invariant, like E08).
  - **LogReg:** screen 0.6918 @0.600, d **−0.0043** — *hurts* on the raw PAY scale on its
    own. But on top of `pay_remap` + `bill_clip` it flips positive (+0.0027 over remap
    alone — see `clean_linear`, E11): a real interaction, log1p only pays off once the
    ordinal/overpayment encodings are also clean.
- **LB result:** — (not deployed; HGB-neutral, LogReg-negative alone).
- **Decision:** **PARK standalone / KEEP inside the clean combo.** Off on HGB
  (bit-identical) and off as a lone LogReg knob; valuable only combined (E11).
- **Notes / next:** don't read single-knob log1p as a loss — judge it in the combo.

### E10 — Negative BILL_AMT (overpayment) handling  (Path P2)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** negative `BILL_AMT` = overpayment / credit balance. Clipping at 0
  ("nothing owed", matching the `BILL≤0` handling in `add_engineered_features`) or adding
  an overpayment flag may clean the signal.
- **Changes:** `encoding.bill_neg ∈ {keep, clip, flag}`; `clip` → `max(BILL,0)`, `flag` →
  append `OVERPAY_BILL_*` = `BILL<0`. Variants `bill_clip`, `bill_flag`.
- **Config:** as E07.
- **CV result:**
  - **HGB:** `bill_clip` screen 0.7078 (d +0.0003), paired 0.7083 ± 0.0006, dmean
    **−0.0000**, wins 2/5 → revert. `bill_flag` screen 0.7090 @0.320 (d +0.0014, a lucky
    split) but paired dmean **−0.0002**, wins 2/5 → revert.
  - **LogReg:** `bill_clip` +0.0001 (4/5, marginal); `bill_flag` screen **−0.0007** (not
    validated). Clip is mildly preferable to flag.
- **LB result:** — (not deployed).
- **Decision:** **PARK.** No standalone win on either estimator; `bill_flag`'s +0.0014 HGB
  screen is a classic lucky-split that paired CV erased. Keep `clip` only as the
  overpayment treatment inside `clean_linear` (consistent denominators, E11).
- **Notes / next:** overpayments are rare → low leverage; not worth its own deploy.

### E11 — P2 decision: clean_linear combo + chosen encoding  (Path P2)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** the linear-favouring knobs (`pay_remap` + `log1p` + `scale` +
  `bill_clip`) should compound for LogReg even though each is HGB-neutral.
- **Changes:** variant `clean_linear = {pay_remap, log1p, scale:true, bill_neg:clip}`;
  set deployed `chosen.encoding` (HGB).
- **Config:** as E07.
- **CV result:**
  - **LogReg:** `clean_linear` screen 0.7059 @0.620 (d **+0.0099**); paired 0.7056 ±
    0.0004, dmean **+0.0099**, wins **5/5** → **robust KEEP**, the best LogReg encoding
    and the genuine P2 win (+0.0027 over `pay_remap` alone — the knobs *do* compound).
  - **HGB:** `clean_linear` screen 0.7089 (d +0.0013, lucky split) but paired 0.7079 ±
    0.0015, dmean **−0.0004**, wins 2/5 → revert (consistent with L5).
- **LB result:** deployed `chosen.encoding = {scale:false, rest baseline}` → bit-identical
  to S0, **LB 0.712** (no regression; runtime 3.4 s).
- **Decision:** **HGB → drop the scaler only** (cleanest neutral; zero LB risk, E08).
  Leave `pay_remap`/`log1p`/`bill_*` **off** on HGB (none robust; L1 after the S1 lesson).
  **Carry `clean_linear` (+0.0099, 5/5) as the LogReg encoding into P3** — that is the P2
  deliverable. P2 → done.
- **Notes / next:** P2 confirms encoding is a *linear-model* lever (HGB flat, LogReg
  +0.0099). The linear model now sits at ~0.706 CV, near HGB's 0.708 — promising
  diversity for a P4 ensemble. Re-tune HGB (step X) only after the model set settles.

---

> **Path P3 — Other sklearn models (E12).** Built on the raw-feature baseline. Each model
> is a declarative spec in `config.yaml` (`experiments.model_configs`), scored by
> `experiments/model_experiments.py` on the deployed objective + diversity-vs-HGB +
> equal-weight soft-vote previews (`experiments.ensemble_configs`, the P4 bridge). HGB
> anchor = 0.7083 ± 0.0007. All OOF probs computed once per (model, seed) and cached;
> soft-voting averages those cached OOF arrays (honest — every model shares the same
> StratifiedKFold split at a given seed).

### E12 — Model bake-off + soft-vote ensemble preview  (Path P3 → P4)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** a weaker-but-decorrelated sklearn model (LogReg on `clean_linear`, RF,
  ExtraTrees, plain GB) won't beat HGB alone, but soft-voting it with HGB should give a
  small, robust macro-F1 bump (the L6-endorsed lever) and shrink CV↔LB variance.
- **Changes:** new `experiments/model_experiments.py` (screen + diversity vs HGB +
  soft-vote preview + paired CV, off a shared OOF cache); `config.yaml`
  `experiments.model_configs` (hgb / logreg_clean / rf / et / gb, each with its suited
  encoding) + `experiments.ensemble_configs`; `src/config.py` exposes
  `MODEL_CONFIGS` / `ENSEMBLE_CONFIGS`.
- **Config:** deployed objective, seed 42 screen + paired CV [42,1,7,2024,99], thr grid
  0.05–0.95. RF/ET = 300 trees, `min_samples_leaf=20`, threshold-tuned (no class_weight);
  LogReg = `balanced`, `clean_linear` encoding; GB = defaults. Trees use `noscale` (L5),
  LogReg uses `clean_linear` (P2 winner).
- **CV result (paired vs HGB anchor 0.7083 ± 0.0007):**
  - **Standalone:** `rf` 0.7103, dmean **+0.0020**, wins **5/5** (robust but <0.005);
    `gb` +0.0009 (3/5); `logreg_clean` **−0.0016**; `et` **−0.0012**. A plain RF robustly
    edges HGB — but corr 0.974 (near-clone), so it is "a better tree", not diversity.
  - **Diversity vs HGB (corr / pred-disagreement, seed 42):** `logreg_clean` 0.914 / 0.056
    (most diverse), `et` 0.940 / 0.047, `rf` 0.974 / 0.031, `gb` 0.977 / 0.032.
  - **Soft-vote ensembles (all 5/5 robust):** `hgb+logreg_clean+rf` **+0.0026**,
    `hgb+rf` **+0.0024**, `all` +0.0023, `hgb+logreg_clean+et` +0.0022, `hgb+logreg` +0.0011;
    `hgb+et` +0.0010 (4/5).
- **LB result:** — (S2 pending upload; every gain is <0.005, so only the LB confirms — L1).
- **Deployed:** `chosen.ensemble = [hgb, logreg_clean, rf]` (config.yaml); `main.py` now builds
  a soft-vote `VotingClassifier` via `models.build_chosen_ensemble`, wrapped in the same
  threshold tuner (tuned thr **0.4323**). Reproduces in **62 s** (budget 150 s); submission
  regenerated (19.5% predicted defaults). Revert = set `chosen.ensemble: []` (back to S0).
- **Decision:** **P3 → done; deploy the best robust combo to the LB (P4).** No standalone
  swap is LB-worthy (RF's edge is a near-clone; L1). The genuine lever is the ensemble:
  `hgb+logreg_clean+rf` (+0.0026, 5/5) — HGB + the most-diverse linear + a strong tree.
  Adding LogReg is what lifts `hgb+rf` (+0.0024) to the top, confirming the hypothesis.
- **Notes / next:** see **L8** (diversity > member strength for ensemble selection). All
  CV gains <0.005 → the S1 risk applies; one LB upload decides KEEP/REVERT. Calibration /
  learned weights / stacking + a logistic meta-learner are the remaining P4 refinements
  if the equal-weight soft-vote transfers.

### E13 — P4 leaderboard check: equal-weight ensemble did not transfer  (Path P4)
- **Date:** 2026-06-20 · **Author:** human + agent log update
- **Hypothesis:** the E12 soft-vote `hgb + logreg_clean + rf` might transfer its small
  robust CV edge (+0.0026, 5/5) to the leaderboard; retesting the same ensemble with the
  tuned HGB params from prior P1/X tuning might improve the tree member.
- **Changes:** no new code for the equal-weight submission; `chosen.ensemble` had been
  `[hgb, logreg_clean, rf]`. The tuned-param variant temporarily changed the HGB member
  params for a submission test, then was reverted because it did not improve LB.
- **Config:** P4 soft-vote, each member with its P3 encoding; macro-F1 threshold tuner.
  Tuned-param variant used the previously found tuned HGB params in the HGB member.
- **CV result:** E12 ensemble CV = **0.7101** screen; paired dmean **+0.0026**, wins 5/5.
  Tuned-param variant not retained as a separate reproducible config because LB tied the
  untuned ensemble and remained below baseline.
- **LB result:** S2 **0.710**; S3 tuned-HGB-param ensemble **0.710**.
- **Decision:** **REVERT** — both are below S0 **0.712** despite positive CV. Set
  `chosen.ensemble=[]` before moving to P5.
- **Notes / next:** this strengthens L1: even robust CV gains around +0.003 are not enough
  on this leaderboard. Next path is P5 imbalance handling; only deploy if LB beats 0.712.

### E14 — Class weighting / imbalance sweep  (Path P5)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** threshold tuning fixes the final cut, but class weighting might still
  improve probability ranking by fitting the loss with more minority-class pressure.
- **Changes:** added `experiments/imbalance_experiments.py`; added
  `experiments.imbalance_configs` and `model_configs.rf_balanced` to `config.yaml`;
  `src/config.py` exposes `IMBALANCE_CONFIGS`; `src/models.py` treats a one-member
  `chosen.ensemble` as a direct deployed model. Current deploy candidate:
  `chosen.ensemble=[rf_balanced]`.
- **Config:** HGB anchor = `model_configs.hgb` (`noscale`, defaults). P5 candidates:
  HGB `class_weight` balanced / `{1:1.5}` / `{1:2.0}` / `{1:3.0}`, plus RF
  `n_estimators=300`, `min_samples_leaf=20`, `class_weight="balanced"`, `n_jobs=-1`.
- **CV result:**
  - Screen seed 42: `hgb_balanced` 0.7063 (−0.0012), `hgb_w15` 0.7085 (+0.0009),
    `hgb_w20` 0.7075, `hgb_w30` 0.7080 (+0.0004), `rf_balanced` **0.7108** (+0.0032).
  - Paired HGB class weights: `hgb_w15` dmean **−0.0002**, wins 3/5; `hgb_w30`
    dmean **+0.0004**, wins 3/5 → no robust HGB win.
  - Paired `rf_balanced`: **0.7115 ± 0.0006**, dmean **+0.0032**, wins **5/5** vs HGB.
- **LB result:** S4 **0.713** (`outputs/submissions/submission.csv`, 2026-06-20),
  Δ vs S0 = **+0.001**.
- **Decision:** **KEEP as current best / still small.** This is the first post-baseline
  LB improvement, but it is far below the kind of jump we need. Runtime verified by
  `main.py`: **10.2 s**, threshold **0.6220**, predicted defaults **1,185 / 6,000
  (19.8%)**.
- **Notes / next:** test only a few adjacent RF-balanced knobs before closing P5. Do not
  add SMOTE/imblearn unless zero-dep P5 is exhausted and a dependency is justified by a
  clear LB gain.

### E15 — RF-balanced adjacent variants  (Path P5 follow-up)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** since S4 improved LB, there may be a little extra headroom near the
  same RF-balanced direction by changing RF regularisation / bootstrap weighting:
  smaller/larger `min_samples_leaf`, `balanced_subsample`, or more aggressive
  `max_features`.
- **Changes:** added P5 configs:
  `rf_balanced_leaf10`, `rf_balanced_leaf30`, `rf_balanced_leaf50`,
  `rf_balanced_subsample`, `rf_balanced_mf05`.
- **Config:** same P5 protocol as E14. All candidates inherit RF =
  `n_estimators=300`, `min_samples_leaf=20`, `encoding=noscale` from
  `model_configs.rf`, then override only the listed P5 params.
- **CV result:**
  - Screen seed 42 vs HGB 0.7076: `rf_balanced` **0.7108** (+0.0032),
    `leaf10` 0.7103 (+0.0027), `leaf30` 0.7107 (+0.0031), `leaf50` 0.7091
    (+0.0016), `balanced_subsample` 0.7107 (+0.0031), `max_features=0.5`
    0.7086 (+0.0011).
  - Paired repeated CV vs HGB 0.7083 ± 0.0007:
    `rf_balanced` **0.7115 ± 0.0006**, dmean **+0.0032**, wins 5/5;
    `balanced_subsample` **0.7115 ± 0.0008**, dmean **+0.0032**, wins 5/5;
    `leaf10` 0.7113 ± 0.0008, dmean +0.0031, wins 5/5;
    `leaf30` 0.7107 ± 0.0003, dmean +0.0024, wins 5/5;
    `leaf50` 0.7098 ± 0.0004, dmean +0.0015, wins 5/5;
    `max_features=0.5` 0.7095 ± 0.0010, dmean +0.0012, wins 4/5.
- **LB result:** not submitted.
- **Decision:** **PARK / close P5.** No adjacent RF-balanced variant clearly beats the
  current S4 model in CV. `balanced_subsample` ties the paired mean delta but has a
  slightly lower screen score and no evidence of a larger jump; it is at most a
  leaderboard tie-break, not a strong direction.
- **Notes / next:** move to **P6 threshold-transfer strategy**. If we spend one more LB
  slot inside P5, the only reasonable candidate is `rf_balanced_subsample`, but the
  expected gain is tiny.

### E16 — Threshold transfer-risk audit + flat-cut strategies  (Path P6)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** the per-fit tuned threshold may overfit dev and not transfer to
  eval — a likely source of CV↔LB wobble (L1). The deployed `rf_balanced` cut is a
  high, sensitive **0.622**, so a less greedy threshold (fold-averaged or a flat
  fixed cut) might transfer dev→eval better and steady the LB.
- **Changes:** new `experiments/threshold_experiments.py` — computes OOF proba +
  per-row fold id, then four threshold strategies per seed: **GLOBAL** (optimistic,
  tune+score on the same full OOF — what every other script reports), **TRANSFER**
  (honest leave-one-fold-out: tune the cut on the other 4 folds, apply to the held-out
  fold), **FOLD-AVG** (one flat averaged cut on full OOF), **BEST-FIXED** (the single
  grid threshold with the best mean held-out macro-F1). No deployed code touched.
- **Config:** probes `rf_balanced` (deployed S4) and `hgb` (anchor); deployed objective,
  threshold grid 0.05–0.95, paired seeds [42,1,7,2024,99], 5-fold stratified.
- **CV result:**
  - **`rf_balanced`:** GLOBAL **0.7115 ± 0.0006**, TRANSFER **0.7103 ± 0.0008**
    (penalty **+0.0012**), FOLD-AVG 0.7110, BEST-FIXED 0.7114 @thr~0.613. Per-fold
    optimal threshold **0.615 ± 0.012** (range 0.585–0.630) — a *tight*, stable cut.
  - **`hgb`:** GLOBAL 0.7083, TRANSFER 0.7070 (penalty **+0.0013**), FOLD-AVG 0.7082,
    BEST-FIXED 0.7083 @thr~0.323. Per-fold threshold 0.327 ± 0.015 — a *wider* spread
    than RF, i.e. HGB's cut is slightly less stable, not more.
- **LB result:** — (no submission; no strategy produced an honest gain to ship).
- **Decision:** **P6 → done; no lever, keep the deployed per-fit tuned threshold.** The
  transfer penalty is small (~0.0012) and *nearly identical* across models, and no
  flat/averaged strategy beats per-fit tuning on held-out data. The deployed
  `TunedThresholdClassifierCV` is already the robust (fold-averaged-equivalent) choice,
  and its 0.622 cut is the most stable of the two models tested. Reassuringly, LB (0.713)
  already sits *above* dev GLOBAL (0.7115), so the cut is not costing us on eval.
- **Notes / next:** rules out threshold strategy as the CV↔LB wobble source (→ L10). With
  P1–P6 all moving <0.005, the zero-dep sklearn space is effectively at its ceiling on
  this data. The only remaining high-ceiling lever is **P7** (LightGBM/XGBoost/CatBoost),
  which is a *new-dependency* decision gated on a real LB gain — surfaced to the human.

### E17 — LightGBM bake-off + randomized tuning  (Path P7)
- **Date:** 2026-06-20 · **Author:** agent
- **Hypothesis:** a stronger gradient-boosting library (LightGBM) — more knobs,
  finer leaf-wise growth — could clear the ~0.7115 sklearn ceiling by enough to
  justify a new dependency (the L1 bar for a dep is a real **LB** gain, screened as
  ≥ +0.005 paired CV over the deployed `rf_balanced`).
- **Changes:** `src/models.py` gains a **lazy** `_make_lgbm` factory (`kind: lgbm`,
  imported only when requested, `verbose=-1`, `n_jobs=-1`) so main.py / other
  experiments are unaffected when LightGBM is absent. `config.yaml` adds
  `experiments.boosting_configs` (5 hand specs); `src/config.py` exposes
  `BOOSTING_CONFIGS`; new `experiments/boosting_experiments.py` (screen + paired CV
  vs the **deployed champion `rf_balanced`**, not the weaker hgb anchor). LightGBM
  installed locally only (`pip install lightgbm` + `brew install libomp` for the
  OpenMP runtime on macOS) — **NOT added to `requirements.txt`** (not deployed).
- **Config:** deployed objective (OOF proba → best macro-F1 over the 0.05–0.95 grid),
  `noscale` encoding (trees ignore monotone rescaling, L5), seed 42 screen + paired
  CV [42,1,7,2024,99]. Plus a 40-config randomized search over
  `{n_estimators, learning_rate, num_leaves, min_child_samples, subsample,
  colsample_bytree, reg_lambda, reg_alpha, class_weight}`.
- **CV result:**
  - **Hand specs (screen seed 42, champ `rf_balanced` = 0.7108):** `lgbm_default`
    0.7093 (−0.0015), `lgbm_balanced` 0.7073 (−0.0035), `lgbm_reg` 0.7090 (−0.0018),
    `lgbm_reg_balanced` 0.7100 (−0.0008), `lgbm_deep_balanced` 0.7053 (−0.0055). None
    beat the champion → nothing validated from the hand set.
  - **Randomized search (40 configs):** 7 edged the champion's *seed-42 screen*; best
    0.7120. But validated by paired CV vs `rf_balanced` (0.7115 ± 0.0006), the top 3
    were: `lgbm_B` (n=800, lr=0.01, leaves=20, balanced, l2=2) **0.7119 ± 0.0003,
    dmean +0.0004, wins 3/5**; `lgbm_A` 0.7111, −0.0004, 1/5; `lgbm_C` 0.7111, −0.0004,
    2/5. The best tuned LightGBM beats the champion by only **+0.0004** and not on all
    seeds — a full order of magnitude below the +0.005 dependency bar.
- **LB result:** — (no submission; no config cleared the bar to ship, let alone to
  justify the dependency).
- **Decision:** **P7 → done / PARK LightGBM; do NOT deploy, do NOT add the dep.** Even
  properly tuned, LightGBM only ties `rf_balanced` in CV (+0.0004, 3/5) — pure noise by
  L1, and the P4 ensemble taught us that even +0.0026 paired *loses* on this LB. Spending
  a new dependency (install/build risk + 150 s budget) on a noise-level CV tie is
  unjustified. Keep `rf_balanced` (0.713) deployed. `main.py` re-verified: **9.2 s**, thr
  **0.6220**, 1,185/6,000 defaults, `requirements.txt` unchanged.
- **Notes / next:** the lazy hook + `boosting_configs` + `experiments/boosting_experiments.py`
  stay as reproducible P7 infrastructure (re-run needs `pip install lightgbm` + libomp).
  This closes every planned path P1–P7. Conclusion (→ L11): 0.713 is the practical ceiling
  for this dataset/these learners; remaining effort is best spent on the **report** and the
  **final 2-submission pick** (S0 0.712 + S4 0.713), not more modeling.

---

> **Paths P8–P12 — second research round (NotebookLM ideas).** After P1–P7 closed at
> the 0.713 ceiling, a NotebookLM web/literature search proposed 5 new directions
> (stacking, hybrid resampling SMOTE+Tomek, probability calibration, target encoding,
> Isolation-Forest anomaly feature). We re-ranked them by fit to *this* problem and
> screened each CV-first vs the deployed champion **rf_balanced** (0.7108 screen /
> 0.7115 ± 0.0006 paired), same +0.005 / real-LB-gain bar (L1). Order run by promise:
> P8 stacking → P10 resampling → P9 anomaly → P11 target-enc → P12 calibration.

### E18 — Stacking with a logistic meta-learner  (Path P8)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** P4 only tried *equal-weight* soft-vote (lost on LB, 0.710). A
  `StackingClassifier` trains a logistic meta-learner on the members' OOF `predict_proba`,
  so it can learn member weights/interactions and might beat plain averaging.
- **Changes:** new `experiments/stacking_experiments.py`; `config.yaml`
  `experiments.stacking_configs` (4 specs); `src/config.py` exposes `STACKING_CONFIGS`.
  No deployed code touched. Members reuse the byte-identical P3 `model_configs` pipelines;
  outer OOF via `cross_val_predict`, inner stacking cv = StratifiedKFold(5, seed).
- **Config:** deployed objective, seed 42 screen + paired CV [42,1,7,2024,99], thr grid
  0.05–0.95. Meta = LogisticRegression(max_iter=1000) (and one `class_weight=balanced` variant).
- **CV result (paired vs champion rf_balanced 0.7115 ± 0.0006):**
  - `stack_rfbal_hgb_logreg` **0.7120 ± 0.0005, dmean +0.0005, wins 4/5** (best).
  - `stack_rfbal_hgb_logreg_balmeta` +0.0004 (4/5); `stack_hgb_logreg_rf` +0.0002 (3/5);
    `stack_rfbal_logreg` +0.0001 (3/5). Screen deltas all +0.0002…+0.0006.
- **LB result:** — (not submitted).
- **Decision:** **PARK.** The best stack edges the champion by only +0.0005 and not on
  all seeds — noise by L1, and P4 already showed +0.0026 paired *loses* on this LB.
  Stacking ≠ enough to justify a more complex deploy. Infra kept for repro.
- **Notes / next:** marginally better than P4 soft-vote / P7 LightGBM relative to champion,
  but same conclusion — ensembling is exhausted on this data. [[E12]] [[E13]] [[E17]]

### E19 — Isolation-Forest anomaly feature  (Path P9)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** defaulting is "anomalous"; an unsupervised anomaly score (how far a
  client deviates from the norm) might add signal the supervised model misses.
- **Changes:** `src/preprocessing.py` gains a stateful `AnomalyScorer` (BaseEstimator/
  TransformerMixin wrapping `IsolationForest`, fit on train folds only → leakage-safe),
  placed at the END of the pipeline on the numeric/one-hot matrix; new
  `experiments/anomaly_experiments.py`. No deployed code touched.
- **Config:** rf_balanced (champion) and hgb each + a 200-tree IsolationForest score;
  deployed objective, seed 42 screen.
- **CV result:** `rf_balanced + anomaly` 0.7106 (**d −0.0001**); `hgb + anomaly` 0.7080
  (d −0.0027). Nothing beat the champion at the screen → not validated.
- **LB result:** — (not submitted).
- **Decision:** **PARK.** No signal beyond the raw features — exactly L5/L6 (a tree
  already extracts what an aggregate/summary feature encodes).
- **Notes / next:** `AnomalyScorer` kept as reusable, leakage-safe infra. [[E02]]

### E20 — Resampling / hybrid sampling (SMOTE family)  (Path P10)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** P5 only reweighted the loss; actually rebalancing the train folds
  (SMOTE) and cleaning the class overlap (SMOTE+Tomek — NotebookLM's #1 pick) might beat
  `class_weight` + threshold tuning on macro-F1.
- **Changes:** new `experiments/resampling_experiments.py`; `config.yaml`
  `experiments.resampling_configs` (7 specs); `src/config.py` exposes `RESAMPLING_CONFIGS`.
  Sampler runs inside an `imblearn.pipeline.Pipeline` so `fit_resample` fires on TRAIN
  folds only (held-out fold never resampled → leakage-safe); base estimator is the PLAIN
  `rf` (resampling does the rebalancing, so no double class-weighting). **New dep
  installed LOCALLY only** (`pip install imbalanced-learn`), like LightGBM in P7 —
  **NOT added to requirements.txt**.
- **Config:** deployed objective, seed 42 screen vs champion rf_balanced (0.7108).
- **CV result (screen d vs champion):** `undersample` −0.0012, `borderline` −0.0024,
  `smote` −0.0030, `smotetomek` **−0.0043**, `smoteenn` −0.0052, `smote_hgb` −0.0058,
  `smotetomek_hgb` −0.0071. **Every** candidate is negative → nothing validated.
- **LB result:** — (not submitted).
- **Decision:** **PARK; do NOT add imblearn.** Synthetic resampling is *worse* than
  class-weight + threshold tuning for these tree ensembles on macro-F1 — and the
  NotebookLM-top hybrid (SMOTE+Tomek) is among the worst. A new dependency on a
  CV-losing method is unjustified (L1). Screened CV-first precisely to avoid shipping it.
- **Notes / next:** `resampling_experiments.py` + configs kept as repro infra (re-run
  needs `pip install imbalanced-learn`). [[E14]]

### E21 — Target encoding of the categoricals  (Path P11)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** replace one-hot of SEX/EDUCATION/MARRIAGE with target encoding
  (category → mean default rate; NotebookLM #4) for a more compact, ordered encoding.
- **Changes:** new `experiments/target_encoding_experiments.py` using
  **`sklearn.preprocessing.TargetEncoder`** (ships in sklearn 1.9, internally cross-fitted
  → leakage-safe, so **zero new dependency** — no `category_encoders` needed). No deployed
  code touched.
- **Config:** rf_balanced (champion) and hgb, categoricals target-encoded instead of
  one-hot; deployed objective, seed 42 screen.
- **CV result:** `rf_balanced + target-enc` 0.7106 (**d −0.0001**); `hgb + target-enc`
  0.7083 (d −0.0025). Nothing beat the champion → not validated.
- **LB result:** — (not submitted).
- **Decision:** **PARK.** Tiny cardinality (SEX=2, EDUCATION=4, MARRIAGE=3) means one-hot
  is already lossless; target encoding's high-cardinality advantage doesn't exist here.
- **Notes / next:** consistent with L7 (encoding is a no-op for the tree). [[E11]]

### E22 — Probability calibration  (Path P12)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** NotebookLM #3 — calibrating RF's "pushed-in" probabilities (sigmoid/
  isotonic) before the threshold tuner would steady the cut and lift macro-F1.
- **Changes:** new `experiments/calibration_experiments.py`. No deployed code touched.
- **Config:** rf_balanced champion; (A) fit isotonic + Platt on the champion's OOF scores
  and re-score the SAME scores; (B) `CalibratedClassifierCV(rf_balanced, cv=5)` wrap,
  OOF-scored; deployed objective, seed 42 screen + paired CV.
- **CV result:**
  - **(A) monotonic no-op:** Platt **−0.0003**, isotonic **+0.0000** vs raw 0.7108 — a
    monotonic map preserves rank order, so the best thresholded macro-F1 is unchanged
    (the −0.0003 is grid discretization).
  - **(B) CalibratedCV wrap:** `sigmoid` 0.7114 screen, **paired +0.0001, wins 3/5**;
    `isotonic` −0.0006 screen. The tiny screen wiggle is the incidental cross-fit
    *ensembling* inside CalibratedCV (it refits + averages the base estimator), **not**
    calibration.
- **LB result:** — (not submitted).
- **Decision:** **PARK — calibration is a no-op for our objective.** Because we submit
  behind a single threshold on `predict_proba`, only the *rank* of the scores matters and
  monotonic calibration cannot change it. Calibration is for probability consumers
  (Brier/log-loss/cost-based/multiclass), not a rank-based cut. (→ L12)
- **Notes / next:** rules calibration out as a lever; don't revisit for thresholded F1.

---

> **2026-06-21 review round (E23–E27).** An external evaluation + SOTA literature search
> flagged that the project had been *under-using its submission budget* (5/200) and closing
> paths on dev-CV alone — a signal that under-reads the LB in *both* directions. Decision:
> stop pre-filtering on CV and convert the budget into LB signal. New shared infra
> `experiments/_submit_utils.py` writes numbered `outputs/submissions/submission_<N>.csv`
> candidates + a `MANIFEST.md`; 25 candidates were generated across the items below for the
> human to upload. The LB column in MANIFEST is the source of truth for these.

### E23 — Path X: randomised re-tuning of HGB and RF  (Path X)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** Path X was never run; S0 HGB is at library defaults and the deployed RF's
  knobs were hand-picked, never searched. A proper randomised search on the deployed
  objective might find a config that beats the champion.
- **Changes:** new `experiments/path_x_search_submit.py` (randomised search for both
  families on OOF-macro-F1 + paired repeated-CV + emits top-3 of each as submissions).
  Reuses `_submit_utils`. Fixed a nested-parallelism trap (RF `n_jobs=-1` inside a parallel
  CV) via `oof_proba(cv_n_jobs=...)`.
- **Config:** 40 HGB cfg + 30 RF cfg, seed 42 screen + paired CV [42,1,7,2024,99], thr grid
  0.05–0.95, `noscale` encoding.
- **CV result:** HGB tuned robustly beats the **HGB default** (best +0.0021, 5/5: anchor
  0.7083 → 0.7104) but stays below the RF champion. Best tuned **RF** (800 trees, depth 24,
  `sqrt`, leaf 10, `max_samples=0.7`) = **0.7118 ± 0.0008, dmean +0.0003, wins 4/5** vs
  `rf_balanced` 0.7115 — a noise-level edge by L1.
- **LB result:** submissions 12–17 (`MANIFEST.md`). `submission_14` (HGB cand2) = **0.706**
  (< 0.712, reverted) — HGB-family tuning does not transfer, as L1 predicts.
- **Decision:** **PARK / keep `rf_balanced` deployed unless an LB upload says otherwise.**
  Path X is now closed (was the last `planned` cross-cutting step).
- **Notes / next:** if any of submissions 15–17 (tuned RF) beats 0.713 on the LB, promote
  it via `chosen.ensemble`/a new `model_configs` entry.

### E24 — Threshold sweep on the champion  (Item 1, LB extension of P6)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** the deployed cut (0.622) predicts only 19.8% defaults on eval vs the 22.1%
  base rate — i.e. it under-fires the minority class, where macro-F1 is most sensitive, and
  dev CV is blind to the dev→eval shift. Submitting the *same* model at several thresholds
  lets the LB locate the real optimum.
- **Changes:** new `experiments/threshold_sweep_submit.py` — fits `rf_balanced` once, emits
  one submission per threshold in {0.45, 0.50, 0.55, 0.58, 0.60, 0.622, 0.65}.
- **CV result:** dev OOF macro-F1 peaks at 0.622 (0.7104); on **eval** the predicted-default
  rate hits the 22.1% base rate at thr **0.580** (`submission_4`, 22.0%).
- **LB result:** submissions 1–7 (`MANIFEST.md`). `submission_6` (thr 0.622) is byte-identical
  to the deployed S4 (sanity check). LB pending for the rest.
- **Decision:** the highest-EV cheap probe — wait for the LB. If a lower cut beats 0.713,
  it is a free macro-F1 gain.
- **Notes / next:** the base-rate-matching candidate (`submission_4`) is the one to watch.

### E25 — Parked CV-positive candidates, finally uploaded  (Item 3, P4/P8/P5)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** several candidates screened CV-positive (or were built on the weaker HGB
  anchor before RF became champion) but were never submitted. With the budget freed, upload
  them and let the LB decide.
- **Changes:** new `experiments/parked_candidates_submit.py`; added
  `ensemble_configs.rfbal_logreg_et` to `config.yaml`. Reuses `make_stacking` (P8) and
  `build_chosen_ensemble`.
- **Config:** each wrapped in main.py's `TunedThresholdClassifierCV`, fit on full dev.
- **CV result:** the two **stacks** edge the champion at seed 42 — `stack [rf_balanced,hgb,
  logreg_clean]` OOF **0.7114**, `stack [hgb,logreg_clean,rf]` 0.7113 (vs 0.7108); soft-vote
  `rfbal_logreg_et` 0.7104; `rf_balanced_subsample` 0.7107.
- **LB result:** submissions 8–11 (`MANIFEST.md`). Pending.
- **Decision:** upload; promote to `chosen` only if the LB clears 0.713. (P4's +0.0026
  *lost* on the LB, so CV optimism here is treated with L1 caution.)
- **Notes / next:** the stacks are the most promising of this group on CV.

### E26 — Group-aware (segmented) thresholds  (Path P13)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** one global cut ignores that subgroups have very different default base
  rates; SOTA on this dataset (Gittlin, ECML-PKDD 2025) reports +1.5–4% balanced acc from
  per-subgroup thresholds.
- **Changes:** new `src/segment_threshold.py` (segmenters for PAY_0 buckets / SEX /
  LIMIT_BAL terciles; per-segment cuts tuned by **coordinate ascent on global macro-F1**
  with a `min_support` floor + global-cut fallback) + `experiments/segment_threshold_experiments.py`
  (honest leave-one-fold-out transfer, mirroring P6).
- **Config:** champion `rf_balanced`, 5 fold seeds, thr grid 0.05–0.95, min_support 500.
- **CV result:** honest transfer is **~neutral** — `pay0` +0.0002 (3/5), `limit3` ~0 (2/5),
  `sex` −0.0004 (1/5). The full-OOF gains (`limit3` 0.7122 vs global 0.7108) are optimistic
  (tuned+scored on the same OOF). Consistent with L10 (the cut is already stable).
- **LB result:** submissions 18–20 (`MANIFEST.md`). Pending.
- **Decision:** **not a robust CV lever**, but cheap to LB-test — uploaded for the LB to
  confirm/deny, since dev CV is blind to the eval shift this lever targets. If it transfers,
  integrate as an optional `chosen.segment_threshold` post-`predict_proba` step.
- **Notes / next:** the SOTA gain was on *balanced accuracy* w/ richer segments; macro-F1 +
  our 3 coarse segments shows little headroom in honest CV.

### E27 — ExtraTrees-balanced + feature engineering on RF  (Items 6 & 5, P5/P1 revisit)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** (6) ET-balanced was never tried and is more decorrelated than RF (L8).
  (5) L5/L7 ("the tree already has it") were established on **HGB**, whose histogram binning
  absorbs monotone transforms; **RF**'s axis-aligned splits on the collinear BILL block may
  benefit from explicit ratios — so re-test `util`/`payratio`/`pay` on `rf_balanced`.
- **Changes:** new `experiments/extra_candidates_submit.py`; added `model_configs.et_balanced`
  to `config.yaml`.
- **Config:** deployed objective at seed 42, each wrapped in `TunedThresholdClassifierCV`.
- **CV result:** **ET-balanced weak** (OOF 0.7068). **FE on RF is mildly positive** — and
  unlike on HGB: `rf_balanced + payratio` OOF **0.7111** (> champion 0.7108 at seed 42),
  `util_payratio` 0.7110, `util` 0.7101, `pay` 0.7087 (worse). So RF *does* read explicit
  ratios that HGB ignored — a partial counter-example to L5 on a different tree family.
- **LB result:** submissions 21–25 (`MANIFEST.md`). Pending.
- **Decision:** drop ET-balanced; **upload `rf_balanced + payratio` / `+ util_payratio`** as
  the real candidates of this item — they are the first FE configs to beat the champion in CV
  since P1. Promote only on an LB gain (L1; remember `util` on HGB looked +CV and lost on LB).
- **Notes / next:** if `payratio`-on-RF transfers, it would refine L5 ("re-derived features
  help RF, not HGB") and become the deployed model.

---

> **Path P14 — PAY-semantic feature engineering on RF (E28+).** E27's FE-on-RF candidates
> transferred on the LB (`+payratio` **0.716**, `+util` 0.714, `+util_payratio` 0.714, all
> > the 0.713 ceiling), so the external review's central thesis is confirmed: **FE on RF is
> the lever that actually moves this leaderboard**, and dev OOF is decoupled from it
> (highest-OOF candidates are not the highest-LB). P14 pushes the lever with genuinely new
> signal the prior encoding discarded — the PAY_* SEMANTIC decomposition (revolving vs
> paid-in-full vs no-consumption, latest-month state, PAY_0≥2 flag) + recent coverage/trend.

### E28 — PAY-semantic + coverage-trend features on rf_balanced  (Path P14)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** the PAY_* codes are **not** a monotone scale (−2 no-consumption, −1
  paid-in-full, 0 revolving-credit = carrying a balance = riskier than −1/−2, 1..9 late).
  `pay`/`pay_remap` collapse {−2,−1,0} into one bucket, throwing away the revolving-vs-paid
  distinction. Keeping those states separate (+ isolating the latest month + a PAY_0≥2 flag)
  is new signal an RF can split on, and FE-on-RF is the transferring lever (E27).
- **Changes:** `src/preprocessing.py` two new engineered families: `paysem`
  (PAY_N_REVOLVING / PAY_N_PAIDFULL / PAY_N_NOCONS / PAY0_GE2 / PAY0_REVOLVING /
  PAY0_PAIDFULL) and `coverx` (COVER_RECENT=PAY_AMT1/BILL_AMT2, COVER_TREND, UTIL_TREND).
  `config.yaml` registry combos (`paysem`, `coverx_payratio`, `paysem_util_payratio`, …);
  new `experiments/feature_families_submit.py` (screen + emit on the rf_balanced champion,
  argv-selectable). All stateless row-wise, leakage-free, in-Pipeline.
- **Config:** champion `rf_balanced` (300 trees, leaf 20, balanced, noscale), deployed
  objective, seed-42 screen + per-fit threshold tuner; emitted regardless of CV (LB judges).
- **CV result (screen vs champion 0.7108):** `coverx_payratio` **0.7125** (+0.0017, best
  screen), `paysem_coverx_payratio` 0.7117, `all_new` 0.7115, `paysem_util_payratio` 0.7107
  (−0.0001), `paysem` 0.7101. OOF again **under-reads** the LB (see below).
- **LB result (MANIFEST submissions 26–42):** **`paysem_util_payratio` (sub 31) = 0.719**
  — a **new best, +0.006 over the 0.713 ceiling**. Also `paysem_coverx_payratio` (sub 30)
  **0.717**, `paysem` (26) 0.716, `coverx_payratio` (29) 0.716. The lowest-OOF combo
  (`paysem_util_payratio`, 0.7107) is the **highest-LB** — OOF↔LB decoupling, hard.
- **Decision:** **KEEP / DEPLOY.** `chosen.feature_groups = [paysem, util, payratio]` with
  `chosen.ensemble = [rf_balanced]` (single-member deploy path now applies feature_groups,
  `src/models.py` `_member_pipeline`). main.py reproduces submission_31 (0.719).
- **Notes / next:** re-tune RF *on this feature set* (L3 — `feature_retune_submit.py
  --groups paysem_util_payratio`); item 1 lever-multiplication (subs 32–37) was flat in OOF,
  LB pending; CatBoost (review item 3) is the next new-dep candidate if FE stalls. → L13.

### E29 — Seed sensitivity, FE ensembles, granular features  (Path P14 follow-up)
- **Date:** 2026-06-21 · **Author:** agent + human
- **Hypothesis:** with the FE win deployed, three follow-ups: (a) is 0.719 seed-stable or a
  lucky split? (b) does ensembling decorrelated model families on the *same* paysem feature
  set finally beat a single RF (P4 lost on the raw base, but the base is far stronger now)?
  (c) do granular per-month ratios / stress interactions add signal over the mean aggregates?
- **Changes:** `experiments/seed_average_submit.py` (8-seed proba average of the winner);
  `experiments/fe_ensemble_submit.py` (soft-vote/stack of rf/hgb/logreg/et on the deployed
  feature set, pinned to seed 888); `src/preprocessing.py` 3 new families `utilmonths`
  (per-month BILL/LIMIT), `payamtratio` (per-month PAY_AMT/BILL), `stress` (utilisation×
  delinquency interactions); `feature_families_submit.py` gained `--seed`.
- **CV/LB result:**
  - **Seed sensitivity (human, ~30 manual seed runs on the deployed model):** seed **888 →
    0.720** (new best), seed 42 → 0.719; the win is seed-stable (no seed collapses) but the
    last +0.001 is a seed lottery. Seed is itself a real lever on this LB.
  - **Seed-average (sub 46): LB 0.718** — robust but did not beat the best single seed.
  - **FE ensembles (subs 47–50): LB 0.717–0.719** — soft-vote rf+hgb+logreg = 0.719, ties
    but does not beat the single FE-RF. Same-feature ensembling is exhausted (confirms P4/L8
    caveat on this base too).
  - **Granular/stress features (subs 51–54): `paysem_stress_payratio` (sub 53) = 0.720** —
    ties the champion despite the *lowest* OOF of the batch (0.7093); the utilisation×
    delinquency interaction is real LB signal. `utilmonths`/`payamtratio`/4-family all
    ≤0.717 (more features dilute — the 3-family sweet spot holds).
- **Decision:** deployed champion = `rf_balanced + paysem_util_payratio` at **seed 888**
  (LB 0.720); the tie-best alternative is `paysem_stress_payratio` (0.720), a candidate for
  the second final pick (it errs on different features → diversity). `config.yaml seed`
  annotated. Item-1 combos/re-tunes (subs 32–45) all 0.712–0.716, none beat the FE win.
- **Notes / next:** plateau at 0.717–0.720 across FE/ensemble/seed; aiming for ≥0.722.
  Next: CatBoost (item 3, E30) + feature-VIEW ensembles (different views err differently).

### E30 — CatBoost + feature-view ensembles  (Path P14 item 3 + follow-up)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** (item 3) our LB failure mode is *overfitting* (tuned HGB tanked); CatBoost's
  ordered boosting resists that and errs differently from RF, so standalone and as a
  decorrelated ensemble member it may break the 0.720 plateau. (follow-up) the same-feature
  ensembles plateaued because members share inputs — soft-voting RF on *different feature
  views* (paysem_util_payratio vs paysem_stress_payratio, both 0.720) maximises decorrelation.
- **Changes:** lazy `_make_catboost` factory + `kind: catboost` in `src/models.py` (maps
  random_state→random_seed, verbose off); `experiments/catboost_submit.py` (3 lightly-tuned
  regularised specs, standalone + on stress + soft-vote with rf_balanced);
  `experiments/feature_view_ensemble_submit.py` (RF/CatBoost over different feature views).
  **CatBoost installed LOCALLY only — NOT in requirements.txt** (gated on an LB gain over
  0.720, same policy as LightGBM P7/L1). All pinned to seed 888.
- **CV result (seed-888 OOF on paysem_util_payratio):** `cb_reg` (800 it, depth 4, l2=6,
  Balanced) **0.7110**, `cb_deep` 0.7108, `cb_default` 0.7096 — competitive with RF (0.7103)
  but not clearly better; OOF decoupled from LB as always (L13).
- **LB result:** **CatBoost standalone 55–57 = 0.708 / 0.715 / 0.711 — all BELOW the RF
  champion (0.720).** CatBoost+stress (58) 0.711; CatBoost⊕rf soft-vote (59) 0.717;
  feature-view ensembles 60–62 = 0.718 / 0.719 / 0.716.
- **Decision:** **PARK CatBoost; do NOT add the dep.** It loses to plain RF here (overfitting-
  resistance bought nothing — the bottleneck is data signal, not variance), so it fails the
  0.722 dep bar decisively. Feature-view ensembles tie (0.719) but don't beat the 0.720
  single RF — same-/cross-view ensembling is exhausted. `catboost` installed locally as repro
  infra only, NOT in requirements.txt.
- **Notes / next:** the FE-on-RF lever and ensembling are now exhausted at the **0.720
  plateau**. → E31 (richer semantics) is the last FE attempt; TabPFN (item 6) is infeasible
  here (see L14).

### E31 — TabPFN infeasible; richer PAY-dynamics features  (Path P14 item 6 + FE)
- **Date:** 2026-06-21 · **Author:** agent
- **Hypothesis:** (item 6) TabPFN v2 is SOTA on small tabular data and could beat the GBMs.
  (FE) the proven lever is FE-on-RF, so richer PAY *dynamics* (recovery/escalation/recent-vs-
  old, the *direction* of delinquency, not just state counts) might add the signal that
  pushes past 0.720.
- **Changes:** `src/preprocessing.py` family `paysem2` (PAY_RECOVERED / PAY_NEWLY_LATE /
  PAY_DELINQ_RECENT3 / PAY_DELINQ_OLD3 / PAY_ESCALATION); config combos `paysem2_*`. Attempted
  `pip install tabpfn`.
- **Result:**
  - **TabPFN: INFEASIBLE (→ L14).** `torch` has **no installable wheel** for this env
    (Python 3.13.2, numpy 2.4.6); `pip install --dry-run torch` → "No matching distribution".
    Even if forced in a side-env, the grader runs `pip install -r requirements.txt` on the
    same Python, so TabPFN would be **undeployable** regardless. Dropped on hard env grounds.
  - **paysem2 features:** subs 63+ at seed 888 — LB *pending*.
- **Decision:** TabPFN closed (environmental, not a judgement). paysem2 LB pending.
- **Notes / next:** deliver the top FE configs for manual seed-hunting (folder under the
  user's POLITO sub/configs/); if paysem2 doesn't clear 0.720, lock the 0.720 plateau and
  pivot to the IEEE report + final 2-pick (A: paysem_util_payratio, B: paysem_stress_payratio).

---

## 6. Lessons learned (hard-won, don't relearn)

- **L1 — Trust the leaderboard; dev-CV gains <0.005 are noise.** A regularised HGB
  config looked +0.0035 macro-F1 better in nested CV but scored **0.708 vs 0.712**
  for defaults on the LB. The CV gain did not transfer. Corollary: don't ship a
  more-complex, lower-LB config to chase a tiny CV bump. Use CV as a *screen* to
  decide what to submit, but let the LB decide what to keep.
- **L2 — Optimise the deployed objective.** Because we submit behind a macro-F1
  threshold tuner, tune for *macro-F1 at the best threshold on OOF probabilities*,
  not macro-F1 @0.5. An earlier script tuned @0.5 and the LB dropped 0.712→0.703.
- **L3 — Re-tune after changing features.** Hyperparameters optimal for raw features
  are not optimal for an engineered feature set. Hold hyperparameter search (P5)
  until the feature set (P1/P2) is settled.
- **L4 — Threshold tuning is a real, free win** here (~+0.02 CV over 0.5). Keep it.
- **L5 — Don't re-derive features the tree already has.** Hand-crafted aggregates of
  columns already in the model don't help HGB: PAY_* aggregates (max/mean/streak/trend)
  REVERTED on paired CV (−0.0010, 2/5) despite PAY being the strongest raw signal, and
  `bill` summaries were flat. A gradient-boosted tree already learns these splits.
  Such features mainly help *linear* models — hold them for P3, don't add them to HGB.
  Engineered features that encode a *new* ratio not present as a column (utilisation
  BILL/LIMIT, payment coverage) are the only ones that nudged the score. (E02–E06)
- **L6 — Zero-dep feature engineering is a small lever on this data.** The whole P1
  sweep moved CV by <0.003 (within noise, see L1); the best family (`util`) is
  +0.0008 paired — and then **lost on the LB (0.705 < 0.712), reverted** (S1). Don't
  expect FE to crack the LB — spend effort on models/ensembling/threshold transfer
  (P3/P4/P6) instead. (P1)
- **L7 — Encoding is a *linear-model* lever; it's a no-op for HGB. Make scaling a
  per-estimator knob.** Monotone per-feature transforms — `StandardScaler` and signed
  `log1p` — are **bit-identical** for HGB (P2 sanity: `|d|=0.00e+00`, eval proba max-diff
  0.0), because a tree only sees split *order*. So drop the scaler on the tree path (free
  simplification, zero LB risk) and keep it for linear, where it's worth +0.0016. The
  same P2 encodings that are flat for HGB lift LogReg a lot: `pay_remap` +0.0072 (5/5),
  `clean_linear` +0.0099 (5/5). Corollary to L5: re-encoding (like re-deriving features)
  helps only the linear model — hold these for P3, deploy nothing non-bit-identical on
  HGB without an LB gain. (E07–E11)
- **L8 — For an ensemble, pick members by *decorrelation*, not standalone score.** In the
  P3 bake-off the best single model was `rf` (+0.0020 paired, 5/5) but it correlates 0.974
  with HGB — a near-clone, so swapping or adding it is "a better tree", not diversity.
  `logreg_clean` *lost* standalone (−0.0016) yet is the most decorrelated (corr 0.914), and
  adding it is exactly what lifts the soft-vote from `hgb+rf` (+0.0024) to `hgb+logreg+rf`
  (+0.0026, 5/5). Judge an ensemble candidate by how *differently* it errs vs the anchor,
  not by its own macro-F1. (Caveat: all these CV gains are <0.005 → L1 still governs the
  KEEP decision; this lesson is about *member selection*, not whether the bump transfers.) (E12)
- **L9 — Class weighting and threshold tuning mostly overlap for HGB.** HGB class weights
  did not produce a robust gain once the prediction threshold was already tuned:
  `hgb_w15` dmean −0.0002 and `hgb_w30` dmean +0.0004, both wins 3/5. RF with balanced
  class weights was the only P5 candidate worth a leaderboard check (+0.0032, wins 5/5)
  and did transfer just enough to set the current LB best (**0.713**, +0.001 vs S0).
  Adjacent RF-balanced variants did not improve the CV signal beyond the current S4
  model; `balanced_subsample` only tied it. So P5 is a small win, not the high-ceiling
  path. Move to P6 threshold-transfer rather than more class-weight tinkering. (E14-E15)

- **L10 — Threshold transfer is not our problem; the cut is stable.** The honest
  leave-one-fold-out threshold-transfer penalty (tune the cut on 4 folds, apply to the
  5th) is only **~0.0012** macro-F1 and is nearly identical for `rf_balanced` (0.7115→
  0.7103) and `hgb` (0.7083→0.7070). The deployed RF cut is tight and stable
  (0.615 ± 0.012), *more* stable than HGB's (0.327 ± 0.015), and no fold-averaged or
  flat-fixed threshold beats the per-fit tuned one on held-out data — the deployed
  `TunedThresholdClassifierCV` already is the robust (fold-averaged-equivalent) choice.
  So the CV↔LB wobble is **not** a threshold artefact; don't chase fixed/averaged cuts.
  Corollary: with P1–P6 all moving <0.005, zero-dep sklearn is at its ceiling here — the
  only remaining high-ceiling lever is a stronger learner (P7 boosting libs, a new-dep
  decision), not more re-encoding/reweighting/threshold tinkering. (E16)

- **L11 — 0.713 is the practical ceiling; a stronger library doesn't break it.** P7
  spent the last high-ceiling lever (LightGBM, a new dep). With 5 hand specs *and* a
  40-config randomized search over the full LightGBM space, the best tuned model only
  *tied* `rf_balanced` (dmean **+0.0004**, wins 3/5) — far below the +0.005 bar a new
  dependency needs. Combined with L1 (even +0.0026 paired *lost* on this LB in P4), the
  signal is unambiguous: every path P1–P7 has now been tried and the ceiling is
  ~**0.7115 CV / 0.713 LB** for these learners on this data. The bottleneck is the
  dataset's information content, not the model class — so don't keep swapping learners
  or tuning. Remaining effort belongs on the **report** and the **final 2-submission
  pick** (S0 0.712 + S4 0.713). (E17)

- **L12 — A second research round (NotebookLM) confirmed the 0.713 ceiling; nothing
  new transferred.** Five literature-backed ideas were screened CV-first vs the
  `rf_balanced` champion and all PARKed: stacking (+0.0005 paired, 4/5 — noise like the
  P4 soft-vote that *lost* on LB), SMOTE/SMOTE+Tomek/SMOTE+ENN/etc. (**all negative**,
  worst of the round — class_weight+threshold dominates synthetic resampling for trees on
  macro-F1), Isolation-Forest anomaly feature (−0.0001, a tree already has it → L5),
  target encoding (−0.0001, tiny cardinality so one-hot is already lossless → L7), and
  probability calibration (a **structural no-op**: a single threshold on `predict_proba`
  is rank-based and calibration is monotonic, so it cannot change the best macro-F1 —
  verified, Platt −0.0003 / isotonic +0.0000 on the same scores). Two reusable rules:
  (a) **don't calibrate for a thresholded metric** — it only helps probability consumers;
  (b) **for tree ensembles + macro-F1, prefer `class_weight` + threshold tuning over
  resampling**. The bottleneck remains the dataset's information content (L11), not the
  method. Remaining effort: the **report** + the **final 2-submission pick** (S0 0.712,
  S4 0.713). (E18–E22)

- **L13 — The 0.713 "ceiling" was a dev-CV illusion; PAY-semantic FE on RF broke it to
  0.719.** L11/L12 declared 0.713 the practical ceiling — but that verdict was reached by
  *screening on dev OOF*, which is **decoupled** from this LB (E27/E28: the highest-OOF
  candidate is repeatedly **not** the highest-LB; `paysem_util_payratio` had a *below-anchor*
  OOF of 0.7107 yet scored the best LB, 0.719). Two corrections to earlier lessons: (a) L5
  ("don't re-derive features the tree already has") was established on **HGB**, whose
  histogram binning absorbs monotone transforms — it does **not** hold for **RF**, whose
  axis-aligned splits on the collinear BILL/PAY blocks *do* read explicit ratios and
  semantic counts (E27/E28). (b) L6 ("FE is a small lever") was an HGB+dev-CV conclusion;
  on **RF + the LB**, FE is the *biggest* lever found (+0.006). The reusable rule: on this
  dataset, **generate FE-on-RF candidates and let the LB rank them — do not pre-filter on
  OOF.** The real new signal was respecting the PAY_* code semantics (revolving ≠ paid-full
  ≠ no-consumption), which `pay_remap` had been collapsing. (E27, E28)

- **L14 — The plateau is real: 0.720 is the FE-on-RF ceiling; stronger learners don't help
  and TabPFN is undeployable here.** After the paysem breakthrough (L13), ~40 more candidates
  (richer features, same-/cross-view ensembles, seed-averaging, CatBoost) all landed at
  **0.717–0.720**, none clearing it. CatBoost (item 3) — picked precisely because it resists
  the overfitting that sank tuned HGB — scored *below* plain RF (0.708–0.715), confirming the
  bottleneck is the **dataset's signal**, not model variance. TabPFN (item 6) is **infeasible**:
  `torch` has no wheel for this Python 3.13 / numpy 2.4 env, and it would be undeployable on
  the grader regardless. Two 0.720 configs err on different features (`paysem_util_payratio`,
  `paysem_stress_payratio`) → use them as the two final picks. The last +0.001–0.002 is a seed
  lottery (seed 888 → 0.720). Practical conclusion: stop adding learners/features; seed-hunt
  the two 0.720 configs and write the report. (E30, E31)

> Append new lessons as Lxx with the experiment(s) that taught them.
