# Report ⟺ code/log parity table (Step 3 → Step 4 verification)

Maps every number / figure / non-trivial claim in `report/report.tex` to its source.
Two reproducible commands generate all CV numbers and figures **at the deployed seed (8888)**:

- `python main.py` → `outputs/submissions/submission.csv` (the deployed submission)
- `python experiments/tune_baseline.py` → baseline Table I numbers, final-model numbers, and
  `outputs/figures/cm_final_tuned.png` (copied to `report/figures/`)

LB = public-leaderboard scores. These are **not reproducible from code** (they are leaderboard
results); they are backed by `outputs/submissions/MANIFEST.md` + the experiment log + the user's
leaderboard record. Graders do not regenerate, so this is expected and acceptable.

| # | Report claim / number | Value | Source (kept code / log) | Reproducible now? |
|---|---|---|---|---|
| 1 | Dataset 30k clients; 24k dev / 6k eval; 23 features | — | `src/data.py`, `config.yaml`, `notebooks/01_eda.ipynb` | ✅ |
| 2 | Class balance 77.9% / 22.1% | 0.779 / 0.221 | `tune_baseline.py` stdout ("Class balance…"); EDA notebook | ✅ |
| 3 | EDUCATION {0,5,6}→4, MARRIAGE {0}→3 | — | `src/preprocessing.py` (CodeFolder) | ✅ |
| 4 | PAY_* codes −2/−1/0/1..9 semantics | — | `src/preprocessing.py` (`paysem` block + comments); Yeh 2009 [1] | ✅ |
| 5 | macro-F1 metric choice | — | scoring in `config.yaml`/`evaluate.py`; He & Garcia [5] | ✅ |
| 6 | Leakage-free Pipeline; ColumnTransformer; median-impute; one-hot; PAY ordinal; scaler bit-identical no-op for trees | — | `src/preprocessing.py`, `src/models.py`; exp-log L7 | ✅ |
| 7 | **Table I** baselines (tuned, seed 8888): dummy 0.505/0.505; logreg 0.697/0.728; **HGB 0.709/0.782** | see table | `python experiments/tune_baseline.py` (`compare_all_tuned`) | ✅ regenerated |
| 8 | Threshold tuning lift: HGB 0.688 (0.5 cut) → 0.709 (@≈0.33), +0.021 | +0.021 | `tune_baseline.py` (`compare_all` 0.6882; `compare_all_tuned` 0.7090 @0.330) | ✅ |
| 9 | HGB randomised search gain ≤ +0.002 | +0.0020 | `tune_baseline.py` (`tune_hgb`, RECOMMENDED cand5) | ✅ |
| 10 | Bake-off: RF most reliable (wins every seed), logreg most decorrelated, gb/et within noise | — | `experiments/model_experiments.py`; exp-log P3/E12 | ✅ (run script) |
| 11 | rf_balanced = RF 300 trees, leaf 20, class_weight balanced; LB 0.713; beats HGB every fold seed | LB 0.713 | `config.yaml` (model_configs); `experiments/imbalance_experiments.py`; exp-log P5/E14; MANIFEST S4 | CV ✅ / LB log |
| 12 | **Table II** CV (seed 8888): HGB 0.709; RF 0.711; +payratio 0.711; +ps+ut+pr 0.711; +ps+st+pr 0.711 | see table | `tune_baseline.py` (final) + `_member_pipeline` CV at seed 8888 | ✅ regenerated |
| 13 | **Table II** LB: 0.712 → 0.713 → 0.716 → 0.720 (A) → 0.721 (B) | see table | MANIFEST (S0/S4/sub23) + configs A/B + user/LB | LB log |
| 14 | Feature formulas: `paysem` (revolving/paid-full/no-cons counts + latest-month flags), `payratio`, `stress`, `util` | — | `src/preprocessing.py` (ENGINEERED_COLUMNS) | ✅ |
| 15 | FE helps RF not HGB; +payratio→0.716, +paysem→0.720–0.721 | — | exp-log P14/L13; MANIFEST subs; `src/preprocessing.py` | LB log + ✅ code |
| 16 | **Final model** (config_B, seed 8888): OOF macro-F1 0.711 @≈0.61; ROC 0.786; per-class 0.866/0.892/0.879 & 0.575/0.516/0.544; CM [[16664,2027],[2570,2739]] | see Fig 1 | `python experiments/tune_baseline.py` (`cross_validate_tuned("final", build_chosen_model())`) → `cm_final_tuned.png` | ✅ regenerated |
| 17 | Deployed = config_B; `main.py` reproduces it; thr 0.6064; 1229/6000 (20.5%) defaults; 9.4 s | — | `python main.py`; `config.yaml` `chosen`+`seed`; `submission.csv` | ✅ regenerated |
| 18 | Headline LB 0.721; two finalists A 0.720 / B 0.721 (diverse errors) | 0.721 / 0.720 | configs A/B; user/LB | LB log |
| 19 | "What did not help": LightGBM only matched RF; SMOTE/resampling underperformed; soft-vote didn't transfer | — | `boosting_experiments.py` (P7), `resampling_experiments.py` (P10), `model_experiments.py` (P4) — **all KEPT** | ✅ (run scripts) |
| 20 | Tree methods plateau ≈0.72 | — | exp-log L14 | LB log |
| 21 | CV↔LB decoupling (OOF doesn't rank candidates; best-LB not best-OOF) | — | Table II (this session, seed 8888); exp-log L13 | ✅ |
| 22 | **Fig 1** deployed confusion matrix | `report/figures/cm_final_tuned.png` | regenerated via `tune_baseline.py` | ✅ regenerated |
| 23 | Reference [3] Breiman, Random Forests (RF justification) | — | `report/report.tex` thebibliography | ✅ |

## Flags for Step 4

- **LB numbers rest on user/MANIFEST authority** (rows 11, 13, 15, 18, 20), not on code — expected,
  since the public leaderboard cannot be reproduced offline. `MANIFEST.md` is on the Step-4 removal
  list, so in the final bundle these numbers are backed only by the report itself + the user's record.
- **Discussion "what did not help" cites only KEPT scripts** (`boosting_experiments.py`,
  `resampling_experiments.py`, `model_experiments.py`) — gold-rule compliant. The report does **not**
  mention any dropped experiment (stacking / calibration / target-encoding / anomaly / CatBoost /
  TabPFN / seed-averaging). ✅
- **Bake-off qualitative claims** (row 10) are reproducible by running `model_experiments.py` but are
  not auto-saved to a figure/log artifact; if Step 4 wants them pinned, capture that script's stdout.
- **Unused figure:** `report/figures/cm_baseline.png` (Step-2 baseline confusion matrix) is no longer
  referenced by `report.tex` (Fig 1 is now the final model). Safe to delete in Step 4 cleanup.
- **`report/report.pdf`** was built locally with `tectonic`; the intended grader build is Overleaf
  (IEEEtran is unmodified and vendored as `report/IEEEtran.cls`).
