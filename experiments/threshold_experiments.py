"""Threshold strategy / transfer-risk sweep (Path P6) — kept OUT of main.py.

We submit behind a macro-F1 threshold tuner, so the *deployed* decision is a
single threshold chosen on dev and then applied, unchanged, to the evaluation
set. Every CV number we report (e.g. rf_balanced 0.7115) picks the threshold on
the **same** OOF probabilities it is then scored on — that is optimistic: it
assumes the dev-optimal cut is also the eval-optimal cut. Lesson L1 keeps biting
us (CV gains <0.005 don't transfer), and the deployed rf_balanced threshold is a
high, sensitive 0.622, so the live question for P6 is:

    How much of our CV macro-F1 is threshold-overfit, and would a less greedy
    threshold (fold-averaged, or a flat fixed cut) transfer dev->eval better?

Protocol (honest, leakage-aware), per fold seed:
  1. Compute OOF positive-class probabilities once (each row scored by a model
     that never saw it), recording each row's fold id.
  2. GLOBAL (optimistic, == what other scripts report): best threshold over the
     project grid on the FULL OOF, scored on the full OOF.
  3. TRANSFER (honest): for each fold k, tune the threshold on the OTHER 4 folds'
     OOF probabilities and apply it to fold k. Assemble those held-out
     predictions and score macro-F1. This simulates "tune on dev, predict eval"
     and its gap below GLOBAL is the threshold-transfer penalty.
  4. FOLD-AVG: average the 5 per-fold-tuned thresholds into one flat cut and
     score it on the full OOF (a deployable single threshold, less greedy).
  5. BEST-FIXED: the single grid threshold with the best honest transfer score
     (does a hard-coded flat cut beat per-fit tuning on held-out data?).

We also report the spread of the per-fold optimal thresholds (instability of the
cut) and the deployed TunedThresholdClassifierCV threshold for reference.

Usage:
    python experiments/threshold_experiments.py                # rf_balanced + hgb
    python experiments/threshold_experiments.py rf_balanced
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402

# Models to probe. rf_balanced is the deployed S4 model; hgb is the anchor whose
# threshold (~0.33) is far less extreme — a useful contrast for transfer risk.
PROBE = ["rf_balanced", "hgb"]
VALIDATION_SEEDS = config.VALIDATION_SEEDS


def _macro_f1_at(proba: np.ndarray, y: np.ndarray, thr: float) -> float:
    return f1_score(y, (proba >= thr).astype(int), average="macro")


def best_threshold(proba: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Best macro-F1 (and its threshold) over the project grid on (proba, y)."""
    f1s = [_macro_f1_at(proba, y, t) for t in config.THRESHOLDS]
    i = int(np.argmax(f1s))
    return float(config.THRESHOLDS[i]), float(f1s[i])


def _spec_for(name: str) -> dict:
    """Resolve a probe name to a model spec (model_configs, possibly via imbalance_configs)."""
    if name in config.MODEL_CONFIGS:
        return dict(config.MODEL_CONFIGS[name])
    if name in config.IMBALANCE_CONFIGS:  # allow imbalance candidates too
        cand = config.IMBALANCE_CONFIGS[name]
        base = dict(config.MODEL_CONFIGS[cand["base_model"]])
        params = dict(base.get("params") or {})
        params.update(cand.get("params") or {})
        base["params"] = params
        return base
    raise ValueError(f"unknown probe {name!r}; known: {list(config.MODEL_CONFIGS)}")


def fold_oof(spec: dict, X, y, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """OOF positive-class probabilities + per-row fold id, at one fold seed.

    A manual fold loop (vs cross_val_predict) so we keep the fold assignment,
    which the honest leave-one-fold-out threshold transfer needs.
    """
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    pipe = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    proba = np.zeros(len(y), dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    yv = np.asarray(y)
    for k, (tr, te) in enumerate(cv.split(X, yv)):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_id[te] = k
    return proba, fold_id


def evaluate_strategies(proba: np.ndarray, y: np.ndarray, fold_id: np.ndarray) -> dict:
    """Compute GLOBAL / TRANSFER / FOLD-AVG / BEST-FIXED for one OOF array."""
    yv = np.asarray(y)
    n_folds = fold_id.max() + 1

    # 2. GLOBAL — optimistic: tune and score on the same full OOF.
    g_thr, g_f1 = best_threshold(proba, yv)

    # 3. TRANSFER — honest: per fold, tune on the other folds, predict this fold.
    held_pred = np.zeros(len(yv), dtype=int)
    per_fold_thr = []
    for k in range(n_folds):
        te = fold_id == k
        tr = ~te
        thr_k, _ = best_threshold(proba[tr], yv[tr])
        per_fold_thr.append(thr_k)
        held_pred[te] = (proba[te] >= thr_k).astype(int)
    t_f1 = f1_score(yv, held_pred, average="macro")
    per_fold_thr = np.array(per_fold_thr)

    # 4. FOLD-AVG — deploy one flat averaged cut, scored on full OOF.
    avg_thr = float(per_fold_thr.mean())
    avg_f1 = _macro_f1_at(proba, yv, avg_thr)

    # 5. BEST-FIXED — the single grid threshold with the best honest transfer
    #    score. "Honest" here: score each fixed threshold on every held-out fold
    #    and average — a flat cut chosen without peeking per-fold.
    fixed_scores = []
    for t in config.THRESHOLDS:
        per_fold = [f1_score(yv[fold_id == k], (proba[fold_id == k] >= t).astype(int),
                             average="macro") for k in range(n_folds)]
        fixed_scores.append(float(np.mean(per_fold)))
    bi = int(np.argmax(fixed_scores))
    bf_thr, bf_f1 = float(config.THRESHOLDS[bi]), fixed_scores[bi]

    return {
        "global_thr": g_thr, "global_f1": g_f1,
        "transfer_f1": t_f1,
        "fold_thr_mean": avg_thr, "fold_thr_std": float(per_fold_thr.std()),
        "fold_thr": per_fold_thr, "foldavg_f1": avg_f1,
        "bestfixed_thr": bf_thr, "bestfixed_f1": bf_f1,
    }


def main() -> None:
    config.set_seed()

    args = sys.argv[1:]
    names = args or PROBE

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Probing: {names}   seeds={VALIDATION_SEEDS}")

    for name in names:
        spec = _spec_for(name)
        print(f"\n{'=' * 72}\n  P6 — THRESHOLD TRANSFER: {name}   {spec.get('params') or {}}\n{'=' * 72}")
        rows = []
        for s in VALIDATION_SEEDS:
            proba, fold_id = fold_oof(spec, X, y, s)
            r = evaluate_strategies(proba, y, fold_id)
            rows.append(r)
            print(f"  seed {s:>4}  GLOBAL {r['global_f1']:.4f}@{r['global_thr']:.3f}   "
                  f"TRANSFER {r['transfer_f1']:.4f}   "
                  f"FOLD-AVG {r['foldavg_f1']:.4f}@{r['fold_thr_mean']:.3f}"
                  f"(±{r['fold_thr_std']:.3f})   "
                  f"BEST-FIXED {r['bestfixed_f1']:.4f}@{r['bestfixed_thr']:.3f}")

        def col(key):
            return np.array([r[key] for r in rows])

        g, t, fa, bf = col("global_f1"), col("transfer_f1"), col("foldavg_f1"), col("bestfixed_f1")
        penalty = g - t
        print(f"\n  --- means over {len(rows)} seeds ---")
        print(f"  GLOBAL (optimistic, what we report) : {g.mean():.4f} +/- {g.std():.4f}")
        print(f"  TRANSFER (honest dev->eval estimate): {t.mean():.4f} +/- {t.std():.4f}"
              f"   penalty {penalty.mean():+.4f}")
        print(f"  FOLD-AVG (deploy one flat avg cut)  : {fa.mean():.4f} +/- {fa.std():.4f}")
        print(f"  BEST-FIXED (best flat held-out cut) : {bf.mean():.4f} +/- {bf.std():.4f}"
              f"   @thr~{np.mean([r['bestfixed_thr'] for r in rows]):.3f}")
        thr_all = np.concatenate([r["fold_thr"] for r in rows])
        print(f"  per-fold optimal threshold spread   : {thr_all.mean():.3f} "
              f"+/- {thr_all.std():.3f}   range [{thr_all.min():.3f}, {thr_all.max():.3f}]")


if __name__ == "__main__":
    main()
