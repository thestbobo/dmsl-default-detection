"""Probability-calibration check (Path P12) — kept OUT of main.py.

NotebookLM idea #3: calibrate ``rf_balanced`` probabilities (isotonic / sigmoid) before
the threshold tuner, on the theory that RF's "pushed-in" probabilities make thresholding
unstable.

The key argument this script makes concrete: we submit behind a SINGLE threshold on
predict_proba, and both sigmoid (Platt) and isotonic calibration are MONOTONIC maps.
A monotonic transform preserves the rank order of the scores, so the set of achievable
(precision, recall) operating points — and therefore the best thresholded macro-F1 — is
unchanged. Calibration cannot help a rank-based cut; it only matters when you consume the
probabilities as probabilities (Brier/log-loss, cost-based decisions, multiclass).

Two demonstrations:
  (A) MONOTONIC-MAP no-op: fit isotonic + Platt on the champion's OOF scores, transform
      the SAME scores, recompute best-threshold macro-F1. Expect ~identical (the math).
  (B) REAL CalibratedClassifierCV(rf_balanced) wrapped + OOF-scored, screen + paired CV.
      Any movement here is the incidental cross-fit *ensembling* inside CalibratedCV
      (it refits the base estimator on internal folds and averages), NOT calibration.

Usage:
    python experiments/calibration_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402
from experiments.tune_baseline import best_threshold_macro_f1  # noqa: E402

SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
MIN_DELTA = 0.005
ANCHOR = "rf_balanced"


def champion_pipe():
    spec = config.MODEL_CONFIGS[ANCHOR]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)


def calibrated_pipe(method: str):
    """rf_balanced wrapped in CalibratedClassifierCV (internal 5-fold)."""
    spec = config.MODEL_CONFIGS[ANCHOR]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    base = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    return CalibratedClassifierCV(base, method=method, cv=5)


def oof(pipe, X, y, seed: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def _label(dmean: float, wins: int, n: int) -> str:
    if wins == n and dmean >= MIN_DELTA:
        return "KEEP (robust)"
    if wins == n and dmean > 0:
        return "keep? (robust but small)"
    if dmean > 0:
        return "watch (not all seeds)"
    return "revert"


def main() -> None:
    config.set_seed()
    dev = load_development()
    X, y = split_xy(dev)
    yv = y.to_numpy()
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")

    champ = oof(champion_pipe(), X, y, SCREEN_SEED)
    base_thr, base_f1 = best_threshold_macro_f1(champ, y)

    print(f"\n{'=' * 64}\n  P12(A) — MONOTONIC-MAP NO-OP (seed {SCREEN_SEED})\n{'=' * 64}")
    print(f"  {'rf_balanced (raw proba)':32s} {base_f1:.4f} @thr {base_thr:.3f}")
    # Platt / sigmoid: strictly monotonic logistic on the score.
    platt = LogisticRegression().fit(champ.reshape(-1, 1), yv).predict_proba(champ.reshape(-1, 1))[:, 1]
    p_thr, p_f1 = best_threshold_macro_f1(platt, y)
    print(f"  {'+ Platt/sigmoid (monotonic)':32s} {p_f1:.4f} @thr {p_thr:.3f}   d={p_f1 - base_f1:+.4f}")
    # Isotonic: monotonic non-decreasing step function.
    iso = IsotonicRegression(out_of_bounds="clip").fit(champ, yv).predict(champ)
    i_thr, i_f1 = best_threshold_macro_f1(iso, y)
    print(f"  {'+ isotonic (monotonic)':32s} {i_f1:.4f} @thr {i_thr:.3f}   d={i_f1 - base_f1:+.4f}")
    print("  -> monotonic calibration preserves rank order => thresholded macro-F1 unchanged.")

    print(f"\n{'=' * 64}\n  P12(B) — CalibratedClassifierCV wrap (seed {SCREEN_SEED})\n{'=' * 64}")
    print(f"  {ANCHOR + ' (champion)':32s} {base_f1:.4f} @thr {base_thr:.3f}")
    candidates = {}
    for method in ("sigmoid", "isotonic"):
        thr, f1 = best_threshold_macro_f1(oof(calibrated_pipe(method), X, y, SCREEN_SEED), y)
        candidates[method] = f1
        print(f"  {'calibrated (' + method + ')':32s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}")

    winners = [m for m in candidates if candidates[m] > base_f1]
    if not winners:
        print(f"\nNothing beat the {ANCHOR} champion at the screen seed — nothing to validate.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {ANCHOR} ---")
    base = np.array([best_threshold_macro_f1(oof(champion_pipe(), X, y, s), y)[1] for s in VALIDATION_SEEDS])
    print(f"  {ANCHOR:32s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")
    results = []
    for method in winners:
        fs = np.array([best_threshold_macro_f1(oof(calibrated_pipe(method), X, y, s), y)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, method, fs))
        print(f"  {'calibrated (' + method + ')':32s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")
    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {ANCHOR} ---")
    for dmean, wins, method, _ in results:
        print(f"  {'calibrated (' + method + ')':32s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
