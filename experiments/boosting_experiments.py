"""Stronger boosting-library sweep, kept OUT of main.py.

The sklearn estimator space plateaued (the sweeps all moved dev CV by
<0.005). This script tries another dependency: lightgbm + an OpenMP runtime.

Usage:
    python experiments/boosting_experiments.py
    python experiments/boosting_experiments.py lgbm_balanced lgbm_reg_balanced
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402

CHAMPION = "rf_balanced"   # the deployed model every candidate must beat
REFERENCE = "hgb"          # printed for context (the original anchor)
SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
MIN_DELTA = 0.005          # the "transfers to the leaderboard" bar


def best_threshold_macro_f1(proba: np.ndarray, y) -> tuple[float, float]:
    """Best macro-F1 over the project threshold grid."""
    f1s = [f1_score(y, (proba >= t).astype(int), average="macro") for t in config.THRESHOLDS]
    i = int(np.argmax(f1s))
    return float(config.THRESHOLDS[i]), float(f1s[i])


def _resolve(name: str) -> dict:
    """Resolve a config name to a model spec (boosting_configs or model_configs)."""
    if name in config.BOOSTING_CONFIGS:
        return dict(config.BOOSTING_CONFIGS[name])
    if name in config.MODEL_CONFIGS:
        return dict(config.MODEL_CONFIGS[name])
    raise ValueError(f"unknown config {name!r}; known boosting: {list(config.BOOSTING_CONFIGS)}")


def oof_proba(spec: dict, X, y, seed: int) -> np.ndarray:
    """OOF positive-class probabilities for one resolved model spec."""
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    pipe = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def _score(spec: dict, X, y, seed: int) -> tuple[float, float]:
    return best_threshold_macro_f1(oof_proba(spec, X, y, seed), y)


def _label(dmean: float, wins: int, n: int) -> str:
    if wins == n and dmean >= MIN_DELTA:
        return "KEEP (robust, clears the bar) -> worth an LB slot"
    if wins == n and dmean > 0:
        return "robust but small (<0.005) -> LB-gate it"
    if dmean > 0:
        return "watch (not all seeds)"
    return "revert"


def main() -> None:
    config.set_seed()

    args = sys.argv[1:]
    names = args or list(config.BOOSTING_CONFIGS)
    unknown = [n for n in names if n not in config.BOOSTING_CONFIGS and n not in config.MODEL_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s) {unknown}; known boosting: {list(config.BOOSTING_CONFIGS)}")

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Candidates: {names}")

    champ_spec = _resolve(CHAMPION)
    ref_spec = _resolve(REFERENCE)
    cand_specs = {n: _resolve(n) for n in names}

    print(f"\n{'=' * 68}\n  LIGHTGBM vs deployed {CHAMPION} (seed {SCREEN_SEED})\n{'=' * 68}")
    ref_thr, ref_f1 = _score(ref_spec, X, y, SCREEN_SEED)
    base_thr, base_f1 = _score(champ_spec, X, y, SCREEN_SEED)
    print(f"  {REFERENCE:18s} {ref_f1:.4f} @thr {ref_thr:.3f}   (reference anchor)")
    print(f"  {CHAMPION:18s} {base_f1:.4f} @thr {base_thr:.3f}   (champion / bar to beat)")

    screened = {}
    for name, spec in cand_specs.items():
        thr, f1 = _score(spec, X, y, SCREEN_SEED)
        screened[name] = f1
        print(f"  {name:18s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f} vs champ")

    candidates = {n: s for n, s in cand_specs.items() if screened[n] > base_f1}
    if not candidates:
        print(f"\nNothing beat the deployed {CHAMPION} at the screen seed — "
              f"no LB-worthy LightGBM config, dependency not justified.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {CHAMPION} ---")
    base = np.array([_score(champ_spec, X, y, s)[1] for s in VALIDATION_SEEDS])
    print(f"  {CHAMPION:18s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")

    results = []
    for name, spec in candidates.items():
        fs = np.array([_score(spec, X, y, s)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(f"  {name:18s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")

    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {CHAMPION} ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:18s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
