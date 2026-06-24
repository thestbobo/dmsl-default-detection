"""Resampling / hybrid-sampling sweep, kept out of main.py.

The imbalance sweep only tried class_weight; this asks whether rebalancing the training
folds (SMOTE and hybrids) beats class weighting + threshold tuning on macro-F1. The
sampler runs inside an imblearn Pipeline, so it only fires on the training folds during
fit (the held-out fold is never resampled). Because resampling does the rebalancing the
base estimator is the plain rf. Scored on the deployed objective vs the rf_balanced
champion. Optional dependency: imbalanced-learn.

Usage:
    python experiments/resampling_experiments.py
    python experiments/resampling_experiments.py smotetomek smote
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from imblearn.combine import SMOTEENN, SMOTETomek  # noqa: E402
from imblearn.over_sampling import SMOTE, BorderlineSMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from imblearn.under_sampling import RandomUnderSampler, TomekLinks  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from experiments._submit_utils import best_threshold_macro_f1, verdict  # noqa: E402
from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402

SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
ANCHOR = "rf_balanced"


def _make_sampler(key: str, seed: int):
    """Resolve a sampler key to an imblearn resampler (seeded where supported)."""
    if key == "smote":
        return SMOTE(random_state=seed)
    if key == "smotetomek":
        return SMOTETomek(random_state=seed)
    if key == "smoteenn":
        return SMOTEENN(random_state=seed)
    if key == "borderline":
        return BorderlineSMOTE(random_state=seed)
    if key == "undersample":
        return RandomUnderSampler(random_state=seed)
    if key == "tomek":
        return TomekLinks()  # deterministic; no random_state
    raise ValueError(f"unknown sampler {key!r}")


def champion_pipe():
    spec = config.MODEL_CONFIGS[ANCHOR]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)


def resampling_pipe(rspec: dict, seed: int) -> ImbPipeline:
    """imblearn Pipeline: base preprocessing -> sampler -> base estimator."""
    base_name = rspec["base_model"]
    spec = config.MODEL_CONFIGS[base_name]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    base = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    sampler = _make_sampler(rspec["sampler"], seed)
    steps = base.steps[:-1] + [("sampler", sampler), base.steps[-1]]
    return ImbPipeline(steps)


def oof(pipe, X, y, seed: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def main() -> None:
    config.set_seed()

    names = sys.argv[1:] or list(config.RESAMPLING_CONFIGS)
    unknown = [n for n in names if n not in config.RESAMPLING_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown resampling config(s) {unknown}; known: {list(config.RESAMPLING_CONFIGS)}")

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Candidates: {names}")

    print(f"\n{'=' * 64}\n  RESAMPLING (deployed objective, seed {SCREEN_SEED})\n{'=' * 64}")
    base_thr, base_f1 = best_threshold_macro_f1(oof(champion_pipe(), X, y, SCREEN_SEED), y)
    print(f"  {ANCHOR + ' (champion)':24s} {base_f1:.4f} @thr {base_thr:.3f}")

    screened: dict[str, float] = {}
    for name in names:
        rspec = config.RESAMPLING_CONFIGS[name]
        thr, f1 = best_threshold_macro_f1(oof(resampling_pipe(rspec, SCREEN_SEED), X, y, SCREEN_SEED), y)
        screened[name] = f1
        print(f"  {name:24s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}   "
              f"{rspec['sampler']} + {rspec['base_model']}")

    winners = [n for n in names if screened[n] > base_f1]
    if not winners:
        print(f"\nNothing beat the {ANCHOR} champion at the screen seed; nothing to validate.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {ANCHOR} ---")
    base = np.array([best_threshold_macro_f1(oof(champion_pipe(), X, y, s), y)[1] for s in VALIDATION_SEEDS])
    print(f"  {ANCHOR:24s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")

    results = []
    for name in winners:
        rspec = config.RESAMPLING_CONFIGS[name]
        fs = np.array([best_threshold_macro_f1(oof(resampling_pipe(rspec, s), X, y, s), y)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(f"  {name:24s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")

    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {ANCHOR} ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:24s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {verdict(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
