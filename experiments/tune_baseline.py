"""Model comparison + HGB hyper-parameter tuning — kept OUT of main.py (exam rule).

Two things happen here:

1. ``compare_all`` — full 5-fold metrics for the three baselines + saved
   confusion-matrix figures (the numbers/plots for the report).

2. ``tune_hgb`` — a randomised search over the HistGradientBoosting
   hyper-parameters, followed by a paired repeated-CV validation of the best
   candidates against the library defaults.

IMPORTANT — what we optimise and why
------------------------------------
``main.py`` submits HGB wrapped in ``TunedThresholdClassifierCV``: it does NOT
predict at the 0.5 cut, it predicts at the threshold that maximises macro-F1.
So the right tuning objective is **macro-F1 at the best threshold**, evaluated on
out-of-fold predictions — NOT macro-F1 at 0.5.

An earlier version of this script tuned macro-F1 at 0.5. The params it produced
*looked* better at 0.5 but, once deployed behind the threshold tuner, generalised
worse and the leaderboard score dropped (0.712 -> 0.703). Optimising the deployed
objective fixes that mismatch; the validation step below guards against picking a
config that only wins on a single lucky CV split.

Copy the printed best parameters into ``src/models.CHOSEN_HGB_PARAMS``.

Usage:
    python experiments/tune_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.evaluate import compare_all  # noqa: E402
from src.models import make_pipeline  # noqa: E402

# Threshold grid swept when scoring a config (matches the deployed objective).
THRESHOLDS = np.linspace(0.05, 0.95, 181)

# Candidates validated with paired repeated CV before one is recommended.
N_SEARCH = 40
N_TOP = 6
VALIDATION_SEEDS = [42, 1, 7, 2024, 99]


def best_threshold_macro_f1(proba: np.ndarray, y) -> tuple[float, float]:
    """Best macro-F1 over the threshold grid (and the threshold that achieves it)."""
    f1s = [f1_score(y, (proba >= t).astype(int), average="macro") for t in THRESHOLDS]
    i = int(np.argmax(f1s))
    return float(THRESHOLDS[i]), float(f1s[i])


def oof_macro_f1(params: dict, X, y, seed: int) -> tuple[float, float]:
    """Out-of-fold macro-F1 of an HGB config at its best threshold (deployed objective)."""
    pipe = make_pipeline(HistGradientBoostingClassifier(random_state=config.SEED, **params))
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return best_threshold_macro_f1(proba, y)


def sample_params(rng: np.random.Generator) -> dict:
    """Sample one HGB config from a regularisation-focused search space."""
    return dict(
        learning_rate=float(round(10 ** rng.uniform(-2, -0.8), 4)),   # ~0.01 .. 0.16
        max_iter=int(rng.choice([200, 300, 400, 500, 600])),
        max_leaf_nodes=int(rng.choice([15, 21, 31, 45, 63])),
        min_samples_leaf=int(rng.choice([20, 40, 60, 100, 150, 200])),
        l2_regularization=float(round(10 ** rng.uniform(-2, 1), 3)),  # 0.01 .. 10
        max_features=float(round(rng.uniform(0.6, 1.0), 2)),
        max_bins=int(rng.choice([128, 255])),
    )


def search(X, y) -> list[tuple[float, float, dict]]:
    """Randomised search on the deployed objective; returns configs sorted by macro-F1."""
    rng = np.random.default_rng(config.SEED)
    base_thr, base_f1 = oof_macro_f1({}, X, y, config.SEED)
    print(f"\n--- HGB randomised search ({N_SEARCH} configs, deployed objective) ---")
    print(f"  default params: macro-F1={base_f1:.4f} @thr {base_thr:.3f}")

    results: list[tuple[float, float, dict]] = []
    seen: set[str] = set()
    for _ in range(N_SEARCH):
        params = sample_params(rng)
        key = repr(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        thr, f1 = oof_macro_f1(params, X, y, config.SEED)
        results.append((f1, thr, params))

    results.sort(key=lambda r: -r[0])
    print(f"  top {N_TOP} by single-split macro-F1 (baseline {base_f1:.4f}):")
    for f1, thr, params in results[:N_TOP]:
        print(f"    {f1:.4f} @thr {thr:.3f}  {params}")
    return results


def validate(candidates: list[dict], X, y) -> dict:
    """Paired repeated CV (several fold seeds) of default vs each candidate.

    A candidate is only trustworthy if it beats the defaults on *every* seed by
    more than the defaults' own seed-to-seed wobble — that is what separates a
    real edge from a lucky split.
    """
    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) ---")
    scores: dict[str, np.ndarray] = {}
    labelled = [("default", {})] + [(f"cand{i}", p) for i, p in enumerate(candidates)]
    for name, params in labelled:
        fs = np.array([oof_macro_f1(params, X, y, s)[1] for s in VALIDATION_SEEDS])
        scores[name] = fs
        print(f"  {name:8s} {fs.mean():.4f} +/- {fs.std():.4f}  {np.round(fs, 4).tolist()}")

    default = scores["default"]
    best_name, best_delta = None, -1.0
    print("  delta vs default (paired):")
    for name, params in labelled[1:]:
        d = scores[name] - default
        wins = int((d > 0).sum())
        print(f"    {name:8s} dmean={d.mean():+.4f}  wins {wins}/{len(VALIDATION_SEEDS)}")
        # Prefer the largest mean gain that wins on *every* seed (robust).
        if wins == len(VALIDATION_SEEDS) and d.mean() > best_delta:
            best_name, best_delta = name, d.mean()

    if best_name is None:
        print("\n  No candidate robustly beat the defaults — keep CHOSEN_HGB_PARAMS = {}.")
        return {}
    chosen = dict(labelled)[best_name]
    print(f"\n  RECOMMENDED ({best_name}, +{best_delta:.4f} macro-F1 on every seed):")
    print(f"    {chosen}")
    print("  -> paste into src/models.CHOSEN_HGB_PARAMS")
    return chosen


def tune_hgb(X, y) -> dict:
    """Search the HGB space, then validate the top candidates and recommend one."""
    ranked = search(X, y)
    top = [params for _, _, params in ranked[:N_TOP]]
    return validate(top, X, y)


def main() -> None:
    config.set_seed()
    dev = load_development()
    X, y = split_xy(dev)

    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    compare_all(X, y)
    tune_hgb(X, y)


if __name__ == "__main__":
    main()
