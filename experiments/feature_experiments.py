"""Feature-engineering sweep — kept OUT of main.py.

Scores each named feature config from 'config.yaml' ('experiments.feature_configs')
on the **deployed objective** — OOF 'predict_proba' then macro-F1 at the best
threshold over the grid, exactly as the submission is graded. It:

1. reproduces the 'baseline' (no engineered features) as a harness sanity check;
2. screens every other config at the primary seed and prints the delta;
3. validates the configs that beat baseline with paired repeated CV across the
   fold seeds — a real edge must win on *every* seed, not just a lucky split
   (single-seed CV gains <0.005 are noise).

Reuses the deployed-objective helper from 'tune_baseline.py' so the scoring is
identical to the HGB tuning path. HGB stays at library defaults for every config so
the *feature* effect is isolated (re-tuning on the winning set is a later step).

Usage:
    python experiments/feature_experiments.py              # sweep all configs
    python experiments/feature_experiments.py pay pay+util # only these named configs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_pipeline  # noqa: E402
from experiments.tune_baseline import best_threshold_macro_f1  # noqa: E402

# Screen at the primary seed; confirm any winner across all fold seeds.
SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
# A robust win must beat baseline on every seed; this much mean gain marks it a
# real edge rather than noise.
MIN_DELTA = 0.005


def oof_macro_f1(feature_groups, X, y, seed: int) -> tuple[float, float]:
    """OOF macro-F1 of HGB(defaults) + 'feature_groups' at its best threshold."""
    pipe = make_pipeline(
        HistGradientBoostingClassifier(random_state=config.SEED),
        feature_groups=feature_groups,
    )
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return best_threshold_macro_f1(proba, y)


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

    names = sys.argv[1:] or list(config.FEATURE_CONFIGS)
    unknown = [n for n in names if n not in config.FEATURE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config name(s) {unknown}; known: {list(config.FEATURE_CONFIGS)}")
    configs = {n: config.FEATURE_CONFIGS[n] for n in names}
    configs = {"baseline": config.FEATURE_CONFIGS["baseline"], **configs}  # always anchor

    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    # 1-2. Screen at the primary seed.
    print(f"\n--- Screen: OOF macro-F1 @best threshold (seed {SCREEN_SEED}) ---")
    base_thr, base_f1 = oof_macro_f1(configs["baseline"], X, y, SCREEN_SEED)
    print(f"  {'baseline':12s} {base_f1:.4f} @thr {base_thr:.3f}   (anchor; expect ~0.7076)")
    screen: dict[str, tuple[float, float, list]] = {}
    for name, groups in configs.items():
        if name == "baseline":
            continue
        thr, f1 = oof_macro_f1(groups, X, y, SCREEN_SEED)
        screen[name] = (f1, thr, groups)
        print(f"  {name:12s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}   {groups}")

    # 3. Paired repeated-CV validation of everything that screened above baseline.
    promising = [n for n, (f1, _, _) in screen.items() if f1 > base_f1]
    if not promising:
        print("\nNo config beat baseline at the screen seed, nothing to validate.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) ---")
    base = np.array([oof_macro_f1(configs["baseline"], X, y, s)[1] for s in VALIDATION_SEEDS])
    print(f"  {'baseline':12s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")
    results = []
    for name in promising:
        fs = np.array([oof_macro_f1(configs[name], X, y, s)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(
            f"  {name:12s} {fs.mean():.4f} +/- {fs.std():.4f}   "
            f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}"
        )

    results.sort(reverse=True)
    print("\n--- Ranked by paired mean delta vs baseline ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:12s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   -> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
