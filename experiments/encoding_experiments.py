"""Encoding / preprocessing sweep, kept OUT of main.py.

A counterpart to 'feature_experiments.py'. Scores each named encoding variant from
'config.yaml' ('experiments.encoding_configs') on the **deployed objective**, OOF
'predict_proba' then macro-F1 at the best threshold over the grid.

Every variant is screened on **two estimators**:

- 'hgb'  — the tree baseline. A tree is invariant to monotone rescaling, so these
  encodings are expected to read as noise; the point is to confirm they are neutral and
  do NOT hurt. The 'noscale' variant must be *bit-identical* to 'baseline' here
  (StandardScaler is a per-feature monotone map) — an explicit sanity check below.
- 'logreg' — 'LogisticRegression(class_weight="balanced")', the linear baseline. A
  clean ordinal scale / compressed tails help a linear model a lot, so this is where the
  real payoff (if any) shows up.

For each estimator it (1) reproduces 'baseline' as an anchor, (2) screens every other
variant at the primary seed and prints the delta, and (3) validates the ones that beat
baseline with paired repeated CV across the fold seeds, a real edge must win on *every*
seed (single-seed CV gains <0.005 are noise). Estimators stay at their baseline
hyper-parameters so the *encoding* effect is isolated.

Usage:
    python experiments/encoding_experiments.py                  # both estimators, all variants
    python experiments/encoding_experiments.py log1p bill_clip  # only these named variants
    python experiments/encoding_experiments.py --kind logreg    # one estimator only
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
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

# The two estimators every variant is judged on (tree baseline + linear baseline).
KINDS = ("hgb", "logreg")


def build_estimator(kind: str):
    """The base estimator for a given kind, at its baseline hyper-parameters."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=config.SEED)
    if kind == "logreg":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=config.SEED
        )
    raise ValueError(f"unknown estimator kind {kind!r} (expected one of {KINDS})")


def oof_macro_f1(kind: str, encoding: dict, X, y, seed: int) -> tuple[float, float]:
    """OOF macro-F1 of 'kind' + 'encoding' at its best threshold (deployed objective)."""
    pipe = make_pipeline(build_estimator(kind), encoding=encoding)
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


def run_kind(kind: str, configs: dict[str, dict], X, y) -> None:
    """Screen + paired-CV validate every encoding variant for one estimator."""
    print(f"\n{'=' * 64}\n  ESTIMATOR: {kind}\n{'=' * 64}")

    # 1-2. Screen at the primary seed.
    print(f"--- Screen: OOF macro-F1 @best threshold (seed {SCREEN_SEED}) ---")
    base_thr, base_f1 = oof_macro_f1(kind, configs["baseline"], X, y, SCREEN_SEED)
    anchor = "   (anchor; expect ~0.7076)" if kind == "hgb" else "   (anchor)"
    print(f"  {'baseline':16s} {base_f1:.4f} @thr {base_thr:.3f}{anchor}")
    screen: dict[str, float] = {}
    for name, enc in configs.items():
        if name == "baseline":
            continue
        thr, f1 = oof_macro_f1(kind, enc, X, y, SCREEN_SEED)
        screen[name] = f1
        print(f"  {name:16s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}")

    # Sanity: dropping StandardScaler is a no-op for a tree (bit-identical OOF).
    if kind == "hgb" and "noscale" in screen:
        same = abs(screen["noscale"] - base_f1) < 1e-9
        print(f"  [sanity] hgb noscale == baseline: {'PASS' if same else 'FAIL'} "
              f"(|d|={abs(screen['noscale'] - base_f1):.2e})")

    # 3. Paired repeated-CV validation of everything that screened above baseline.
    promising = [n for n, f1 in screen.items() if f1 > base_f1]
    if not promising:
        print("  No variant beat baseline at the screen seed, nothing to validate.")
        return

    print(f"--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) ---")
    base = np.array([oof_macro_f1(kind, configs["baseline"], X, y, s)[1] for s in VALIDATION_SEEDS])
    print(f"  {'baseline':16s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")
    results = []
    for name in promising:
        fs = np.array([oof_macro_f1(kind, configs[name], X, y, s)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(
            f"  {name:16s} {fs.mean():.4f} +/- {fs.std():.4f}   "
            f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}"
        )

    results.sort(reverse=True)
    print(f"--- {kind}: ranked by paired mean delta vs baseline ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:16s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


def main() -> None:
    config.set_seed()

    # Optional '--kind hgb|logreg' filter; remaining args name encoding variants.
    args = sys.argv[1:]
    kinds = list(KINDS)
    if "--kind" in args:
        i = args.index("--kind")
        kinds = [args[i + 1]]
        args = args[:i] + args[i + 2:]

    names = args or list(config.ENCODING_CONFIGS)
    unknown = [n for n in names if n not in config.ENCODING_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown variant(s) {unknown}; known: {list(config.ENCODING_CONFIGS)}")
    configs = {n: config.ENCODING_CONFIGS[n] for n in names}
    configs = {"baseline": config.ENCODING_CONFIGS["baseline"], **configs}  # always anchor

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Variants: {list(configs)}")

    for kind in kinds:
        run_kind(kind, configs, X, y)


if __name__ == "__main__":
    main()
