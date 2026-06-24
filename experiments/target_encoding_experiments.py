"""Target-encoding sweep (Path P11) — kept OUT of main.py.

NotebookLM idea #4: replace the one-hot of SEX / EDUCATION / MARRIAGE with target
encoding (category -> mean default rate). Note this is ZERO-dependency: sklearn 1.9
ships ``sklearn.preprocessing.TargetEncoder``, which internally cross-fits the encoding
(leakage-safe) — no ``category_encoders`` needed.

Prior: our categoricals are tiny-cardinality (SEX=2, EDUCATION=4 after folding,
MARRIAGE=3), where one-hot is already lossless, so target encoding's usual high-
cardinality advantage doesn't apply. Tested anyway, CV-first, on the champion
``rf_balanced`` (and the hgb anchor), deployed objective, screen + paired CV.

Usage:
    python experiments/target_encoding_experiments.py
    python experiments/target_encoding_experiments.py rf_balanced
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import TargetEncoder  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402
from src.preprocessing import make_code_folder  # noqa: E402
from experiments.tune_baseline import best_threshold_macro_f1  # noqa: E402

SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
MIN_DELTA = 0.005
ANCHOR = "rf_balanced"
BASES = ["rf_balanced", "hgb"]


def _target_preprocessor(seed: int) -> ColumnTransformer:
    """Numeric impute + TARGET-encode the categoricals (cross-fitted, leakage-safe)."""
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("target", TargetEncoder(random_state=seed)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), list(config.NUMERIC)),
            ("cat", cat_pipe, config.CATEGORICAL),
        ],
        remainder="drop",
    )


def champion_pipe():
    spec = config.MODEL_CONFIGS[ANCHOR]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)


def target_pipe(name: str, seed: int) -> Pipeline:
    spec = config.MODEL_CONFIGS[name]
    est = make_estimator(spec["kind"], spec.get("params"))
    return Pipeline(steps=[
        ("fold", make_code_folder()),
        ("pre", _target_preprocessor(seed)),
        ("model", est),
    ])


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

    names = sys.argv[1:] or BASES
    unknown = [n for n in names if n not in config.MODEL_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; known: {list(config.MODEL_CONFIGS)}")

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Bases (one-hot -> target encoding of {config.CATEGORICAL}): {names}")

    print(f"\n{'=' * 64}\n  P11 — TARGET ENCODING (seed {SCREEN_SEED})\n{'=' * 64}")
    base_thr, base_f1 = best_threshold_macro_f1(oof(champion_pipe(), X, y, SCREEN_SEED), y)
    print(f"  {ANCHOR + ' (one-hot, champion)':32s} {base_f1:.4f} @thr {base_thr:.3f}")

    screened: dict[str, float] = {}
    for name in names:
        thr, f1 = best_threshold_macro_f1(oof(target_pipe(name, SCREEN_SEED), X, y, SCREEN_SEED), y)
        screened[name] = f1
        print(f"  {name + ' + target-enc':32s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}")

    winners = [n for n in names if screened[n] > base_f1]
    if not winners:
        print(f"\nNothing beat the {ANCHOR} champion at the screen seed — nothing to validate.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {ANCHOR} ---")
    base = np.array([best_threshold_macro_f1(oof(champion_pipe(), X, y, s), y)[1] for s in VALIDATION_SEEDS])
    print(f"  {ANCHOR:32s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")

    results = []
    for name in winners:
        fs = np.array([best_threshold_macro_f1(oof(target_pipe(name, s), X, y, s), y)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(f"  {name + ' + target-enc':32s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")

    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {ANCHOR} ---")
    for dmean, wins, name, _ in results:
        print(f"  {name + ' + target-enc':32s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
