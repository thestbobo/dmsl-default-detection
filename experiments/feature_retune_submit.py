"""Path P14 / external-review item 1 — multiply the two working levers + re-tune.

The review's highest-EV move: the best FEATURES (payratio, LB 0.716) and the best
HYPERPARAMETERS (Path-X tuned RF cand2 -> LB 0.715, cand0 -> 0.713) live in SEPARATE
submissions — cand2 was tuned on raw features, payratio rode on default rf_balanced.
This script (a) combines them, and (b) runs a FRESH randomized RF search *on the payratio
feature set*, because the optimum shifts once the features change (Lesson L3).

All candidates are emitted as numbered submissions regardless of CV — the LB judges (L1).

    python experiments/feature_retune_submit.py             # combine + re-tune + submit
    python experiments/feature_retune_submit.py --no-submit # screen only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV  # noqa: E402

from experiments._submit_utils import best_threshold_macro_f1, emit_submission, oof_proba  # noqa: E402
from src import config  # noqa: E402
from src.data import (  # noqa: E402
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import make_estimator, make_pipeline  # noqa: E402

NOSCALE = config.ENCODING_CONFIGS["noscale"]
N_SEARCH = 30
N_SUBMIT_TUNED = 3
# Feature set the fresh RF search runs on. Default = the proven LB-0.716 payratio set
# (review item 1); override with `--groups <feature_config_name>` to re-tune on the
# current LB winner (e.g. paysem_util_payratio -> 0.719). `--retune-only` skips the
# fixed lever-multiplication combos and just runs the search on --groups.
RETUNE_GROUPS = ["payratio"]

# Path-X RF winners (from outputs/submissions/MANIFEST.md): the params that transferred.
CAND2 = {"n_estimators": 800, "max_depth": 32, "max_features": 0.3, "min_samples_leaf": 40,
         "min_samples_split": 2, "max_samples": None, "criterion": "entropy",
         "class_weight": "balanced", "n_jobs": -1}                          # #17 -> LB 0.715
CAND0 = {"n_estimators": 800, "max_depth": 24, "max_features": "sqrt", "min_samples_leaf": 10,
         "min_samples_split": 5, "max_samples": 0.7, "criterion": "gini",
         "class_weight": "balanced", "n_jobs": -1}                          # #15 -> LB 0.713

# (params, feature_groups, label) — the lever-multiplication grid.
COMBOS = [
    (CAND2, ["payratio"], "Path-X RF cand2 + payratio"),
    (CAND2, ["coverx", "payratio"], "Path-X RF cand2 + coverx_payratio"),
    (CAND0, ["payratio"], "Path-X RF cand0 + payratio"),
]


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _sample_rf(rng) -> dict:
    return dict(
        n_estimators=_pick(rng, [300, 500, 800]),
        max_depth=_pick(rng, [None, 10, 16, 24, 32]),
        max_features=_pick(rng, ["sqrt", 0.3, 0.5, 0.7]),
        min_samples_leaf=_pick(rng, [5, 10, 20, 40]),
        min_samples_split=_pick(rng, [2, 5, 10]),
        max_samples=_pick(rng, [None, 0.7, 0.9]),
        criterion=_pick(rng, ["gini", "entropy"]),
        class_weight="balanced",
        n_jobs=-1,
    )


def _pipe(params, groups):
    return make_pipeline(make_estimator("rf", params), feature_groups=groups, encoding=NOSCALE)


def _score(params, groups, X, y, seed=config.SEED):
    return best_threshold_macro_f1(oof_proba(_pipe(params, groups), X, y, seed, cv_n_jobs=1), y)


def _tuned(estimator):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=estimator, scoring="f1_macro", cv=cv, refit=True, random_state=config.SEED
    )


def main() -> None:
    config.set_seed()
    argv = sys.argv[1:]
    submit = "--no-submit" not in argv
    retune_only = "--retune-only" in argv
    global RETUNE_GROUPS
    if "--groups" in argv:
        name = argv[argv.index("--groups") + 1]
        if name not in config.FEATURE_CONFIGS:
            raise SystemExit(f"Unknown feature config {name!r}; known: {list(config.FEATURE_CONFIGS)}")
        RETUNE_GROUPS = config.FEATURE_CONFIGS[name]

    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  base default rate={y.mean():.1%}")

    _, base_f1 = _score(config.MODEL_CONFIGS["rf_balanced"]["params"], RETUNE_GROUPS, X, y)
    print(f"\nanchor: rf_balanced + {RETUNE_GROUPS} OOF={base_f1:.4f}")

    # --- (a) Lever-multiplication combos: tuned params x best features ---
    if not retune_only:
        print("\n--- Combos: Path-X tuned params x engineered features ---")
        for params, groups, label in COMBOS:
            thr, f1 = _score(params, groups, X, y)
            print(f"  {label:36s} {f1:.4f} @thr {thr:.3f}  d={f1 - base_f1:+.4f}")

    # --- (b) Fresh RF search on the chosen feature set ---
    print(f"\n--- RF randomized search ({N_SEARCH} cfg) on feature set {RETUNE_GROUPS} ---")
    rng = np.random.default_rng(config.SEED)
    ranked, seen = [], set()
    while len(ranked) < N_SEARCH:
        p = _sample_rf(rng)
        key = repr(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        thr, f1 = _score(p, RETUNE_GROUPS, X, y)
        ranked.append((f1, thr, p))
    ranked.sort(key=lambda r: -r[0])
    print(f"  top {N_SUBMIT_TUNED} by seed-{config.SEED} OOF macro-F1:")
    for f1, thr, p in ranked[:N_SUBMIT_TUNED]:
        print(f"    {f1:.4f} @thr {thr:.3f}  {p}")

    if not submit:
        return

    if not retune_only:
        print("\n  emitting combos:")
        for params, groups, label in COMBOS:
            _, f1 = _score(params, groups, X, y)
            model = _tuned(_pipe(params, groups))
            model.fit(X, y)
            preds = model.predict(X_eval)
            emit_submission(eval_ids, eval_df, preds, description=f"{label}: {params}", oof_f1=f1)

    print(f"\n  emitting top {N_SUBMIT_TUNED} re-tuned RF (on {RETUNE_GROUPS}):")
    for i, (f1, thr, p) in enumerate(ranked[:N_SUBMIT_TUNED]):
        model = _tuned(_pipe(p, RETUNE_GROUPS))
        model.fit(X, y)
        preds = model.predict(X_eval)
        emit_submission(
            eval_ids, eval_df, preds,
            description=f"RF re-tuned on {RETUNE_GROUPS} cand{i} (OOF {f1:.4f}): {p}",
            oof_f1=f1,
        )


if __name__ == "__main__":
    main()
