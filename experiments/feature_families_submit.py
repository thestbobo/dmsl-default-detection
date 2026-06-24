"""Engineered feature families on the rf_balanced champion, kept out of main.py.

Feature engineering on the random forest is the lever that transfers to the leaderboard.
Each family (paysem, coverx, and stacks on payratio) is screened by OOF macro-F1 on
rf_balanced, then the full-dev-fit model is emitted as a numbered submission regardless
of the CV verdict; the leaderboard judges.

    python experiments/feature_families_submit.py            # screen + submit
    python experiments/feature_families_submit.py --no-submit # screen only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
RF_PARAMS = config.MODEL_CONFIGS["rf_balanced"]["params"]

# Feature-config names (from config.yaml experiments.feature_configs) to screen + submit.
# Override on the CLI: 'python experiments/feature_families_submit.py paysem_util ...'.
DEFAULT_CANDIDATES = [
    "paysem",
    "paysem_payratio",
    "coverx",
    "coverx_payratio",
    "paysem_coverx_payratio",
    "paysem_util_payratio",
]


def _pipe(groups):
    return make_pipeline(make_estimator("rf", RF_PARAMS), feature_groups=groups, encoding=NOSCALE)


def _tuned(estimator):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=estimator, scoring="f1_macro", cv=cv, refit=True, random_state=config.SEED
    )


def main() -> None:
    argv = sys.argv[1:]
    if "--seed" in argv:
        # Pin a specific seed regardless of config.yaml, so candidates are comparable.
        i = argv.index("--seed")
        config.SEED = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    config.set_seed()
    args = [a for a in argv if not a.startswith("--")]
    submit = "--no-submit" not in argv
    candidates = args or DEFAULT_CANDIDATES
    unknown = [n for n in candidates if n not in config.FEATURE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown feature config(s) {unknown}; known: {list(config.FEATURE_CONFIGS)}")

    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  base default rate={y.mean():.1%}")

    # Anchor: champion with no engineered features (RF parallelises -> CV sequential).
    _, base_f1 = best_threshold_macro_f1(oof_proba(_pipe([]), X, y, config.SEED, cv_n_jobs=1), y)
    print(f"\n--- Screen: rf_balanced OOF macro-F1 @best threshold (seed {config.SEED}) ---")
    print(f"  {'rf_balanced (raw)':28s} {base_f1:.4f}   (anchor)")

    screened = {}
    for name in candidates:
        groups = config.FEATURE_CONFIGS[name]
        thr, f1 = best_threshold_macro_f1(oof_proba(_pipe(groups), X, y, config.SEED, cv_n_jobs=1), y)
        screened[name] = (f1, groups)
        print(f"  {name:28s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}   {groups}")

    if not submit:
        return

    print("\n  emitting candidates as submissions (LB is the judge):")
    for name in candidates:
        f1, groups = screened[name]
        model = _tuned(_pipe(groups))
        model.fit(X, y)
        preds = model.predict(X_eval)
        emit_submission(
            eval_ids, eval_df, preds,
            description=f"rf_balanced + {name} features {groups}",
            oof_f1=f1,
        )


if __name__ == "__main__":
    main()
