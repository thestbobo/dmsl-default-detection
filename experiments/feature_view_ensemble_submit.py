"""Path P14 follow-up — ensemble models trained on DIFFERENT feature views.

The same-feature ensembles (E29, subs 47-50) plateaued at 0.717-0.719 because the members
share the same inputs and err similarly. But two of our configs reached 0.720 on *different*
feature sets — `paysem_util_payratio` and `paysem_stress_payratio` — so an RF on each view
errs on genuinely different signal (maximal decorrelation, L8). Soft-voting those views (and
mixing a CatBoost view) is the most promising remaining ensemble route toward >=0.722.

Pinned to seed 888 (best LB seed). LB judges (L1/L13).

    python experiments/feature_view_ensemble_submit.py [--no-submit]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import VotingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV  # noqa: E402

from experiments._submit_utils import emit_submission  # noqa: E402
from src import config  # noqa: E402
from src.data import (  # noqa: E402
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import make_estimator, make_pipeline  # noqa: E402

config.SEED = 888  # propagates to make_estimator / CV / tuner (read at call time)
NOSCALE = config.ENCODING_CONFIGS["noscale"]
RF = config.MODEL_CONFIGS["rf_balanced"]["params"]
CB_REG = {"iterations": 800, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 6,
          "auto_class_weights": "Balanced"}

V = config.FEATURE_CONFIGS  # feature views


def _rf(view):
    return make_pipeline(make_estimator("rf", RF), feature_groups=V[view], encoding=NOSCALE)


def _cb(view):
    return make_pipeline(make_estimator("catboost", CB_REG), feature_groups=V[view], encoding=NOSCALE)


# (label, [(member_name, pipeline), ...])
ENSEMBLES = {
    "rfviews_util_stress": [("rf_util", _rf("paysem_util_payratio")),
                            ("rf_stress", _rf("paysem_stress_payratio"))],
    "rfviews_util_stress_allnew": [("rf_util", _rf("paysem_util_payratio")),
                                   ("rf_stress", _rf("paysem_stress_payratio")),
                                   ("rf_allnew", _rf("all_new"))],
    "rf_util__cb_stress": [("rf_util", _rf("paysem_util_payratio")),
                           ("cb_stress", _cb("paysem_stress_payratio"))],
}


def _tuned(estimator):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=estimator, scoring="f1_macro", cv=cv, refit=True, random_state=config.SEED
    )


def main() -> None:
    config.set_seed()
    submit = "--no-submit" not in sys.argv[1:]

    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  seed={config.SEED}")

    for label, members in ENSEMBLES.items():
        est = VotingClassifier(estimators=members, voting="soft", n_jobs=-1)
        if not submit:
            print(f"  built {label}: {[m for m, _ in members]}")
            continue
        model = _tuned(est)
        model.fit(X, y)
        preds = model.predict(X_eval)
        emit_submission(eval_ids, eval_df, preds,
                        description=f"feature-view soft-vote {label} (seed 888)")


if __name__ == "__main__":
    main()
