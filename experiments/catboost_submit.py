"""Path P14 / external-review item 3 — CatBoost on the FE winners (NEW DEP, gated).

CatBoost is the review's recommended next learner specifically because our failure mode on
this LB is *overfitting* (tuned HGB tanked to 0.704-0.710): CatBoost's ordered boosting is
built to resist exactly that. We run it LIGHTLY tuned on the deployed feature sets, both
standalone and — the more promising route — as a *decorrelated ensemble member* with the
rf_balanced champion (L8). Pinned to seed 888 (best LB seed) for comparability with the
0.720 champion.

CatBoost is installed LOCALLY only (`pip install catboost`) and is **NOT** in
requirements.txt — gated on a real LB gain over 0.720, exactly as LightGBM was (P7/L1).

    python experiments/catboost_submit.py            # screen + submit
    python experiments/catboost_submit.py --no-submit # screen only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import VotingClassifier  # noqa: E402
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

REF_SEED = 888
config.SEED = REF_SEED  # read at call time everywhere -> propagates

NOSCALE = config.ENCODING_CONFIGS["noscale"]
WIN = config.FEATURE_CONFIGS["paysem_util_payratio"]   # deployed winner (LB 0.719/0.720)
STRESS = config.FEATURE_CONFIGS["paysem_stress_payratio"]  # the 0.720 stress combo

# Lightly-tuned, regularised CatBoost specs (overfitting is the risk, so keep depth modest
# and l2 high). auto_class_weights mirrors the rf_balanced direction (L9).
CB_SPECS = {
    "cb_default": {"iterations": 500, "depth": 6, "auto_class_weights": "Balanced"},
    "cb_reg":     {"iterations": 800, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 6,
                   "auto_class_weights": "Balanced"},
    "cb_deep":    {"iterations": 600, "depth": 8, "l2_leaf_reg": 3,
                   "auto_class_weights": "Balanced"},
}


def _cb_pipe(params, groups):
    return make_pipeline(make_estimator("catboost", params), feature_groups=groups, encoding=NOSCALE)


def _rf_pipe(groups):
    rf = config.MODEL_CONFIGS["rf_balanced"]["params"]
    return make_pipeline(make_estimator("rf", rf), feature_groups=groups, encoding=NOSCALE)


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
    print(f"dev rows={len(y):,}  seed={config.SEED}  winner feats={WIN}")

    # Screen standalone CatBoost on the winning feature set.
    print("\n--- Screen: CatBoost OOF macro-F1 @best thr (seed 888, on paysem_util_payratio) ---")
    screened = {}
    for name, params in CB_SPECS.items():
        thr, f1 = best_threshold_macro_f1(oof_proba(_cb_pipe(params, WIN), X, y, config.SEED), y)
        screened[name] = f1
        print(f"  {name:12s} {f1:.4f} @thr {thr:.3f}  {params}")

    if not submit:
        return

    # (a) standalone CatBoost on the winning feature set
    print("\n  emitting standalone CatBoost (paysem_util_payratio):")
    for name, params in CB_SPECS.items():
        model = _tuned(_cb_pipe(params, WIN))
        model.fit(X, y)
        preds = model.predict(X_eval)
        emit_submission(eval_ids, eval_df, preds,
                        description=f"CatBoost {name} + paysem_util_payratio (seed 888): {params}",
                        oof_f1=screened[name])

    # (b) best CatBoost spec on the stress feature set (the other 0.720 combo)
    best_cb = max(screened, key=screened.get)
    print(f"\n  emitting CatBoost {best_cb} on paysem_stress_payratio:")
    model = _tuned(_cb_pipe(CB_SPECS[best_cb], STRESS))
    model.fit(X, y)
    preds = model.predict(X_eval)
    emit_submission(eval_ids, eval_df, preds,
                    description=f"CatBoost {best_cb} + paysem_stress_payratio (seed 888)")

    # (c) decorrelated ensemble: CatBoost + rf_balanced champion (soft-vote), the L8 route
    print(f"\n  emitting soft-vote [CatBoost {best_cb} + rf_balanced] on paysem_util_payratio:")
    ens = VotingClassifier(
        estimators=[("catboost", _cb_pipe(CB_SPECS[best_cb], WIN)), ("rf", _rf_pipe(WIN))],
        voting="soft", n_jobs=-1,
    )
    model = _tuned(ens)
    model.fit(X, y)
    preds = model.predict(X_eval)
    emit_submission(eval_ids, eval_df, preds,
                    description=f"soft-vote [CatBoost {best_cb} + rf_balanced] on paysem_util_payratio (seed 888)")


if __name__ == "__main__":
    main()
