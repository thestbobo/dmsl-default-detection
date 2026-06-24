"""Items 5 & 6 — extra candidates around the champion, never tried before.

Item 6: ExtraTrees + balanced class weights (``et_balanced``). P5 only weighted HGB and
RF; ET is more decorrelated than RF (L8), so it is both a standalone candidate and a
future ensemble member.

Item 5: feature engineering on RF specifically. L5/L7 ("the tree already has it") were
established on HGB, whose histogram binning absorbs monotone transforms; RF's axis-aligned
splits on the highly collinear BILL_AMT block can benefit from explicit ratios. So re-test
util / payratio / pay on the rf_balanced champion (these were only ever tried on HGB).

Each candidate is wrapped in main.py's TunedThresholdClassifierCV, fit on full dev, and
emitted as a numbered submission. The LB decides (review 2026-06-21).

    python experiments/extra_candidates_submit.py
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
FEATURE_GROUPS = ["util", "payratio", "pay", "util_payratio"]


def _tuned(estimator):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=estimator, scoring="f1_macro", cv=cv, refit=True, random_state=config.SEED
    )


def _candidates():
    out = []
    # Item 6: ExtraTrees-balanced (from model_configs).
    et_spec = config.MODEL_CONFIGS["et_balanced"]
    out.append((
        "et_balanced (ExtraTrees + balanced weights)",
        make_pipeline(
            make_estimator(et_spec["kind"], et_spec.get("params")),
            encoding=config.ENCODING_CONFIGS[et_spec.get("encoding", "baseline")],
        ),
    ))
    # Item 5: rf_balanced + each engineered feature group.
    for name in FEATURE_GROUPS:
        groups = config.FEATURE_CONFIGS[name]
        out.append((
            f"rf_balanced + {name} features {groups}",
            make_pipeline(make_estimator("rf", RF_PARAMS), feature_groups=groups, encoding=NOSCALE),
        ))
    return out


def main() -> None:
    config.set_seed()
    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  base default rate={y.mean():.1%}\n")

    for desc, base_est in _candidates():
        print(f"== {desc}")
        # All candidates are tree ensembles with n_jobs=-1 -> sequential CV folds.
        oof_thr, oof_f1 = best_threshold_macro_f1(oof_proba(base_est, X, y, cv_n_jobs=1), y)
        print(f"   OOF macro-F1={oof_f1:.4f} @best thr {oof_thr:.3f}")
        model = _tuned(base_est)
        model.fit(X, y)
        preds = model.predict(X_eval)
        print(f"   tuned threshold={model.best_threshold_:.4f}")
        emit_submission(eval_ids, eval_df, preds, description=desc, oof_f1=oof_f1)
        print()


if __name__ == "__main__":
    main()
