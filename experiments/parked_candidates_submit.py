"""Item 3 — Submit the parked, CV-positive candidates that were never uploaded.

Several candidates screened CV-positive (or were built around the weaker HGB anchor
before rf_balanced became champion) but were PARKed on the "dev-CV <0.005 is noise"
rule and never sent to the leaderboard. With >180 free slots and a CV that under-reads
the LB, the right move is to upload them and let the LB decide (review, 2026-06-21).

Each candidate is wrapped in the SAME TunedThresholdClassifierCV main.py uses (macro-F1,
inner CV, leakage-safe), fit on the full development set, and written as a numbered
candidate submission at its tuned threshold.

    python experiments/parked_candidates_submit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV  # noqa: E402

from experiments._submit_utils import emit_submission, oof_proba, best_threshold_macro_f1  # noqa: E402
from experiments.stacking_experiments import make_stacking  # noqa: E402
from src import config  # noqa: E402
from src.data import (  # noqa: E402
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import build_chosen_ensemble, make_estimator, make_pipeline  # noqa: E402


def _imbalance_pipeline(name: str):
    """Build an IMBALANCE_CONFIGS candidate as a full pipeline (mirrors _merged_spec)."""
    cand = config.IMBALANCE_CONFIGS[name]
    base = config.MODEL_CONFIGS[cand["base_model"]]
    params = {**(base.get("params") or {}), **(cand.get("params") or {})}
    enc = config.ENCODING_CONFIGS[base.get("encoding", "baseline")]
    return make_pipeline(make_estimator(base["kind"], params), encoding=enc)


def _tuned(estimator):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=estimator, scoring="f1_macro", cv=cv, refit=True, random_state=config.SEED
    )


def _candidates():
    """(description, base_estimator) for each parked candidate."""
    return [
        (
            "soft-vote [rf_balanced + logreg_clean + et] (P4 rebuilt on champion)",
            build_chosen_ensemble(config.ENSEMBLE_CONFIGS["rfbal_logreg_et"]),
        ),
        (
            "stack [rf_balanced, hgb, logreg_clean] -> logreg (P8)",
            make_stacking(config.STACKING_CONFIGS["stack_rfbal_hgb_logreg"], config.SEED),
        ),
        (
            "stack [hgb, logreg_clean, rf] -> logreg (P8)",
            make_stacking(config.STACKING_CONFIGS["stack_hgb_logreg_rf"], config.SEED),
        ),
        (
            "rf_balanced_subsample (P5 tie-break)",
            _imbalance_pipeline("rf_balanced_subsample"),
        ),
    ]


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
        # Context: OOF macro-F1 at best threshold (same protocol as the experiments).
        oof_thr, oof_f1 = best_threshold_macro_f1(oof_proba(base_est, X, y), y)
        print(f"   OOF macro-F1={oof_f1:.4f} @best thr {oof_thr:.3f}")
        model = _tuned(base_est)
        model.fit(X, y)
        preds = model.predict(X_eval)
        print(f"   tuned threshold={model.best_threshold_:.4f}")
        emit_submission(eval_ids, eval_df, preds, description=desc, oof_f1=oof_f1)
        print()


if __name__ == "__main__":
    main()
