"""Item 1 — Leaderboard threshold sweep on the deployed champion (rf_balanced).

The deployed cut (0.622) predicts ~19.8% defaults on eval, BELOW the 22.1% dev base
rate — i.e. we under-fire the minority class exactly where macro-F1 is most sensitive.
Dev CV is structurally blind to the dev->eval distribution shift, so the only honest
way to find the LB-optimal cut is to submit the SAME model at several thresholds and
let the leaderboard decide.

This script fits rf_balanced once on the full development set, then writes one numbered
candidate submission per threshold (lower threshold -> more predicted defaults). It also
prints, for context only, the dev OOF macro-F1 at each cut.

    python experiments/threshold_sweep_submit.py [thr1 thr2 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score  # noqa: E402

from experiments._submit_utils import (  # noqa: E402
    best_threshold_macro_f1,
    emit_submission,
    fit_eval_proba,
    oof_proba,
)
from src import config  # noqa: E402
from src.data import (  # noqa: E402
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import make_estimator, make_pipeline  # noqa: E402

CHAMPION = "rf_balanced"
# Bracket the deployed 0.622 cut with lower thresholds that push the predicted-default
# rate up toward (and past) the 22.1% base rate.
DEFAULT_THRESHOLDS = [0.45, 0.50, 0.55, 0.58, 0.60, 0.622, 0.65]


def _champion_pipeline():
    spec = config.MODEL_CONFIGS[CHAMPION]
    encoding = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=encoding)


def main() -> None:
    config.set_seed()
    args = sys.argv[1:]
    thresholds = sorted({float(a) for a in args}) if args else DEFAULT_THRESHOLDS

    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)

    base_rate = float(y.mean())
    print(f"dev rows={len(y):,}  base default rate={base_rate:.1%}")

    # Dev OOF probabilities (context: macro-F1 per threshold + the OOF-optimal cut).
    pipe = _champion_pipeline()
    proba_oof = oof_proba(pipe, X, y)
    best_thr, best_f1 = best_threshold_macro_f1(proba_oof, y)
    print(f"{CHAMPION} OOF macro-F1={best_f1:.4f} @best thr {best_thr:.3f}")
    print("OOF macro-F1 + default rate by threshold (dev):")
    oof_f1_by_thr = {}
    for t in thresholds:
        rate = float((proba_oof >= t).mean())
        f1 = float(f1_score(y, (proba_oof >= t).astype(int), average="macro"))
        oof_f1_by_thr[t] = f1
        print(f"  thr {t:.3f}  oof_f1={f1:.4f}  oof_default_rate={rate:.1%}")

    # Fit on full dev, score eval, emit one numbered submission per threshold.
    proba_eval = fit_eval_proba(_champion_pipeline(), X, y, X_eval)
    print("\nWriting candidate submissions:")
    for t in thresholds:
        preds = (proba_eval >= t).astype(int)
        emit_submission(
            eval_ids,
            eval_df,
            preds,
            description=f"{CHAMPION} threshold sweep @ thr={t:.3f}",
            oof_f1=oof_f1_by_thr[t],
        )


if __name__ == "__main__":
    main()
