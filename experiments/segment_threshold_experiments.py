"""Item 4 / Path P13 — Group-aware (segmented) decision thresholds.

We only ever tune ONE global cut, yet the deployed champion under-fires the minority
class on eval (19.8% vs 22.1% base rate) and subgroups have very different default rates.
This script tunes a per-segment threshold (coordinate ascent on global macro-F1,
src/segment_threshold.py) and measures it honestly — leave-one-fold-out, the same
transfer protocol as P6 (threshold_experiments.py) — so we know whether the segmented cut
*transfers* before trusting it. Then it emits one numbered submission per segmentation
scheme (segment cuts tuned on dev OOF, applied per-segment on eval).

    python experiments/segment_threshold_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments._submit_utils import emit_submission, fit_eval_proba  # noqa: E402
from experiments.threshold_experiments import best_threshold, fold_oof  # noqa: E402
from src import config  # noqa: E402
from src.data import (  # noqa: E402
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import make_estimator, make_pipeline  # noqa: E402
from src.segment_threshold import (  # noqa: E402
    SEGMENTERS,
    apply_segment_thresholds,
    make_segments,
    tune_segment_thresholds,
)

CHAMPION = "rf_balanced"
VALIDATION_SEEDS = config.VALIDATION_SEEDS


def _champion_pipeline():
    spec = config.MODEL_CONFIGS[CHAMPION]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)


def _honest_transfer(proba, y, fold_id, seg) -> tuple[float, float]:
    """Leave-one-fold-out macro-F1 for GLOBAL vs SEGMENTED cuts (honest, no peeking)."""
    yv = np.asarray(y)
    n_folds = fold_id.max() + 1
    g_pred = np.zeros(len(yv), dtype=int)
    s_pred = np.zeros(len(yv), dtype=int)
    for k in range(n_folds):
        te = fold_id == k
        tr = ~te
        g_thr, _ = best_threshold(proba[tr], yv[tr])
        g_pred[te] = (proba[te] >= g_thr).astype(int)
        seg_thr = tune_segment_thresholds(proba[tr], yv[tr], seg[tr], g_thr)
        s_pred[te] = apply_segment_thresholds(proba[te], seg[te], seg_thr, g_thr)
    return (
        f1_score(yv, g_pred, average="macro"),
        f1_score(yv, s_pred, average="macro"),
    )


def main() -> None:
    config.set_seed()
    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  base default rate={y.mean():.1%}")

    # --- Honest leave-one-fold-out transfer, averaged over fold seeds ---
    print(f"\n=== Honest transfer (leave-one-fold-out) over {len(VALIDATION_SEEDS)} seeds ===")
    schemes = list(SEGMENTERS)
    glob = {s: [] for s in schemes}
    segm = {s: [] for s in schemes}
    for seed in VALIDATION_SEEDS:
        proba, fold_id = fold_oof(config.MODEL_CONFIGS[CHAMPION], X, y, seed)
        for kind in schemes:
            seg = make_segments(X, kind)
            g, s = _honest_transfer(proba, y, fold_id, seg)
            glob[kind].append(g)
            segm[kind].append(s)
    for kind in schemes:
        g = np.array(glob[kind])
        s = np.array(segm[kind])
        d = s - g
        wins = int((d > 0).sum())
        print(
            f"  {kind:8s} GLOBAL={g.mean():.4f}  SEGMENTED={s.mean():.4f}  "
            f"dmean={d.mean():+.4f}  wins {wins}/{len(VALIDATION_SEEDS)}"
        )

    # --- Emit one submission per scheme (cuts tuned on full-dev OOF, applied to eval) ---
    print("\n=== Emitting segmented-threshold submissions ===")
    proba_oof, _ = fold_oof(config.MODEL_CONFIGS[CHAMPION], X, y, config.SEED)
    g_thr, g_f1 = best_threshold(proba_oof, y)
    print(f"global OOF cut={g_thr:.3f}  OOF macro-F1={g_f1:.4f}")
    proba_eval = fit_eval_proba(_champion_pipeline(), X, y, X_eval)

    for kind in schemes:
        seg_dev = make_segments(X, kind)
        seg_eval = make_segments(X_eval, kind)
        seg_thr = tune_segment_thresholds(proba_oof, y, seg_dev, g_thr)
        oof_pred = apply_segment_thresholds(proba_oof, seg_dev, seg_thr, g_thr)
        oof_f1 = float(f1_score(y, oof_pred, average="macro"))
        cuts_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(seg_thr.items()))
        print(f"  {kind:8s} cuts [{cuts_str}]  (fallback {g_thr:.3f})  OOF macro-F1={oof_f1:.4f}")
        preds = apply_segment_thresholds(proba_eval, seg_eval, seg_thr, g_thr)
        emit_submission(
            eval_ids, eval_df, preds,
            description=f"{CHAMPION} segmented threshold by {kind} [{cuts_str}]",
            oof_f1=oof_f1,
        )


if __name__ == "__main__":
    main()
