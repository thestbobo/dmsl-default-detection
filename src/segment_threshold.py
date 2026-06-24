"""Group-aware (segmented) decision thresholds — Path P13.

Everywhere else we pick a single global cut on ``predict_proba``. But the dataset's
subgroups have very different default base rates, and macro-F1 is most sensitive exactly
where the minority class is mis-fired. SOTA on this exact dataset (Gittlin, ECML-PKDD
2025) reports +1.5–4% balanced accuracy from per-subgroup thresholds, often making
synthetic augmentation redundant — and our own work already shows threshold tuning is the
one free, transferable win (L4) while resampling lost (P10).

This module assigns each row to a low-cardinality, high-signal segment and tunes one
threshold per segment. Crucially the per-segment cuts are tuned by **coordinate ascent on
the GLOBAL macro-F1**, not by maximising each segment in isolation: a row's contribution
to the global confusion matrix is additive, but macro-F1 is non-linear in those global
counts, so the segment cuts are coupled and must be optimised jointly.

Regularisation against over-fitting a tiny segment: segments with support below
``min_support`` are not given their own cut — they fall back to the global threshold.

The tuning is a stateless function of (OOF proba, y, segment labels), so it stays
leakage-free when fed out-of-fold probabilities, exactly like the existing threshold
tuners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config

# Segmentation schemes: name -> callable(X) -> pd.Series of segment labels (one per row).
# Each is low-cardinality and high-signal, so per-segment cuts have enough support.


def _pay0_buckets(X: pd.DataFrame) -> pd.Series:
    """PAY_0 (most-recent repayment status), the dominant predictor: on-time / 1mo / 2mo+."""
    p = X["PAY_0"]
    return pd.Series(
        np.where(p <= 0, "ontime", np.where(p == 1, "late1", "late2plus")),
        index=X.index,
        name="pay0_bucket",
    )


def _sex(X: pd.DataFrame) -> pd.Series:
    return X["SEX"].astype(int).map({1: "male", 2: "female"}).fillna("other").rename("sex")


def _limit_terciles(X: pd.DataFrame) -> pd.Series:
    """LIMIT_BAL terciles (low / mid / high credit line)."""
    return pd.qcut(X["LIMIT_BAL"], q=3, labels=["low", "mid", "high"], duplicates="drop").astype(
        str
    ).rename("limit_tercile")


SEGMENTERS = {
    "pay0": _pay0_buckets,
    "sex": _sex,
    "limit3": _limit_terciles,
}


def make_segments(X: pd.DataFrame, kind: str) -> np.ndarray:
    """Return an array of segment labels for each row of ``X`` under scheme ``kind``."""
    if kind not in SEGMENTERS:
        raise ValueError(f"unknown segmentation {kind!r}; known: {list(SEGMENTERS)}")
    return SEGMENTERS[kind](X).to_numpy()


def _macro_f1(y, proba, seg, thresholds: dict, global_thr: float) -> float:
    return f1_score(y, apply_segment_thresholds(proba, seg, thresholds, global_thr), average="macro")


def apply_segment_thresholds(
    proba: np.ndarray, seg: np.ndarray, thresholds: dict, global_thr: float
) -> np.ndarray:
    """Binarise ``proba`` using each row's segment threshold (global cut as fallback)."""
    cuts = np.array([thresholds.get(s, global_thr) for s in seg], dtype=float)
    return (proba >= cuts).astype(int)


def tune_segment_thresholds(
    proba: np.ndarray,
    y,
    seg: np.ndarray,
    global_thr: float,
    min_support: int = 500,
    max_passes: int = 5,
) -> dict:
    """Per-segment thresholds maximising GLOBAL macro-F1 via coordinate ascent.

    Starts every segment at ``global_thr`` (so the result can only match-or-beat the
    single-cut baseline on the tuning data), then repeatedly sweeps each high-support
    segment over the project threshold grid, keeping whichever cut most improves the
    *global* macro-F1, until a full pass makes no change. Segments below ``min_support``
    keep the global cut.
    """
    yv = np.asarray(y)
    grid = config.THRESHOLDS
    labels, counts = np.unique(seg, return_counts=True)
    tunable = [s for s, c in zip(labels, counts) if c >= min_support]
    thresholds = {s: float(global_thr) for s in tunable}

    best_f1 = _macro_f1(yv, proba, seg, thresholds, global_thr)
    for _ in range(max_passes):
        changed = False
        for s in tunable:
            cur = thresholds[s]
            best_t = cur
            for t in grid:
                thresholds[s] = float(t)
                f1 = _macro_f1(yv, proba, seg, thresholds, global_thr)
                if f1 > best_f1 + 1e-12:
                    best_f1, best_t = f1, float(t)
            thresholds[s] = best_t
            if best_t != cur:
                changed = True
        if not changed:
            break
    return thresholds
