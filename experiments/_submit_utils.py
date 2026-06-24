"""Shared helpers for the experiment sweeps and numbered candidate submissions.

Dev CV is a weak predictor of the public leaderboard here, so instead of pre-filtering
on CV we generate several candidates and upload them. Each is written to
outputs/submissions/submission_<N>.csv (never clobbering the deployed submission.csv)
and logged to outputs/submissions/MANIFEST.md.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import validate_submission, write_submission  # noqa: E402

# Mean macro-F1 gain that marks a robust win rather than noise (shared by the sweeps).
MIN_DELTA = 0.005

MANIFEST = config.OUT_SUB_DIR / "MANIFEST.md"
_MANIFEST_HEADER = (
    "# Candidate submission manifest\n\n"
    "Each `submission_<N>.csv` in this folder, newest at the bottom. Upload these and\n"
    "paste the returned leaderboard score into the `LB` column. `submission.csv` (no\n"
    "number) is the deployed model.\n\n"
    "| # | Date | Default rate | OOF macro-F1 | Description | LB |\n"
    "|---|------|--------------|--------------|-------------|----|\n"
)


def best_threshold_macro_f1(proba: np.ndarray, y) -> tuple[float, float]:
    """Best macro-F1 over the project threshold grid (and the threshold achieving it)."""
    f1s = [f1_score(y, (proba >= t).astype(int), average="macro") for t in config.THRESHOLDS]
    i = int(np.argmax(f1s))
    return float(config.THRESHOLDS[i]), float(f1s[i])


def verdict(dmean: float, wins: int, n: int) -> str:
    """Label a candidate from its paired-CV result (wins on every seed + mean gain)."""
    if wins == n and dmean >= MIN_DELTA:
        return "KEEP (robust)"
    if wins == n and dmean > 0:
        return "keep? (robust but small)"
    if dmean > 0:
        return "watch (not all seeds)"
    return "revert"


def oof_proba(pipeline, X, y, seed: int = config.SEED, cv_n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold positive-class probabilities (deployed CV protocol).

    Set cv_n_jobs=1 when the estimator already parallelises (RF/ET with n_jobs=-1), so
    a parallel estimator inside a parallel CV does not oversubscribe the cores.
    """
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=cv_n_jobs
    )[:, 1]


def fit_eval_proba(model, X, y, X_eval) -> np.ndarray:
    """Fit the model on the full development set, return eval positive-class proba."""
    model.fit(X, y)
    return model.predict_proba(X_eval)[:, 1]


def next_submission_number() -> int:
    """Next free N for submission_<N>.csv in the submissions folder."""
    nums = []
    for p in config.OUT_SUB_DIR.glob("submission_*.csv"):
        stem = p.stem.split("_", 1)[1]
        if stem.isdigit():
            nums.append(int(stem))
    return (max(nums) + 1) if nums else 1


def _append_manifest(n: int, default_rate: float, oof_f1, description: str) -> None:
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(_MANIFEST_HEADER)
    f1_str = f"{oof_f1:.4f}" if oof_f1 is not None else "-"
    row = (
        f"| {n} | {date.today().isoformat()} | {default_rate:.1%} | {f1_str} | "
        f"{description} |  |\n"
    )
    with MANIFEST.open("a") as fh:
        fh.write(row)


def emit_submission(eval_ids, eval_df, preds, description: str, oof_f1=None) -> int:
    """Write the next numbered candidate submission + validate + log it. Returns N."""
    preds = np.asarray(preds).astype(int)
    n = next_submission_number()
    path = config.OUT_SUB_DIR / f"submission_{n}.csv"
    write_submission(eval_ids, preds, path=path)
    validate_submission(path, eval_df)
    default_rate = float(preds.mean())
    _append_manifest(n, default_rate, oof_f1, description)
    print(
        f"  -> submission_{n}.csv  defaults={preds.sum()}/{len(preds)} "
        f"({default_rate:.1%})  {description}"
    )
    return n
