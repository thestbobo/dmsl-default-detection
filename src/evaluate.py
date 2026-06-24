"""Cross-validation and metric reporting for the candidate models.

This module produces the numbers and figures that go into the report. It is
*not* imported by main.py (CV is kept out of the 150 s submission path).

Primary metric is macro-F1; we also report per-class precision/recall/F1,
ROC-AUC, and a saved confusion-matrix figure per model.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved, not shown

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from . import config
from .models import get_candidate_models

CLASS_NAMES = ["no default (0)", "default (1)"]


def _make_cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED
    )


def _save_confusion_matrix(name: str, cm: np.ndarray) -> str:
    """Save a confusion-matrix heatmap to outputs/figures/cm_<name>.png."""
    config.OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUT_FIG_DIR / f"cm_{name}.png"

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def cross_validate_model(name: str, pipeline, X, y, save_fig: bool = True) -> dict:
    """Run StratifiedKFold CV for one model and return a metrics dict."""
    cv = _make_cv()

    # Per-fold macro-F1 (primary metric) -> mean +/- std.
    f1_folds = cross_val_score(pipeline, X, y, scoring="f1_macro", cv=cv, n_jobs=None)

    # Out-of-fold probabilities for ROC-AUC; labels via the 0.5 threshold.
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, proba)
    report = classification_report(
        y, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, y_pred)

    fig_path = _save_confusion_matrix(name, cm) if save_fig else None

    return {
        "name": name,
        "macro_f1_mean": float(f1_folds.mean()),
        "macro_f1_std": float(f1_folds.std()),
        "macro_f1_folds": f1_folds.tolist(),
        "roc_auc": float(roc_auc),
        "report": report,
        "confusion_matrix": cm,
        "figure": fig_path,
    }


def _best_macro_f1_threshold(proba: np.ndarray, y) -> tuple[float, float]:
    """Threshold on the config grid that maximises macro-F1 on OOF probabilities.

    This is the *deployed objective*: main.py submits behind
    ``TunedThresholdClassifierCV(scoring="f1_macro")``, so the honest report metrics
    are taken at the macro-F1-optimal cut, not the arbitrary 0.5 one.
    """
    f1s = [f1_score(y, (proba >= t).astype(int), average="macro") for t in config.THRESHOLDS]
    i = int(np.argmax(f1s))
    return float(config.THRESHOLDS[i]), float(f1s[i])


def cross_validate_tuned(name: str, pipeline, X, y, save_fig: bool = True) -> dict:
    """Like :func:`cross_validate_model` but at the tuned (deployed) threshold.

    Computes OOF probabilities once, picks the macro-F1-optimal threshold, and reports
    the confusion matrix + per-class precision/recall/F1 *there* — so the figure and
    metrics match the operating point main.py actually submits at. ROC-AUC is
    threshold-independent. The figure is saved as ``cm_<name>_tuned.png`` (the 0.5-cut
    ``cm_<name>.png`` from :func:`cross_validate_model` is left untouched).
    """
    cv = _make_cv()
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    threshold, macro_f1 = _best_macro_f1_threshold(proba, y)
    y_pred = (proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y, proba)
    report = classification_report(
        y, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, y_pred)
    fig_path = _save_confusion_matrix(f"{name}_tuned", cm) if save_fig else None

    return {
        "name": name,
        "threshold": threshold,
        "macro_f1": macro_f1,
        "roc_auc": float(roc_auc),
        "report": report,
        "confusion_matrix": cm,
        "figure": fig_path,
    }


def _print_tuned_report(res: dict) -> None:
    rep = res["report"]
    print(f"\n=== {res['name']} (tuned threshold {res['threshold']:.3f}) ===")
    print(f"  macro-F1 : {res['macro_f1']:.4f}  (deployed objective, OOF)")
    print(f"  ROC-AUC  : {res['roc_auc']:.4f}")
    for cls in CLASS_NAMES:
        m = rep[cls]
        print(
            f"  {cls:>16}: precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} f1={m['f1-score']:.3f}"
        )
    if res["figure"]:
        print(f"  confusion matrix saved -> {res['figure']}")


def compare_all_tuned(X, y, models: dict | None = None) -> dict[str, dict]:
    """Deployed-objective (tuned-threshold) comparison of the candidate baselines.

    Apples-to-apples counterpart to :func:`compare_all`: every model is scored at its
    own macro-F1-optimal threshold, which is the fair comparison given the deployed
    threshold tuner. Produces the baseline numbers + confusion-matrix figure used in
    the report.
    """
    models = models or get_candidate_models()
    results = {name: cross_validate_tuned(name, pipe, X, y) for name, pipe in models.items()}

    for res in results.values():
        _print_tuned_report(res)

    print("\n--- Macro-F1 ranking (tuned threshold, deployed objective) ---")
    ranked = sorted(results.values(), key=lambda r: r["macro_f1"], reverse=True)
    for res in ranked:
        print(f"  {res['macro_f1']:.4f} @thr {res['threshold']:.3f}  {res['name']}")

    return results


def _print_model_report(res: dict) -> None:
    rep = res["report"]
    print(f"\n=== {res['name']} ===")
    print(
        f"  macro-F1 : {res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f} "
        f"(5-fold)"
    )
    print(f"  ROC-AUC  : {res['roc_auc']:.4f}")
    for cls in CLASS_NAMES:
        m = rep[cls]
        print(
            f"  {cls:>16}: precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} f1={m['f1-score']:.3f}"
        )
    if res["figure"]:
        print(f"  confusion matrix saved -> {res['figure']}")


def compare_all(X, y, models: dict | None = None) -> dict[str, dict]:
    """Cross-validate every candidate and print a ranked macro-F1 summary."""
    models = models or get_candidate_models()
    results = {name: cross_validate_model(name, pipe, X, y) for name, pipe in models.items()}

    for res in results.values():
        _print_model_report(res)

    print("\n--- Macro-F1 ranking (5-fold mean) ---")
    ranked = sorted(results.values(), key=lambda r: r["macro_f1_mean"], reverse=True)
    for res in ranked:
        print(f"  {res['macro_f1_mean']:.4f}  {res['name']}")

    return results
