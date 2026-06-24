"""Cross-validation and metric reporting for the candidate models.

Produces the report's numbers and figures (macro-F1, per-class precision/recall/F1,
ROC-AUC, confusion-matrix figures). Not imported by main.py, CV stays out of the 150 s
submission path.
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
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def cross_validate_model(name: str, pipeline, X, y, save_fig: bool = True) -> dict:
    """StratifiedKFold CV for one model at the 0.5 threshold; returns a metrics dict."""
    cv = _make_cv()
    f1_folds = cross_val_score(pipeline, X, y, scoring="f1_macro", cv=cv, n_jobs=None)
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
    """Grid threshold that maximises macro-F1 on OOF probabilities (the deployed cut)."""
    f1s = [f1_score(y, (proba >= t).astype(int), average="macro") for t in config.THRESHOLDS]
    i = int(np.argmax(f1s))
    return float(config.THRESHOLDS[i]), float(f1s[i])


def cross_validate_tuned(name: str, pipeline, X, y, save_fig: bool = True) -> dict:
    """Like cross_validate_model but at the macro-F1 optimal threshold (the deployed cut).

    Reports the confusion matrix + per-class metrics there, matching what main.py submits.
    The figure is saved as cm_<name>_tuned.png.
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


def _print_report(res: dict, header: str, f1_line: str) -> None:
    rep = res["report"]
    print(f"\n=== {header} ===")
    print(f"  macro-F1 : {f1_line}")
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
    """Tuned-threshold comparison of the candidates, the fair view given the deployed
    threshold tuner. Produces the report's baseline + final-model numbers and figures."""
    models = models or get_candidate_models()
    results = {name: cross_validate_tuned(name, pipe, X, y) for name, pipe in models.items()}

    for res in results.values():
        _print_report(
            res,
            f"{res['name']} (tuned threshold {res['threshold']:.3f})",
            f"{res['macro_f1']:.4f}  (deployed objective, OOF)",
        )

    print("\n--- Macro-F1 ranking (tuned threshold, deployed objective) ---")
    ranked = sorted(results.values(), key=lambda r: r["macro_f1"], reverse=True)
    for res in ranked:
        print(f"  {res['macro_f1']:.4f} @thr {res['threshold']:.3f}  {res['name']}")

    return results


def compare_all(X, y, models: dict | None = None) -> dict[str, dict]:
    """Cross-validate every candidate at 0.5 and print a ranked macro-F1 summary."""
    models = models or get_candidate_models()
    results = {name: cross_validate_model(name, pipe, X, y) for name, pipe in models.items()}

    for res in results.values():
        _print_report(
            res,
            res["name"],
            f"{res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f} (5-fold)",
        )

    print("\n--- Macro-F1 ranking (5-fold mean) ---")
    ranked = sorted(results.values(), key=lambda r: r["macro_f1_mean"], reverse=True)
    for res in ranked:
        print(f"  {res['macro_f1_mean']:.4f}  {res['name']}")

    return results
