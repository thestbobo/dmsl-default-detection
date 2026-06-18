"""Example tuning script — kept OUT of main.py (exam rule).

Run this to (a) compare the three baselines with full 5-fold metrics and saved
confusion-matrix figures, and (b) run a small grid search over the
HistGradientBoosting hyper-parameters, optimising macro-F1.

Copy the printed best parameters into ``src/models.CHOSEN_HGB_PARAMS`` so that
``main.py`` trains the tuned configuration.

Usage:
    python experiments/tune_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.evaluate import compare_all  # noqa: E402
from src.models import make_pipeline  # noqa: E402


def tune_hgb(X, y) -> dict:
    """Small grid search over HGB params, scored by macro-F1."""
    pipe = make_pipeline(HistGradientBoostingClassifier(random_state=config.SEED))
    param_grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [31, 63],
        "model__l2_regularization": [0.0, 1.0],
    }
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(
        pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X, y)

    # Strip the "model__" pipeline prefix so params can be pasted into models.py.
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print("\n--- HGB grid search (macro-F1) ---")
    print(f"  best macro-F1 : {search.best_score_:.4f}")
    print(f"  best params   : {best}")
    print("  -> paste into src/models.CHOSEN_HGB_PARAMS")
    return best


def main() -> None:
    config.set_seed()
    dev = load_development()
    X, y = split_xy(dev)

    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    compare_all(X, y)
    tune_hgb(X, y)


if __name__ == "__main__":
    main()
