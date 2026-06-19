"""Model factory.

- ``make_pipeline`` glues the cleaning + preprocessing in front of any estimator.
- ``get_candidate_models`` returns the three baselines compared in evaluate.py.
- ``build_chosen_pipeline`` returns the *single* model that main.py trains for
  the submission: the HistGradientBoosting pipeline wrapped in a
  threshold tuner that optimises macro-F1 (leakage-safe, via inner CV).

To change the "best" config after tuning in experiments/, edit only
``CHOSEN_HGB_PARAMS`` / ``build_chosen_pipeline`` here.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor, make_code_folder

# Best HGB hyper-parameters found via experiments/tune_baseline.py.
#
# Currently EMPTY (library defaults) on purpose. A regularised config tuned to the
# deployment objective looked +0.0035 macro-F1 better in nested CV but scored WORSE
# on the actual leaderboard (0.708 vs 0.712 for defaults). The dev-CV gain did not
# transfer to the evaluation set, so we trust the leaderboard and keep defaults as
# the anchor. Lesson: on this data, sub-0.005 CV gains are noise — don't ship a
# more-complex, lower-scoring config to chase them. Re-tune AFTER changing features
# (params for raw features won't be optimal for an engineered feature set anyway).
CHOSEN_HGB_PARAMS: dict = {}


def make_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with code-folding + preprocessing."""
    return Pipeline(
        steps=[
            ("fold", make_code_folder()),
            ("pre", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_candidate_models() -> dict[str, Pipeline]:
    """The three baselines compared in evaluate.py (all fully implemented)."""
    return {
        # Floor: predicts by sampling from the class distribution.
        "dummy": make_pipeline(
            DummyClassifier(strategy="stratified", random_state=config.SEED)
        ),
        # Interpretable linear baseline; class_weight handles the imbalance.
        "logreg": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.SEED,
            )
        ),
        # Stronger baseline (gradient-boosted trees); the chosen family.
        "hgb": make_pipeline(
            HistGradientBoostingClassifier(random_state=config.SEED)
        ),
    }


def build_chosen_model() -> Pipeline:
    """The HGB pipeline used for the final model (no threshold tuning)."""
    return make_pipeline(
        HistGradientBoostingClassifier(random_state=config.SEED, **CHOSEN_HGB_PARAMS)
    )


def build_chosen_pipeline() -> TunedThresholdClassifierCV:
    """The model main.py trains and predicts with.

    Wraps the chosen HGB pipeline in ``TunedThresholdClassifierCV`` so the
    decision threshold is chosen to maximise macro-F1 on inner CV folds only
    (leakage-safe). ``.predict`` then applies that tuned threshold.
    """
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=build_chosen_model(),
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=config.SEED,
    )


# Human-readable name used in summaries/reports.
CHOSEN_MODEL_NAME = "HistGradientBoosting + macro-F1 threshold tuning"
