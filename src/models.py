"""Model factory.

- 'make_pipeline' glues the cleaning + preprocessing in front of any estimator.
- 'get_candidate_models' returns the three baselines compared in evaluate.py.
- 'build_chosen_pipeline' returns the single model that main.py trains for the
  submission: the configured pipeline (a class-balanced random forest on the
  engineered feature set, by default) wrapped in a threshold tuner that optimises
  macro-F1 (leakage-safe, via inner CV).

To change the deployed model after tuning in experiments/, edit the 'chosen:'
block in 'config.yaml' ('ensemble' / 'feature_groups' / 'encoding'), NOT
this file.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import (
    build_preprocessor,
    encoding_extra_columns,
    engineered_feature_names,
    make_code_folder,
    make_encoder,
    make_feature_engineer,
)


def make_pipeline(estimator, feature_groups=(), encoding=None) -> Pipeline:
    """Wrap an estimator with code-folding + (optional) feature engineering + encoding + preprocessing.

    'feature_groups' is an iterable of engineered-feature family names (see
    'preprocessing.ENGINEERED_COLUMNS'); 'encoding' is an optional encoding
    spec (see 'preprocessing.apply_encoding' / 'config.ENCODING_DEFAULTS'). With
    neither (the defaults) the pipeline is identical to the raw-feature baseline.

    Order is 'fold -> [engineer] -> [encode] -> pre': feature engineering runs
    *before* encoding so engineered ratios are computed from the raw amounts;
    encoding only re-expresses the model columns. Engineered / encoding-flag columns
    are routed through the numeric pipe, and the 'scale' knob gates
    'StandardScaler' there.
    """
    feature_groups = list(feature_groups)
    encoding = dict(encoding or {})
    steps = [("fold", make_code_folder())]
    if feature_groups:
        steps.append(("engineer", make_feature_engineer(feature_groups)))
    if encoding:
        steps.append(("encode", make_encoder(encoding)))
    extra_numeric = encoding_extra_columns(encoding) + engineered_feature_names(feature_groups)
    steps.append(("pre", build_preprocessor(extra_numeric, scale=encoding.get("scale", True))))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _make_lgbm(**kwargs):
    """LightGBM classifier, imported lazily.

    LightGBM is the only non-sklearn estimator and an optional dependency: importing
    it here (not at module top) means 'main.py' and every other experiment keep
    working even when LightGBM / its OpenMP runtime is absent, the import only fires
    if a 'kind: lgbm' config is actually requested. 'verbose=-1' silences its
    per-fit chatter. sklearn-compatible API, so it drops into the Pipeline unchanged.
    """
    from lightgbm import LGBMClassifier

    kwargs.setdefault("verbose", -1)
    kwargs.setdefault("n_jobs", -1)
    return LGBMClassifier(**kwargs)


# Estimator families addressable by the config.yaml 'kind' knob (experiments
# model_configs + chosen.ensemble), shared by experiments/model_experiments.py and the
# deployed ensemble below so the deployed members match the ones scored in the bake-off.
# Every class accepts random_state (injected in make_estimator); 'lgbm' is a lazy
# factory (optional dep), the rest are sklearn classes.
_ESTIMATORS = {
    "hgb": HistGradientBoostingClassifier,
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "et": ExtraTreesClassifier,
    "gb": GradientBoostingClassifier,
    "lgbm": _make_lgbm,
}


def make_estimator(kind: str, params: dict | None = None):
    """Instantiate an estimator family by name, seeded for reproducibility."""
    if kind not in _ESTIMATORS:
        raise ValueError(f"unknown estimator kind {kind!r} (expected one of {list(_ESTIMATORS)})")
    kwargs = dict(params or {})
    kwargs.setdefault("random_state", config.SEED)
    return _ESTIMATORS[kind](**kwargs)


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


def _member_pipeline(name: str, feature_groups=()) -> Pipeline:
    """Build one ensemble member from its ``experiments.model_configs`` spec.

    Each member is a full preprocessing+estimator pipeline using its own named
    encoding variant, so the deployed soft-vote is identical to what
    'experiments/model_experiments.py' scored. 'feature_groups' prepends
    engineered-feature families; only the single deployed member uses it
    (multi-member soft-votes keep each member's own raw feature space).
    """
    spec = config.MODEL_CONFIGS[name]
    encoding = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(
        make_estimator(spec["kind"], spec.get("params")),
        feature_groups=feature_groups,
        encoding=encoding,
    )


def build_chosen_ensemble(members) -> VotingClassifier:
    """Equal-weight soft-vote ('VotingClassifier') of the named model configs.

    Averages the members' ``predict_proba``; members fit in parallel ('n_jobs=-1').
    Wrapped by 'build_chosen_pipeline' in the same threshold tuner as the single model.
    """
    estimators = [(name, _member_pipeline(name)) for name in members]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def build_chosen_model():
    """The model used for the final submission (no threshold tuning yet).

    If 'config.yaml' 'chosen.ensemble' lists members, deploy their equal-weight
    soft-vote; with a single member, deploy that one estimator. Feature groups,
    encoding spec and any HGB params come from the 'chosen:' block.
    """
    if config.CHOSEN_ENSEMBLE:
        if len(config.CHOSEN_ENSEMBLE) == 1:
            # The deployed single model carries the chosen engineered-feature groups;
            # chosen.feature_groups applies only to this single-member deploy path.
            return _member_pipeline(config.CHOSEN_ENSEMBLE[0], config.CHOSEN_FEATURE_GROUPS)
        return build_chosen_ensemble(config.CHOSEN_ENSEMBLE)
    return make_pipeline(
        HistGradientBoostingClassifier(random_state=config.SEED, **config.CHOSEN_HGB_PARAMS),
        feature_groups=config.CHOSEN_FEATURE_GROUPS,
        encoding=config.CHOSEN_ENCODING,
    )


def build_chosen_pipeline() -> TunedThresholdClassifierCV:
    """The model main.py trains and predicts with.

    Wraps the chosen pipeline in 'TunedThresholdClassifierCV' so the decision
    threshold is chosen to maximise macro-F1 on inner CV folds only (leakage-safe).
    '.predict' then applies that tuned threshold.
    """
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=build_chosen_model(),
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=config.SEED,
    )


# Human-readable name used in summaries/reports (reflects the chosen.ensemble setting).
CHOSEN_MODEL_NAME = (
    (
        f"{config.CHOSEN_ENSEMBLE[0]} + macro-F1 threshold tuning"
        if len(config.CHOSEN_ENSEMBLE) == 1
        else f"Soft-vote ensemble [{' + '.join(config.CHOSEN_ENSEMBLE)}] + macro-F1 threshold tuning"
    )
    if config.CHOSEN_ENSEMBLE
    else "HistGradientBoosting + macro-F1 threshold tuning"
)
