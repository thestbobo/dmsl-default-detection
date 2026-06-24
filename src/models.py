"""Model factory.

- make_pipeline puts cleaning + preprocessing in front of any estimator.
- get_candidate_models returns the three baselines compared in evaluate.py.
- build_chosen_pipeline returns the model main.py trains: the configured pipeline
  wrapped in a macro-F1 threshold tuner (leakage-safe inner CV).

To change the deployed model, edit the 'chosen:' block in config.yaml, not this file.
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
    """Estimator behind fold -> [engineer] -> [encode] -> preprocess.

    Engineering runs before encoding so ratios use the raw amounts. With no feature
    groups and no encoding the pipeline is the raw-feature baseline.
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
    """LightGBM classifier, imported lazily so the optional dependency is only needed
    when a 'kind: lgbm' config is actually requested."""
    from lightgbm import LGBMClassifier

    kwargs.setdefault("verbose", -1)
    kwargs.setdefault("n_jobs", -1)
    return LGBMClassifier(**kwargs)


# Estimator families addressable by the config.yaml 'kind' knob. 'lgbm' is a lazy
# factory (optional dep); the rest are sklearn classes seeded in make_estimator.
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
    """Build one model from its config.MODEL_CONFIGS spec, using its own encoding variant.

    'feature_groups' prepends engineered features; only the single deployed member uses
    it (multi-member soft-votes keep each member's raw feature space).
    """
    spec = config.MODEL_CONFIGS[name]
    encoding = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    return make_pipeline(
        make_estimator(spec["kind"], spec.get("params")),
        feature_groups=feature_groups,
        encoding=encoding,
    )


def build_chosen_ensemble(members) -> VotingClassifier:
    """Equal-weight soft-vote of the named model configs (averages their predict_proba)."""
    estimators = [(name, _member_pipeline(name)) for name in members]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def build_chosen_model():
    """The chosen model (no threshold tuning yet), per the config.yaml 'chosen' block.

    A multi-member 'ensemble' deploys their soft-vote; a single member deploys that one
    estimator (carrying chosen.feature_groups); empty deploys the tuned HGB.
    """
    if config.CHOSEN_ENSEMBLE:
        if len(config.CHOSEN_ENSEMBLE) == 1:
            return _member_pipeline(config.CHOSEN_ENSEMBLE[0], config.CHOSEN_FEATURE_GROUPS)
        return build_chosen_ensemble(config.CHOSEN_ENSEMBLE)
    return make_pipeline(
        HistGradientBoostingClassifier(random_state=config.SEED, **config.CHOSEN_HGB_PARAMS),
        feature_groups=config.CHOSEN_FEATURE_GROUPS,
        encoding=config.CHOSEN_ENCODING,
    )


def build_chosen_pipeline() -> TunedThresholdClassifierCV:
    """The chosen model wrapped in TunedThresholdClassifierCV: predicts at the macro-F1
    optimal threshold found on inner CV folds (leakage-safe). This is what main.py trains."""
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=build_chosen_model(),
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=config.SEED,
    )


def _chosen_model_name() -> str:
    """Human-readable name for summaries/reports, reflecting the chosen.ensemble setting."""
    members = config.CHOSEN_ENSEMBLE
    if not members:
        base = "HistGradientBoosting"
    elif len(members) == 1:
        base = members[0]
    else:
        base = f"Soft-vote ensemble [{' + '.join(members)}]"
    return f"{base} + macro-F1 threshold tuning"


CHOSEN_MODEL_NAME = _chosen_model_name()
