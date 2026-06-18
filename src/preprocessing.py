"""Cleaning and feature transformation.

Two pieces:

1. ``CodeFolder`` — folds the undocumented categorical codes into the
   documented "other" buckets. This is a *stateless, deterministic* domain
   mapping (nothing is learned from the data), so it is leakage-free and can
   sit at the front of the pipeline, applied identically to train and eval.

2. ``build_preprocessor`` — a ``ColumnTransformer`` that imputes + scales the
   numeric columns and imputes + one-hot-encodes the (cleaned) categoricals.
   Because it lives inside the estimator Pipeline, it is *fit on training
   folds only* during cross-validation: no data leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

# Undocumented -> documented "other" mappings (domain knowledge, not learned).
EDUCATION_FOLD = {0: 4, 5: 4, 6: 4}  # spec documents only 1..4
MARRIAGE_FOLD = {0: 3}  # spec documents only 1..3


def fold_undocumented_codes(X: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented EDUCATION/MARRIAGE codes into 'other'.

    EDUCATION {0, 5, 6} -> 4 ; MARRIAGE {0} -> 3. Returns a new DataFrame.
    """
    X = X.copy()
    if "EDUCATION" in X.columns:
        X["EDUCATION"] = X["EDUCATION"].replace(EDUCATION_FOLD)
    if "MARRIAGE" in X.columns:
        X["MARRIAGE"] = X["MARRIAGE"].replace(MARRIAGE_FOLD)
    return X


def make_code_folder() -> FunctionTransformer:
    """A stateless transformer wrapping :func:`fold_undocumented_codes`."""
    return FunctionTransformer(fold_undocumented_codes, feature_names_out="one-to-one")


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: numeric (impute+scale) and categorical (impute+one-hot).

    Imputation guards against unexpected missing values even though the dataset
    is normally complete. Scaling helps linear models and is harmless for trees.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC),
            ("cat", categorical_pipe, config.CATEGORICAL),
        ],
        remainder="drop",
    )
