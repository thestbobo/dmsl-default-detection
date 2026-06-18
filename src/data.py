"""Data loading and light schema validation.

Loaders return raw pandas DataFrames; feature/target separation and id
extraction are done with small helpers so the same logic is reused by
``main.py``, ``evaluate.py`` and the experiments.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _read_csv(path) -> pd.DataFrame:
    """Read a CSV, with a clear message pointing at the download guide."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "The dataset is not committed to the repo. See data/raw/README.md "
            "for the Google Drive download link and place the three CSV files "
            "in data/raw/."
        )
    return pd.read_csv(path)


def load_development() -> pd.DataFrame:
    """Load the development set (features + label)."""
    return _read_csv(config.DEV_CSV)


def load_evaluation() -> pd.DataFrame:
    """Load the evaluation set (features only, plus the id column)."""
    return _read_csv(config.EVAL_CSV)


def _check_features(df: pd.DataFrame) -> None:
    """Ensure every expected feature column is present."""
    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected feature columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the model feature columns, in a fixed order.

    Selecting by ``FEATURE_COLS`` guarantees the development and evaluation
    matrices line up and silently drops the id/ID column (and the label, if
    present) from the feature space.
    """
    _check_features(df)
    return df[config.FEATURE_COLS].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a development DataFrame into (X features, y target)."""
    label = config.detect_label_col(df)
    X = get_features(df)
    y = df[label].astype(int)
    return X, y


def get_eval_ids(df: pd.DataFrame) -> pd.Series:
    """Return the id column from the evaluation set (used for the submission)."""
    id_col = config.detect_id_col(df)
    return df[id_col]
