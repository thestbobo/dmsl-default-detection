"""Data loading, schema checks, and submission I/O.

Loaders return raw DataFrames; the feature/target/id helpers are shared by main.py,
evaluate.py and the experiments. The submission writer/validator enforce the strict
Id,Predicted format the assignment expects.
"""

from __future__ import annotations

import pandas as pd

from . import config

SUBMISSION_COLUMNS = ["Id", "Predicted"]


def _read_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "The dataset is not committed. See data/raw/README.md for the download "
            "link and place the three CSV files in data/raw/."
        )
    return pd.read_csv(path)


def load_development() -> pd.DataFrame:
    """Load the development set (features + label)."""
    return _read_csv(config.DEV_CSV)


def load_evaluation() -> pd.DataFrame:
    """Load the evaluation set (features + id column)."""
    return _read_csv(config.EVAL_CSV)


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model feature columns in a fixed order (drops id and label)."""
    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}. Got: {list(df.columns)}")
    return df[config.FEATURE_COLS].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a development DataFrame into (X features, y target)."""
    label = config.detect_label_col(df)
    return get_features(df), df[label].astype(int)


def get_eval_ids(df: pd.DataFrame) -> pd.Series:
    """Return the evaluation id column (used for the submission)."""
    return df[config.detect_id_col(df)]


def write_submission(ids, preds, path=config.OUT_SUB) -> int:
    """Write predictions as an Id,Predicted CSV. Returns the row count."""
    ids = pd.Series(ids).reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True).astype(int)
    if len(ids) != len(preds):
        raise ValueError(f"ids ({len(ids)}) and preds ({len(preds)}) length mismatch")
    out = pd.DataFrame({"Id": ids, "Predicted": preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)


def validate_submission(path, eval_df: pd.DataFrame) -> None:
    """Check a written submission's header, row count, and ids against the eval set."""
    sub = pd.read_csv(path)
    if list(sub.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Bad header {list(sub.columns)}; expected {SUBMISSION_COLUMNS}")
    eval_ids = eval_df[config.detect_id_col(eval_df)].reset_index(drop=True)
    if len(sub) != len(eval_ids):
        raise ValueError(f"Row count {len(sub)} != evaluation rows {len(eval_ids)}")
    if not sub["Id"].reset_index(drop=True).equals(eval_ids.astype(sub["Id"].dtype)):
        raise ValueError("Submission Id column does not match evaluation ids")
