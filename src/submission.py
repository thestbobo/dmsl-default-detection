"""Build and validate the competition submission file.

Submission format is strict (from the assignment):
- header exactly ``Id,Predicted``
- one row per evaluation record
- ``Id`` equals the evaluation ``id`` column
"""

from __future__ import annotations

import pandas as pd

from . import config

SUBMISSION_COLUMNS = ["Id", "Predicted"]


def write_submission(ids, preds, path=config.OUT_SUB) -> int:
    """Write predictions to a strict ``Id,Predicted`` CSV. Returns row count."""
    ids = pd.Series(ids).reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True).astype(int)

    if len(ids) != len(preds):
        raise ValueError(
            f"ids ({len(ids)}) and preds ({len(preds)}) length mismatch"
        )

    out = pd.DataFrame({"Id": ids, "Predicted": preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)


def validate_submission(path, eval_df: pd.DataFrame) -> None:
    """Sanity-check a written submission against the evaluation set.

    Verifies the header, the row count, and that the Ids exactly match the
    evaluation id column (same set, same order). Raises on any mismatch.
    """
    sub = pd.read_csv(path)

    if list(sub.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Bad header {list(sub.columns)}; expected {SUBMISSION_COLUMNS}"
        )

    eval_ids = eval_df[config.detect_id_col(eval_df)].reset_index(drop=True)
    if len(sub) != len(eval_ids):
        raise ValueError(
            f"Row count {len(sub)} != evaluation rows {len(eval_ids)}"
        )
    if not sub["Id"].reset_index(drop=True).equals(eval_ids.astype(sub["Id"].dtype)):
        raise ValueError("Submission Id column does not match evaluation ids")
