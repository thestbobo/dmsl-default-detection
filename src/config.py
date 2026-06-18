"""Central configuration: paths, the global seed, column groups, and robust
column detection helpers.

Everything that other modules need to agree on lives here so there is a single
source of truth (e.g. the SEED, the exact submission header, the column groups).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every source of randomness we rely on.

    scikit-learn estimators are seeded explicitly via ``random_state``; this
    covers the stdlib ``random`` and global NumPy RNG for anything else.
    """
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Paths (anchored at the repository root, i.e. the parent of ``src/``)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DEV_CSV = DATA_RAW / "development.csv"
EVAL_CSV = DATA_RAW / "evaluation.csv"
SAMPLE_SUB = DATA_RAW / "sample_submission.csv"

OUTPUTS = ROOT / "outputs"
OUT_SUB_DIR = OUTPUTS / "submissions"
OUT_FIG_DIR = OUTPUTS / "figures"
OUT_SUB = OUT_SUB_DIR / "submission.csv"

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
# Categorical (nominal) features — one-hot encoded after folding undocumented codes.
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

# Repayment status. Documented as -1 / 1..9, but the real data also contains
# -2 and 0. Treated as ordinal numeric (per the assignment), not one-hot.
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Bill statement amounts (NT$) for Sep..Apr 2005.
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]

# Previous payment amounts (NT$) for Sep..Apr 2005.
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Everything modelled as numeric (PAY_* kept as ordinal numeric).
NUMERIC = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS

# Full set of feature columns the model expects (order-independent).
FEATURE_COLS = CATEGORICAL + NUMERIC

# --------------------------------------------------------------------------- #
# Label / id detection
# --------------------------------------------------------------------------- #
# The development set may name the target ``label`` (DSLE export) or
# ``default.payment.next.month`` (original UCI naming).
LABEL_CANDIDATES = ["label", "default.payment.next.month"]

# Cross-validation
N_SPLITS = 5


def _find_col(columns, candidates) -> str | None:
    """Return the first column matching any candidate, case-insensitively."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def detect_label_col(df) -> str:
    """Return the name of the target column, raising if none is present."""
    col = _find_col(df.columns, LABEL_CANDIDATES)
    if col is None:
        raise KeyError(
            "No label column found. Expected one of "
            f"{LABEL_CANDIDATES} (case-insensitive); got columns: {list(df.columns)}"
        )
    return col


def detect_id_col(df) -> str:
    """Return the name of the id column (``id``/``ID``), raising if absent."""
    col = _find_col(df.columns, ["id"])
    if col is None:
        raise KeyError(
            "No id column found (expected 'id'/'ID', case-insensitive); "
            f"got columns: {list(df.columns)}"
        )
    return col
