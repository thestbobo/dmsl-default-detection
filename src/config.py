"""Loads config.yaml once at import and re-exports the values (seed, CV, threshold
grid, column groups, chosen config, experiment registry) under the names the rest of
the code imports. Paths, label/id detection and set_seed live here as code, not knobs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# Paths, anchored at the repo root (parent of src/).
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DEV_CSV = DATA_RAW / "development.csv"
EVAL_CSV = DATA_RAW / "evaluation.csv"
SAMPLE_SUB = DATA_RAW / "sample_submission.csv"

OUTPUTS = ROOT / "outputs"
OUT_SUB_DIR = OUTPUTS / "submissions"
OUT_FIG_DIR = OUTPUTS / "figures"
OUT_SUB = OUT_SUB_DIR / "submission.csv"

CONFIG_YAML = ROOT / "config.yaml"

with CONFIG_YAML.open("r") as _fh:
    _CFG = yaml.safe_load(_fh)

# Reproducibility. Estimators are seeded via random_state; this covers stdlib random
# and the global NumPy RNG.
SEED = int(_CFG["seed"])


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# Cross-validation. validation_seeds are the fold seeds for paired repeated-CV checks.
N_SPLITS = int(_CFG["cv"]["n_splits"])
VALIDATION_SEEDS = list(_CFG["cv"]["validation_seeds"])

# Deployed objective: threshold grid swept on OOF probabilities (macro-F1).
_TG = _CFG["threshold_grid"]
THRESHOLDS = np.linspace(float(_TG["start"]), float(_TG["stop"]), int(_TG["num"]))

# Column groups.
_COLS = _CFG["columns"]
# Categorical (nominal) features, one-hot encoded after folding undocumented codes.
CATEGORICAL = list(_COLS["categorical"])

# Repayment status. Documented as -1 / 1..9, but the real data also contains
# -2 and 0. Treated as ordinal numeric (per the assignment), not one-hot.
PAY_COLS = list(_COLS["pay"])

# Bill statement amounts (NT$) for Sep..Apr 2005.
BILL_COLS = list(_COLS["bill"])

# Previous payment amounts (NT$) for Sep..Apr 2005.
PAY_AMT_COLS = list(_COLS["pay_amt"])

# Everything modelled as numeric (PAY_* kept as ordinal numeric).
NUMERIC = list(_COLS["base_numeric"]) + PAY_COLS + BILL_COLS + PAY_AMT_COLS

# Full set of feature columns the model expects (order-independent).
FEATURE_COLS = CATEGORICAL + NUMERIC

# Chosen production config (what main.py trains) + experiment registry.
CHOSEN_FEATURE_GROUPS: tuple = tuple(_CFG["chosen"].get("feature_groups") or [])
CHOSEN_HGB_PARAMS: dict = dict(_CFG["chosen"].get("hgb_params") or {})
# If non-empty, main.py deploys these MODEL_CONFIGS members: one entry -> that single
# estimator, several -> their equal-weight soft-vote. Empty -> the single HGB above.
CHOSEN_ENSEMBLE: list[str] = list(_CFG["chosen"].get("ensemble") or [])

# Named feature-group configs swept by experiments/feature_experiments.py.
FEATURE_CONFIGS: dict[str, list[str]] = {
    name: list(groups or [])
    for name, groups in (_CFG.get("experiments", {}).get("feature_configs", {}) or {}).items()
}

# Encoding / preprocessing knobs. config.yaml specs may be partial and are merged onto
# these defaults, which reproduce the baseline encoding (scale on, raw columns).
ENCODING_DEFAULTS: dict = {
    "pay_remap": False,   # fold PAY_* {-2,-1,0} -> 0 (monotone "months delinquent")
    "pay_flags": False,   # append per-month IS_DELINQ_PAY_* binaries
    "scale": True,        # StandardScaler in the numeric pipe (no-op for trees)
    "log1p": False,       # signed log1p on monetary cols (LIMIT_BAL/BILL_AMT*/PAY_AMT*)
    "bill_neg": "keep",   # negative BILL_AMT: keep / clip (at 0) / flag (overpayment)
}


def normalize_encoding(spec) -> dict:
    """Merge a possibly partial encoding spec onto ENCODING_DEFAULTS."""
    return {**ENCODING_DEFAULTS, **(spec or {})}


# Encoding spec for the deployed model (config.yaml 'chosen.encoding').
CHOSEN_ENCODING: dict = normalize_encoding(_CFG["chosen"].get("encoding"))

# Named encoding variants swept by experiments/encoding_experiments.py.
ENCODING_CONFIGS: dict[str, dict] = {
    name: normalize_encoding(spec)
    for name, spec in (_CFG.get("experiments", {}).get("encoding_configs", {}) or {}).items()
}

# Model / ensemble registries. Named model specs ({kind, params, encoding name}) swept
# by experiments/model_experiments.py.
MODEL_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("model_configs", {}) or {}
)

# Named equal-weight soft-vote ensembles (lists of MODEL_CONFIGS names) previewed by
# experiments/model_experiments.py.
ENSEMBLE_CONFIGS: dict[str, list[str]] = {
    name: list(members or [])
    for name, members in (_CFG.get("experiments", {}).get("ensemble_configs", {}) or {}).items()
}

# class_weight configs (base MODEL_CONFIGS member + param overrides) swept by
# experiments/imbalance_experiments.py.
IMBALANCE_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("imbalance_configs", {}) or {}
)

# LightGBM configs swept by experiments/boosting_experiments.py (optional dependency).
BOOSTING_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("boosting_configs", {}) or {}
)

# Resampling configs (base_model + sampler) swept by
# experiments/resampling_experiments.py (optional dependency: imbalanced-learn).
RESAMPLING_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("resampling_configs", {}) or {}
)

# Label / id detection. The target may be named 'label' (DSLE export) or
# 'default.payment.next.month' (original UCI naming).
LABEL_CANDIDATES = ["label", "default.payment.next.month"]


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
    """Return the name of the id column ('id'/'ID'), raising if absent."""
    col = _find_col(df.columns, ["id"])
    if col is None:
        raise KeyError(
            "No id column found (expected 'id'/'ID', case-insensitive); "
            f"got columns: {list(df.columns)}"
        )
    return col
