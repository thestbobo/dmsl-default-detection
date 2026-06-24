"""Central configuration: loads 'config.yaml' and exposes the project knobs.

The tunable values (seed, CV, threshold grid, column groups, the chosen
production config, the experiment registry) live in 'config.yaml' at the repo
root, that file is the single source of truth. This module loads it once at
import and re-exports the values under the **same names** other modules already
import ('config.SEED', 'config.NUMERIC', 'config.N_SPLITS', …), so nothing
downstream changes.

Things that are *code, not knobs* stay here: repo-relative paths, the
label/id detection helpers, and 'set_seed'.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Paths (anchored at the repository root, i.e. the parent of 'src/')
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

CONFIG_YAML = ROOT / "config.yaml"

# --------------------------------------------------------------------------- #
# Load the declarative config
# --------------------------------------------------------------------------- #
with CONFIG_YAML.open("r") as _fh:
    _CFG = yaml.safe_load(_fh)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = int(_CFG["seed"])


def set_seed(seed: int = SEED) -> None:
    """Seed every source of randomness we rely on.

    scikit-learn estimators are seeded explicitly via 'random_state'; this
    covers the stdlib 'random' and global NumPy RNG for anything else.
    """
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
N_SPLITS = int(_CFG["cv"]["n_splits"])
# Fold seeds for paired repeated-CV robustness checks.
VALIDATION_SEEDS = list(_CFG["cv"]["validation_seeds"])

# --------------------------------------------------------------------------- #
# Deployed objective: threshold grid swept on OOF probabilities (macro-F1).
# --------------------------------------------------------------------------- #
_TG = _CFG["threshold_grid"]
THRESHOLDS = np.linspace(float(_TG["start"]), float(_TG["stop"]), int(_TG["num"]))

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Chosen production config (what main.py trains) + experiment registry
# --------------------------------------------------------------------------- #
# The chosen engineered-feature groups and any tuned HGB params live in config.yaml.
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

# --------------------------------------------------------------------------- #
# Encoding / preprocessing knobs
# --------------------------------------------------------------------------- #
# Canonical shape of an encoding spec. config.yaml entries may be partial; they are
# merged onto these defaults so every spec is complete. The defaults reproduce the
# baseline encoding (scale on, raw columns).
ENCODING_DEFAULTS: dict = {
    "pay_remap": False,   # fold PAY_* {-2,-1,0} -> 0 (monotone "months delinquent")
    "pay_flags": False,   # append per-month IS_DELINQ_PAY_* binaries
    "scale": True,        # StandardScaler in the numeric pipe (no-op for trees)
    "log1p": False,       # signed log1p on monetary cols (LIMIT_BAL/BILL_AMT*/PAY_AMT*)
    "bill_neg": "keep",   # negative BILL_AMT: keep / clip (at 0) / flag (overpayment)
}


def normalize_encoding(spec) -> dict:
    """Merge a (possibly partial) encoding spec onto :data:`ENCODING_DEFAULTS`."""
    return {**ENCODING_DEFAULTS, **(spec or {})}


# Encoding spec for the deployed model (config.yaml 'chosen.encoding').
CHOSEN_ENCODING: dict = normalize_encoding(_CFG["chosen"].get("encoding"))

# Named encoding variants swept by experiments/encoding_experiments.py.
ENCODING_CONFIGS: dict[str, dict] = {
    name: normalize_encoding(spec)
    for name, spec in (_CFG.get("experiments", {}).get("encoding_configs", {}) or {}).items()
}

# --------------------------------------------------------------------------- #
# Model / ensemble registries
# --------------------------------------------------------------------------- #
# Named model specs ({kind, params, encoding-variant-name}) swept by
# experiments/model_experiments.py; the script resolves 'kind' -> estimator class and
# 'encoding' -> ENCODING_CONFIGS[name]. Left as raw dicts (estimator construction is
# code, not a knob).
MODEL_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("model_configs", {}) or {}
)

# Named equal-weight soft-vote ensembles (lists of MODEL_CONFIGS names) previewed by
# experiments/model_experiments.py.
ENSEMBLE_CONFIGS: dict[str, list[str]] = {
    name: list(members or [])
    for name, members in (_CFG.get("experiments", {}).get("ensemble_configs", {}) or {}).items()
}

# Named class-weight / loss-reweighting configs swept by
# experiments/imbalance_experiments.py. Each entry points at a base MODEL_CONFIGS
# member and overrides estimator params.
IMBALANCE_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("imbalance_configs", {}) or {}
)

# Named LightGBM (boosting-library) configs swept by
# experiments/boosting_experiments.py. Same {kind, params, encoding} shape as
# MODEL_CONFIGS; 'kind: lgbm' resolves to the lazy LGBMClassifier factory.
BOOSTING_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("boosting_configs", {}) or {}
)

# Named resampling configs swept by experiments/resampling_experiments.py (optional
# dependency: imbalanced-learn). Each entry: 'base_model' (a MODEL_CONFIGS name) +
# 'sampler' (resolved to an imblearn resampler by the script).
RESAMPLING_CONFIGS: dict[str, dict] = dict(
    _CFG.get("experiments", {}).get("resampling_configs", {}) or {}
)

# --------------------------------------------------------------------------- #
# Label / id detection
# --------------------------------------------------------------------------- #
# The development set may name the target 'label' (DSLE export) or
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
