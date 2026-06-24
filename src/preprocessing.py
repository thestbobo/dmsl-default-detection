"""Cleaning and feature transformation.

All transforms are stateless row-wise mappings (or live inside the estimator Pipeline),
so they are leakage-free and identical on train and eval:

- fold_undocumented_codes: folds undocumented EDUCATION/MARRIAGE codes into 'other'.
- add_engineered_features: appends repayment-history aggregates, selected by 'groups'.
- apply_encoding: re-expresses the raw columns per a declarative encoding spec.
- build_preprocessor: ColumnTransformer that imputes/scales numerics and one-hot-encodes
  categoricals, fit on training folds only inside the Pipeline.
"""

from __future__ import annotations

import numpy as np
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
    """Fold undocumented codes into 'other': EDUCATION {0,5,6} -> 4, MARRIAGE {0} -> 3."""
    X = X.copy()
    if "EDUCATION" in X.columns:
        X["EDUCATION"] = X["EDUCATION"].replace(EDUCATION_FOLD)
    if "MARRIAGE" in X.columns:
        X["MARRIAGE"] = X["MARRIAGE"].replace(MARRIAGE_FOLD)
    return X


def make_code_folder() -> FunctionTransformer:
    """Stateless transformer wrapping fold_undocumented_codes."""
    return FunctionTransformer(fold_undocumented_codes, feature_names_out="one-to-one")


# Engineered features. Group name -> the column names it produces (order matters so
# build_preprocessor can select them by name).
ENGINEERED_COLUMNS: dict[str, list[str]] = {
    # PAY_* delinquency aggregates (PAY_0 = most recent month ... PAY_6 = oldest).
    "pay": [
        "PAY_MAX",          # worst delinquency across the 6 months
        "PAY_MEAN",         # average repayment status
        "PAY_N_LATE",       # # months late (PAY >= 1)
        "PAY_N_DULY",       # # months paid duly (PAY <= 0)
        "PAY_CURR_DELINQ",  # currently delinquent (PAY_0 >= 1)
        "PAY_TREND",        # PAY_0 - PAY_6 ; > 0 means worsening recently
        "PAY_WORST_STREAK", # longest run of consecutive late months
    ],
    # Credit utilisation = BILL_AMT_t / LIMIT_BAL.
    "util": ["UTIL_MEAN", "UTIL_MAX", "UTIL_LAST"],
    # Payment coverage / ratios.
    "payratio": [
        "FRAC_BILL_PAID_MEAN",  # mean PAY_AMT_t / BILL_AMT_t (same month)
        "COVER_PREV_MEAN",      # mean PAY_AMT_t / previous month's bill
        "N_ZERO_PAY",           # # months with no payment
        "N_PAID_FULL",          # # months paid in full (PAY_AMT >= BILL, BILL > 0)
    ],
    # Bill dynamics + remaining credit.
    "bill": [
        "BILL_MEAN",
        "BILL_MAX",
        "BILL_STD",         # volatility
        "BILL_TREND",       # BILL_AMT1 - BILL_AMT6 (recent - oldest)
        "REM_CREDIT_LAST",  # LIMIT_BAL - BILL_AMT1
        "REM_CREDIT_MEAN",  # LIMIT_BAL - mean(BILL)
    ],
    # PAY_* semantic decomposition. The codes are not a monotone scale: -2 no consumption,
    # -1 paid in full, 0 revolving, 1..9 months late. Keeps those states separate (the
    # 'pay' family collapses {-2,-1,0}) and isolates the most-recent month.
    "paysem": [
        "PAY_N_REVOLVING",  # # months carrying a revolving balance (PAY == 0)
        "PAY_N_PAIDFULL",   # # months paid in full (PAY == -1)
        "PAY_N_NOCONS",     # # months with no consumption (PAY == -2)
        "PAY0_GE2",         # latest month >=2 months late (PDP jumps here)
        "PAY0_REVOLVING",   # latest month is revolving credit (PAY_0 == 0)
        "PAY0_PAIDFULL",    # latest month paid in full (PAY_0 == -1)
    ],
    # Recent-month payment coverage + trend ('payratio' only has the mean across months).
    "coverx": [
        "COVER_RECENT",  # PAY_AMT1 / BILL_AMT2 (latest payment vs prior bill)
        "COVER_TREND",   # recent coverage - oldest coverage (improving/worsening)
        "UTIL_TREND",    # latest utilisation - oldest utilisation (BILL/LIMIT)
    ],
    # Per-month utilisation (BILL_AMT_t / LIMIT_BAL); granular version of 'util'.
    "utilmonths": ["UTIL_1", "UTIL_2", "UTIL_3", "UTIL_4", "UTIL_5", "UTIL_6"],
    # Per-month same-month payment ratio (PAY_AMT_t / BILL_AMT_t); granular 'payratio'.
    "payamtratio": ["PAYR_1", "PAYR_2", "PAYR_3", "PAYR_4", "PAYR_5", "PAYR_6"],
    # Credit-stress interactions (utilisation x delinquency), which an axis-aligned tree
    # cannot form from the raw columns.
    "stress": [
        "STRESS_DELINQ_UTIL",  # PAY0>=2 flag * latest utilisation
        "STRESS_MAXPAY_UTIL",  # worst delinquency * mean utilisation
        "STRESS_CURR",         # max(PAY_0,0) * latest utilisation
    ],
    # PAY_* dynamics: the trajectory of delinquency (recovery/escalation/recent-vs-old),
    # not just the per-state counts in 'paysem'.
    "paysem2": [
        "PAY_RECOVERED",      # was ever late (max>=1) but currently duly (PAY_0<=0)
        "PAY_NEWLY_LATE",     # late this month, not last (PAY_0>=1 & PAY_2<=0)
        "PAY_DELINQ_RECENT3", # # late months in the recent half (PAY_0,2,3)
        "PAY_DELINQ_OLD3",    # # late months in the older half (PAY_4,5,6)
        "PAY_ESCALATION",     # PAY_0 - max(older months): is the latest the worst yet
    ],
}


def engineered_feature_names(groups) -> list[str]:
    """Flat list of engineered column names produced by 'groups' (in order)."""
    names: list[str] = []
    for g in groups:
        names.extend(ENGINEERED_COLUMNS[g])
    return names


def _safe_ratio(num: np.ndarray, den: np.ndarray, valid: np.ndarray,
                fallback: float) -> np.ndarray:
    """num/den where 'valid', else 'fallback', never divides by zero."""
    den_safe = np.where(valid, den, 1.0)
    return np.where(valid, num / den_safe, fallback)


def _longest_late_streak(late: np.ndarray) -> np.ndarray:
    """Longest run of consecutive True along axis 1 (per row)."""
    run = np.zeros(late.shape[0])
    best = np.zeros(late.shape[0])
    for j in range(late.shape[1]):
        run = np.where(late[:, j], run + 1.0, 0.0)
        best = np.maximum(best, run)
    return best


def add_engineered_features(X: pd.DataFrame, groups=()) -> pd.DataFrame:
    """Append the requested engineered-feature families (names from ENGINEERED_COLUMNS).

    No groups is a no-op (the raw-feature baseline). Divide-by-zero is guarded with safe
    denominators; residual non-finite values become NaN and are imputed in the Pipeline.
    """
    groups = list(groups)
    if not groups:
        return X

    X = X.copy()
    pay = X[config.PAY_COLS].to_numpy(dtype=float)       # col 0 = PAY_0 (recent)
    bill = X[config.BILL_COLS].to_numpy(dtype=float)     # col 0 = BILL_AMT1 (recent)
    pay_amt = X[config.PAY_AMT_COLS].to_numpy(dtype=float)
    limit = X["LIMIT_BAL"].to_numpy(dtype=float)

    new: dict[str, np.ndarray] = {}

    if "pay" in groups:
        new["PAY_MAX"] = pay.max(axis=1)
        new["PAY_MEAN"] = pay.mean(axis=1)
        new["PAY_N_LATE"] = (pay >= 1).sum(axis=1).astype(float)
        new["PAY_N_DULY"] = (pay <= 0).sum(axis=1).astype(float)
        new["PAY_CURR_DELINQ"] = (pay[:, 0] >= 1).astype(float)
        new["PAY_TREND"] = pay[:, 0] - pay[:, -1]
        new["PAY_WORST_STREAK"] = _longest_late_streak(pay >= 1)

    if "util" in groups:
        limit_col = limit[:, None]
        valid = limit_col > 0
        util = _safe_ratio(bill, limit_col, valid, np.nan)  # (n, 6)
        new["UTIL_MEAN"] = util.mean(axis=1)
        new["UTIL_MAX"] = util.max(axis=1)
        new["UTIL_LAST"] = util[:, 0]

    if "payratio" in groups:
        # Same-month fraction of bill paid; previous bill has no payment to match.
        frac = _safe_ratio(pay_amt, bill, bill > 0, 1.0)  # BILL<=0 -> nothing owed
        new["FRAC_BILL_PAID_MEAN"] = frac.mean(axis=1)
        # Payment in month t vs the previous month's statement (older = higher idx).
        prev_bill = bill[:, 1:]            # BILL_AMT2..6
        this_pay = pay_amt[:, :-1]         # PAY_AMT1..5
        cover = _safe_ratio(this_pay, prev_bill, prev_bill > 0, 1.0)
        new["COVER_PREV_MEAN"] = cover.mean(axis=1)
        new["N_ZERO_PAY"] = (pay_amt == 0).sum(axis=1).astype(float)
        new["N_PAID_FULL"] = ((pay_amt >= bill) & (bill > 0)).sum(axis=1).astype(float)

    if "paysem" in groups:
        # PAY_* codes: -2 no-consumption, -1 paid-in-full, 0 revolving, 1..9 months late.
        new["PAY_N_REVOLVING"] = (pay == 0).sum(axis=1).astype(float)
        new["PAY_N_PAIDFULL"] = (pay == -1).sum(axis=1).astype(float)
        new["PAY_N_NOCONS"] = (pay == -2).sum(axis=1).astype(float)
        new["PAY0_GE2"] = (pay[:, 0] >= 2).astype(float)
        new["PAY0_REVOLVING"] = (pay[:, 0] == 0).astype(float)
        new["PAY0_PAIDFULL"] = (pay[:, 0] == -1).astype(float)

    if "coverx" in groups:
        # Recent coverage = PAY_AMT1 / BILL_AMT2; oldest = PAY_AMT5 / BILL_AMT6.
        cover_recent = _safe_ratio(pay_amt[:, 0], bill[:, 1], bill[:, 1] > 0, 1.0)
        cover_old = _safe_ratio(pay_amt[:, 4], bill[:, 5], bill[:, 5] > 0, 1.0)
        new["COVER_RECENT"] = cover_recent
        new["COVER_TREND"] = cover_recent - cover_old
        util_last = _safe_ratio(bill[:, 0], limit, limit > 0, np.nan)
        util_old = _safe_ratio(bill[:, 5], limit, limit > 0, np.nan)
        new["UTIL_TREND"] = util_last - util_old

    if "utilmonths" in groups:
        util_m = _safe_ratio(bill, limit[:, None], limit[:, None] > 0, np.nan)  # (n, 6)
        for k in range(6):
            new[f"UTIL_{k + 1}"] = util_m[:, k]

    if "payamtratio" in groups:
        payr = _safe_ratio(pay_amt, bill, bill > 0, 1.0)  # BILL<=0 -> nothing owed
        for k in range(6):
            new[f"PAYR_{k + 1}"] = payr[:, k]

    if "stress" in groups:
        util_last = _safe_ratio(bill[:, 0], limit, limit > 0, np.nan)
        util_mean = _safe_ratio(bill, limit[:, None], limit[:, None] > 0, np.nan).mean(axis=1)
        new["STRESS_DELINQ_UTIL"] = (pay[:, 0] >= 2).astype(float) * util_last
        new["STRESS_MAXPAY_UTIL"] = pay.max(axis=1) * util_mean
        new["STRESS_CURR"] = np.clip(pay[:, 0], 0, None) * util_last

    if "paysem2" in groups:
        ever_late = pay.max(axis=1) >= 1
        new["PAY_RECOVERED"] = (ever_late & (pay[:, 0] <= 0)).astype(float)
        new["PAY_NEWLY_LATE"] = ((pay[:, 0] >= 1) & (pay[:, 1] <= 0)).astype(float)
        new["PAY_DELINQ_RECENT3"] = (pay[:, :3] >= 1).sum(axis=1).astype(float)
        new["PAY_DELINQ_OLD3"] = (pay[:, 3:] >= 1).sum(axis=1).astype(float)
        new["PAY_ESCALATION"] = pay[:, 0] - pay[:, 1:].max(axis=1)

    if "bill" in groups:
        new["BILL_MEAN"] = bill.mean(axis=1)
        new["BILL_MAX"] = bill.max(axis=1)
        new["BILL_STD"] = bill.std(axis=1)
        new["BILL_TREND"] = bill[:, 0] - bill[:, -1]
        new["REM_CREDIT_LAST"] = limit - bill[:, 0]
        new["REM_CREDIT_MEAN"] = limit - bill.mean(axis=1)

    eng = pd.DataFrame(new, index=X.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([X, eng], axis=1)


def make_feature_engineer(groups) -> FunctionTransformer:
    """A stateless transformer that appends the requested engineered features."""
    return FunctionTransformer(add_engineered_features, kw_args={"groups": list(groups)})


# Encoding variants. Stateless row-wise rewrites of the raw columns, gated by toggles in
# an encoding spec (see config.ENCODING_DEFAULTS). A monotone rescaling is a no-op for
# trees but matters for the linear model, so variants are screened on both.

# Monetary columns compressed by the signed-log1p knob (AGE excluded).
MONETARY_COLS: list[str] = ["LIMIT_BAL"] + list(config.BILL_COLS) + list(config.PAY_AMT_COLS)


def _signed_log1p(a: np.ndarray) -> np.ndarray:
    """sign(x) * log1p(|x|): compresses heavy tails while preserving sign."""
    return np.sign(a) * np.log1p(np.abs(a))


def encoding_extra_columns(encoding) -> list[str]:
    """Appended column names produced by an encoding spec (pay_flags, bill_neg: flag)."""
    if not encoding:
        return []
    names: list[str] = []
    if encoding.get("pay_flags"):
        names.extend(f"IS_DELINQ_{c}" for c in config.PAY_COLS)
    if encoding.get("bill_neg") == "flag":
        names.extend(f"OVERPAY_{c}" for c in config.BILL_COLS)
    return names


def apply_encoding(X: pd.DataFrame, encoding=None) -> pd.DataFrame:
    """Re-encode the raw columns per an encoding spec (stateless, leakage-free).

    Toggles applied in a fixed order so they compose deterministically:
    pay_remap (fold {-2,-1,0} to 0, a monotone 'months delinquent' ordinal),
    pay_flags (append per-month IS_DELINQ binaries), bill_neg (clip/flag/keep negative
    BILL), log1p (signed log1p on MONETARY_COLS). All-default is a no-op; 'scale' is
    handled in build_preprocessor.
    """
    if not encoding:
        return X

    X = X.copy()

    if encoding.get("pay_remap"):
        X[config.PAY_COLS] = X[config.PAY_COLS].clip(lower=0)

    new: dict[str, np.ndarray] = {}
    if encoding.get("pay_flags"):
        for c in config.PAY_COLS:
            new[f"IS_DELINQ_{c}"] = (X[c].to_numpy(dtype=float) >= 1).astype(float)

    bill_neg = encoding.get("bill_neg", "keep")
    if bill_neg == "clip":
        X[config.BILL_COLS] = X[config.BILL_COLS].clip(lower=0)
    elif bill_neg == "flag":
        for c in config.BILL_COLS:
            new[f"OVERPAY_{c}"] = (X[c].to_numpy(dtype=float) < 0).astype(float)

    if encoding.get("log1p"):
        X[MONETARY_COLS] = _signed_log1p(X[MONETARY_COLS].to_numpy(dtype=float))

    if new:
        eng = pd.DataFrame(new, index=X.index)
        X = pd.concat([X, eng], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def make_encoder(encoding) -> FunctionTransformer:
    """Stateless transformer that applies an encoding spec (see apply_encoding)."""
    return FunctionTransformer(apply_encoding, kw_args={"encoding": dict(encoding or {})})


def build_preprocessor(extra_numeric=(), scale: bool = True) -> ColumnTransformer:
    """ColumnTransformer: numeric (impute, optional scale) and categorical (impute + one-hot).

    'extra_numeric' are engineered / encoding-flag columns routed through the numeric
    pipe. 'scale' gates StandardScaler: keep for linear models, drop for trees.
    """
    numeric_cols = list(config.NUMERIC) + list(extra_numeric)
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, config.CATEGORICAL),
        ],
        remainder="drop",
    )
