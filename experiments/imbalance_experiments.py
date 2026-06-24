"""Imbalance-handling sweep, kept out of main.py.

Threshold tuning already handles the decision cut; this tests the next lever, changing
the fitted loss with class_weight. Each candidate in experiments.imbalance_configs
starts from a model_configs entry and overrides estimator params. Scored on the deployed
objective with screen + paired repeated-CV, like the other sweeps.

Usage:
    python experiments/imbalance_experiments.py
    python experiments/imbalance_experiments.py hgb_w15 hgb_w20
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from experiments._submit_utils import best_threshold_macro_f1, verdict  # noqa: E402
from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402

ANCHOR = "hgb"
SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS


def _merged_spec(candidate: dict) -> dict:
    """Resolve a candidate onto its base model spec."""
    base_name = candidate["base_model"]
    if base_name not in config.MODEL_CONFIGS:
        raise ValueError(f"unknown base_model {base_name!r}; known: {list(config.MODEL_CONFIGS)}")
    base = dict(config.MODEL_CONFIGS[base_name])
    params = dict(base.get("params") or {})
    params.update(candidate.get("params") or {})
    base["params"] = params
    return base


def oof_proba(spec: dict, X, y, seed: int) -> np.ndarray:
    """OOF positive-class probabilities for one resolved model spec."""
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    pipe = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def _score(spec: dict, X, y, seed: int) -> tuple[float, float]:
    return best_threshold_macro_f1(oof_proba(spec, X, y, seed), y)


def main() -> None:
    config.set_seed()

    args = sys.argv[1:]
    names = args or list(config.IMBALANCE_CONFIGS)
    unknown = [n for n in names if n not in config.IMBALANCE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown imbalance config(s) {unknown}; known: {list(config.IMBALANCE_CONFIGS)}")

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Candidates: {names}")

    anchor_spec = config.MODEL_CONFIGS[ANCHOR]
    cand_specs = {n: _merged_spec(config.IMBALANCE_CONFIGS[n]) for n in names}

    print(f"\n{'=' * 64}\n  IMBALANCE HANDLING (class_weight, seed {SCREEN_SEED})\n{'=' * 64}")
    base_thr, base_f1 = _score(anchor_spec, X, y, SCREEN_SEED)
    print(f"  {ANCHOR:14s} {base_f1:.4f} @thr {base_thr:.3f}   (anchor)")

    screened = {}
    for name, spec in cand_specs.items():
        thr, f1 = _score(spec, X, y, SCREEN_SEED)
        screened[name] = f1
        print(f"  {name:14s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}   {spec.get('params') or {}}")

    candidates = {n: s for n, s in cand_specs.items() if screened[n] > base_f1}
    if not candidates:
        print(f"\nNothing beat the {ANCHOR} anchor at the screen seed; nothing to validate.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {ANCHOR} ---")
    base = np.array([_score(anchor_spec, X, y, s)[1] for s in VALIDATION_SEEDS])
    print(f"  {ANCHOR:14s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")

    results = []
    for name, spec in candidates.items():
        fs = np.array([_score(spec, X, y, s)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(f"  {name:14s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")

    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {ANCHOR} ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:14s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {verdict(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
