"""Stacking bake-off (Path P8) — kept OUT of main.py.

P4 only tried an EQUAL-WEIGHT soft-vote of diverse members; it scored 0.710 on the
LB (< S0 0.712) and was reverted. This script tries the natural next refinement that
E12's notes flagged: a ``StackingClassifier`` with a logistic meta-learner. Instead
of averaging the members' probabilities, the meta-learner is *trained* on their
out-of-fold ``predict_proba`` and can therefore learn member weights / interactions
(e.g. trust the linear model only in the region where it disagrees with the trees).

Each candidate in ``config.yaml -> experiments.stacking_configs`` lists ``members``
(named ``model_configs`` entries — every member is a full encoded pipeline, so the
stack reuses the byte-identical P3 pipelines) and a ``meta`` final estimator. The
scoring protocol is the deployed objective used everywhere else: outer OOF
``predict_proba`` (``cross_val_predict``), best macro-F1 over the project threshold
grid, then paired repeated-CV across the fixed fold seeds. The anchor is the deployed
champion ``rf_balanced`` (not the weaker hgb), and the +0.005 bar from Lesson L1
applies — a stacking gain below that is noise on this leaderboard.

Usage:
    python experiments/stacking_experiments.py
    python experiments/stacking_experiments.py stack_rfbal_logreg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import _member_pipeline, make_estimator, make_pipeline  # noqa: E402
from experiments.tune_baseline import best_threshold_macro_f1  # noqa: E402

# Screen at the primary seed; confirm any winner across all fold seeds.
SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
MIN_DELTA = 0.005
# Every challenger is judged against the DEPLOYED champion, not the weaker hgb anchor.
ANCHOR = "rf_balanced"


def _make_meta(key: str, seed: int):
    """Resolve a ``meta`` key to the stacking final estimator."""
    if key == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if key == "logreg_balanced":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    raise ValueError(f"unknown meta {key!r} (expected 'logreg' / 'logreg_balanced')")


def make_stacking(spec: dict, seed: int) -> StackingClassifier:
    """Build a StackingClassifier from a stacking_configs spec at one fold seed.

    Members are the named model_configs pipelines (encoding included). The stack's
    internal cv (for the meta-features) uses the same StratifiedKFold seed as the
    outer scoring, so the whole thing is reproducible and honest.
    """
    estimators = [(name, _member_pipeline(name)) for name in spec["members"]]
    inner_cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=_make_meta(spec.get("meta", "logreg"), seed),
        stack_method="predict_proba",
        cv=inner_cv,
        passthrough=bool(spec.get("passthrough", False)),
        n_jobs=-1,
    )


def champion_oof(X, y, seed: int) -> np.ndarray:
    """OOF positive-class probabilities for the deployed champion at one seed."""
    spec = config.MODEL_CONFIGS[ANCHOR]
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    pipe = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def stacking_oof(spec: dict, X, y, seed: int) -> np.ndarray:
    """OOF positive-class probabilities for a stacking spec at one seed."""
    stack = make_stacking(spec, seed)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(stack, X, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]


def _score_proba(proba: np.ndarray, y) -> tuple[float, float]:
    return best_threshold_macro_f1(proba, y)


def _label(dmean: float, wins: int, n: int) -> str:
    if wins == n and dmean >= MIN_DELTA:
        return "KEEP (robust)"
    if wins == n and dmean > 0:
        return "keep? (robust but small)"
    if dmean > 0:
        return "watch (not all seeds)"
    return "revert"


def main() -> None:
    config.set_seed()

    args = sys.argv[1:]
    names = args or list(config.STACKING_CONFIGS)
    unknown = [n for n in names if n not in config.STACKING_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown stacking config(s) {unknown}; known: {list(config.STACKING_CONFIGS)}")

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Candidates: {names}")

    print(f"\n{'=' * 64}\n  P8 — STACKING (deployed objective, seed {SCREEN_SEED})\n{'=' * 64}")
    base_thr, base_f1 = _score_proba(champion_oof(X, y, SCREEN_SEED), y)
    print(f"  {ANCHOR:30s} {base_f1:.4f} @thr {base_thr:.3f}   (champion)")

    screened: dict[str, float] = {}
    for name in names:
        spec = config.STACKING_CONFIGS[name]
        thr, f1 = _score_proba(stacking_oof(spec, X, y, SCREEN_SEED), y)
        screened[name] = f1
        print(f"  {name:30s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}   {spec['members']} -> {spec.get('meta', 'logreg')}")

    winners = [n for n in names if screened[n] > base_f1]
    if not winners:
        print(f"\nNothing beat the {ANCHOR} champion at the screen seed — nothing to validate.")
        return

    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {ANCHOR} ---")
    base = np.array([_score_proba(champion_oof(X, y, s), y)[1] for s in VALIDATION_SEEDS])
    print(f"  {ANCHOR:30s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")

    results = []
    for name in winners:
        spec = config.STACKING_CONFIGS[name]
        fs = np.array([_score_proba(stacking_oof(spec, X, y, s), y)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(f"  {name:30s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")

    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {ANCHOR} ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:30s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


if __name__ == "__main__":
    main()
