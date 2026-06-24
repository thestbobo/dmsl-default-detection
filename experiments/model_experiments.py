"""Model bake-off + soft-vote ensemble preview, kept OUT of main.py.

A counterpart to 'feature_experiments.py' / 'encoding_experiments.py'. Two questions:

1. **Standalone:** how does each non-HGB sklearn model (LogReg on the
   'clean_linear' encoding, RandomForest, ExtraTrees, plain GradientBoosting) score on
   the **deployed objective** — OOF 'predict_proba' then macro-F1 at the best
   threshold, versus the deployed HGB anchor? Each model uses the encoding that suits
   it (trees: 'noscale', a monotone rescaling is a no-op for them; LogReg: the
   'clean_linear' variant). We do NOT expect a challenger to beat HGB outright.

2. **Diversity + ensemble preview:** the point of a weaker-but-different model is the
   *ensemble*. So we (a) measure each model's prediction diversity vs HGB (correlation
   of OOF probabilities + hard-prediction disagreement) and (b) preview equal-weight
   soft-vote ensembles ('experiments.ensemble_configs') by averaging the models' OOF
   probabilities at a shared CV split and threshold-tuning the average. Averaging OOF
   arrays is an honest OOF estimate because every model uses the *same* StratifiedKFold
   split at a given seed.

Discipline matches the other sweeps: screen at the primary seed, then **paired
repeated-CV** across the fold seeds, a real edge must beat the HGB anchor on *every*
seed (single-split CV gains <0.005 are noise). All OOF probabilities are computed once
per (model, seed) and cached, so the ensembles are cheap to evaluate.

Usage:
    python experiments/model_experiments.py                 # full sweep + ensembles
    python experiments/model_experiments.py --models-only    # skip the ensemble preview
    python experiments/model_experiments.py rf et            # only these model configs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src import config  # noqa: E402
from src.data import load_development, split_xy  # noqa: E402
from src.models import make_estimator, make_pipeline  # noqa: E402
from experiments.tune_baseline import best_threshold_macro_f1  # noqa: E402

# Screen at the primary seed; confirm any winner across all fold seeds.
SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
# A robust win must beat the anchor on every seed; this much mean gain marks a real
# edge rather than noise.
MIN_DELTA = 0.005
# Every challenger / ensemble is judged against this model.
ANCHOR = "hgb"


def oof_proba(spec: dict, X, y, seed: int) -> np.ndarray:
    """OOF positive-class probabilities for a model spec at one fold seed.

    The estimator gets the encoding variant named in its spec (resolved from
    'config.ENCODING_CONFIGS'); the whole pipeline is fit inside 'cross_val_predict'
    so every transform sees training folds only (no leakage).
    """
    enc = config.ENCODING_CONFIGS[spec.get("encoding", "baseline")]
    pipe = make_pipeline(make_estimator(spec["kind"], spec.get("params")), encoding=enc)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def build_oof_cache(specs: dict[str, dict], X, y, seeds) -> dict:
    """Compute & cache OOF probabilities for every (model, seed) once."""
    cache: dict[tuple[str, int], np.ndarray] = {}
    for name, spec in specs.items():
        for s in seeds:
            cache[(name, s)] = oof_proba(spec, X, y, s)
    return cache


def softvote(cache: dict, members: list[str], seed: int) -> np.ndarray:
    """Equal-weight average of the members' OOF probabilities at one seed."""
    return np.mean([cache[(m, seed)] for m in members], axis=0)


def _label(dmean: float, wins: int, n: int) -> str:
    if wins == n and dmean >= MIN_DELTA:
        return "KEEP (robust)"
    if wins == n and dmean > 0:
        return "keep? (robust but small)"
    if dmean > 0:
        return "watch (not all seeds)"
    return "revert"


def screen_models(specs: dict[str, dict], cache: dict, y) -> dict[str, float]:
    """Standalone OOF macro-F1 @best threshold at the screen seed."""
    print(f"\n{'=' * 64}\n  STANDALONE MODELS (deployed objective, seed {SCREEN_SEED})\n{'=' * 64}")
    base_thr, base_f1 = best_threshold_macro_f1(cache[(ANCHOR, SCREEN_SEED)], y)
    print(f"  {ANCHOR:14s} {base_f1:.4f} @thr {base_thr:.3f}   (anchor)")
    scores = {ANCHOR: base_f1}
    for name in specs:
        if name == ANCHOR:
            continue
        thr, f1 = best_threshold_macro_f1(cache[(name, SCREEN_SEED)], y)
        scores[name] = f1
        print(f"  {name:14s} {f1:.4f} @thr {thr:.3f}   d={f1 - base_f1:+.4f}")
    return scores


def diversity_vs_anchor(specs: dict[str, dict], cache: dict, y) -> None:
    """Correlation + hard-prediction disagreement of each model vs the HGB anchor.

    A model that is *both* decent and decorrelated from HGB is the best ensemble
    partner, a near-clone (high corr, low disagreement) adds nothing to a soft-vote.
    """
    print(f"\n--- Diversity vs {ANCHOR} (seed {SCREEN_SEED}): lower corr / higher disagree = better partner ---")
    p_anchor = cache[(ANCHOR, SCREEN_SEED)]
    thr_a, _ = best_threshold_macro_f1(p_anchor, y)
    pred_a = (p_anchor >= thr_a).astype(int)
    for name in specs:
        if name == ANCHOR:
            continue
        p = cache[(name, SCREEN_SEED)]
        thr, _ = best_threshold_macro_f1(p, y)
        corr = float(np.corrcoef(p_anchor, p)[0, 1])
        disagree = float(np.mean((p >= thr).astype(int) != pred_a))
        print(f"  {name:14s} corr={corr:+.3f}   pred-disagreement={disagree:.3f}")


def screen_ensembles(ensembles: dict[str, list[str]], cache: dict, y, anchor_f1: float) -> dict[str, float]:
    """Equal-weight soft-vote OOF macro-F1 @best threshold at the screen seed."""
    print(f"\n{'=' * 64}\n  EQUAL-WEIGHT SOFT-VOTE PREVIEW (seed {SCREEN_SEED})\n{'=' * 64}")
    print(f"  {ANCHOR + ' (anchor)':22s} {anchor_f1:.4f}")
    scores: dict[str, float] = {}
    for name, members in ensembles.items():
        thr, f1 = best_threshold_macro_f1(softvote(cache, members, SCREEN_SEED), y)
        scores[name] = f1
        print(f"  {name:22s} {f1:.4f} @thr {thr:.3f}   d={f1 - anchor_f1:+.4f}   {members}")
    return scores


def validate_paired(named_probas, cache: dict, X, y) -> None:
    """Paired repeated-CV of each candidate vs the HGB anchor across the fold seeds.

    'named_probas' maps a candidate name -> a callable(seed) returning its OOF
    probabilities (a single model from the cache, or a soft-vote average). A candidate
    is only trustworthy if it beats the anchor on *every* seed.
    """
    print(f"\n--- Paired repeated-CV validation ({len(VALIDATION_SEEDS)} seeds) vs {ANCHOR} ---")
    base = np.array([best_threshold_macro_f1(cache[(ANCHOR, s)], y)[1] for s in VALIDATION_SEEDS])
    print(f"  {ANCHOR:22s} {base.mean():.4f} +/- {base.std():.4f}   {np.round(base, 4).tolist()}")
    results = []
    for name, proba_fn in named_probas.items():
        fs = np.array([best_threshold_macro_f1(proba_fn(s), y)[1] for s in VALIDATION_SEEDS])
        d = fs - base
        wins = int((d > 0).sum())
        results.append((float(d.mean()), wins, name, fs))
        print(f"  {name:22s} {fs.mean():.4f} +/- {fs.std():.4f}   "
              f"dmean={d.mean():+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   {np.round(fs, 4).tolist()}")

    results.sort(reverse=True)
    print(f"\n--- ranked by paired mean delta vs {ANCHOR} ---")
    for dmean, wins, name, _ in results:
        print(f"  {name:22s} dmean={dmean:+.4f}   wins {wins}/{len(VALIDATION_SEEDS)}   "
              f"-> {_label(dmean, wins, len(VALIDATION_SEEDS))}")


def main() -> None:
    config.set_seed()

    args = sys.argv[1:]
    models_only = "--models-only" in args
    args = [a for a in args if a != "--models-only"]

    names = args or list(config.MODEL_CONFIGS)
    if ANCHOR not in names:
        names = [ANCHOR] + names  # always include the anchor
    unknown = [n for n in names if n not in config.MODEL_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; known: {list(config.MODEL_CONFIGS)}")
    specs = {n: config.MODEL_CONFIGS[n] for n in names}

    dev = load_development()
    X, y = split_xy(dev)
    print(f"Loaded development set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Models: {list(specs)}")

    # Compute every OOF probability once (models x seeds); everything below reuses it.
    cache = build_oof_cache(specs, X, y, VALIDATION_SEEDS)

    model_f1 = screen_models(specs, cache, y)
    diversity_vs_anchor(specs, cache, y)

    # Candidates for paired-CV: any standalone model that beat the anchor at the screen.
    anchor_f1 = model_f1[ANCHOR]
    candidates = {
        n: (lambda s, n=n: cache[(n, s)])
        for n, f1 in model_f1.items()
        if n != ANCHOR and f1 > anchor_f1
    }

    if not models_only:
        ensembles = {
            name: members for name, members in config.ENSEMBLE_CONFIGS.items()
            if all(m in specs for m in members)  # only ensembles whose members were run
        }
        ens_f1 = screen_ensembles(ensembles, cache, y, anchor_f1)
        for name, f1 in ens_f1.items():
            if f1 > anchor_f1:
                members = ensembles[name]
                candidates[name] = lambda s, m=members: softvote(cache, m, s)

    if candidates:
        validate_paired(candidates, cache, X, y)
    else:
        print(f"\nNothing beat the {ANCHOR} anchor at the screen seed, nothing to validate.")


if __name__ == "__main__":
    main()
