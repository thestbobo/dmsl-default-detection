"""Item 2 / Path X — Real randomized hyperparameter search for HGB *and* RF.

Path X (re-tune the deployed model) was in the tracker as `planned` but never run:
S0 HGB is at library defaults and the deployed RF uses hand-picked
(n_estimators=300, min_samples_leaf=20) knobs that were never searched. This script
runs a randomized search on the DEPLOYED objective (OOF macro-F1 at the best threshold,
the same objective tune_baseline.py uses) for both families, validates the top few with
paired repeated-CV, and — per the 2026-06-21 review — emits the top candidates of each
family as numbered submissions REGARDLESS of the CV verdict (the LB is the judge).

    python experiments/path_x_search_submit.py            # full search + submit
    python experiments/path_x_search_submit.py --no-submit # search only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV  # noqa: E402

from experiments._submit_utils import best_threshold_macro_f1, emit_submission, oof_proba  # noqa: E402
from src import config  # noqa: E402
from src.data import (  # noqa: E402
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import make_estimator, make_pipeline  # noqa: E402

SCREEN_SEED = config.SEED
VALIDATION_SEEDS = config.VALIDATION_SEEDS
N_SEARCH_HGB = 40
N_SEARCH_RF = 30
N_TOP = 6       # paired-CV validate this many
N_SUBMIT = 3    # emit this many submissions per family
NOSCALE = config.ENCODING_CONFIGS["noscale"]


def _sample_hgb(rng: np.random.Generator) -> dict:
    """HGB config from a regularisation-focused space (mirrors tune_baseline.sample_params)."""
    return dict(
        learning_rate=float(round(10 ** rng.uniform(-2, -0.8), 4)),
        max_iter=int(rng.choice([200, 300, 400, 500, 600])),
        max_leaf_nodes=int(rng.choice([15, 21, 31, 45, 63])),
        min_samples_leaf=int(rng.choice([20, 40, 60, 100, 150, 200])),
        l2_regularization=float(round(10 ** rng.uniform(-2, 1), 3)),
        max_features=float(round(rng.uniform(0.6, 1.0), 2)),
        max_bins=int(rng.choice([128, 255])),
    )


def _pick(rng: np.random.Generator, options: list):
    """Pick one element by integer index (avoids numpy dtype coercion of mixed lists)."""
    return options[int(rng.integers(len(options)))]


def _sample_rf(rng: np.random.Generator) -> dict:
    """RF config; class_weight stays 'balanced' (the champion direction, L9)."""
    return dict(
        n_estimators=_pick(rng, [300, 500, 800]),
        max_depth=_pick(rng, [None, 10, 16, 24, 32]),
        max_features=_pick(rng, ["sqrt", 0.3, 0.5, 0.7]),
        min_samples_leaf=_pick(rng, [5, 10, 20, 40]),
        min_samples_split=_pick(rng, [2, 5, 10]),
        max_samples=_pick(rng, [None, 0.7, 0.9]),
        criterion=_pick(rng, ["gini", "entropy"]),
        class_weight="balanced",
        n_jobs=-1,
    )


def _pipe(kind: str, params: dict):
    return make_pipeline(make_estimator(kind, params), encoding=NOSCALE)


def _score(kind: str, params: dict, X, y, seed: int) -> tuple[float, float]:
    # RF parallelises internally (n_jobs=-1) -> run CV folds sequentially to avoid
    # nested-parallelism oversubscription; HGB has no internal parallelism so let CV
    # use all cores.
    cv_n_jobs = 1 if kind in ("rf", "et") else -1
    return best_threshold_macro_f1(oof_proba(_pipe(kind, params), X, y, seed, cv_n_jobs=cv_n_jobs), y)


def _search(kind: str, sampler, n_search: int, X, y) -> list[tuple[float, float, dict]]:
    rng = np.random.default_rng(config.SEED)
    print(f"\n--- {kind.upper()} randomized search ({n_search} configs, deployed objective) ---")
    results: list[tuple[float, float, dict]] = []
    seen: set[str] = set()
    while len(results) < n_search:
        params = sampler(rng)
        key = repr(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        thr, f1 = _score(kind, params, X, y, SCREEN_SEED)
        results.append((f1, thr, params))
    results.sort(key=lambda r: -r[0])
    print(f"  top {N_TOP} by seed-{SCREEN_SEED} macro-F1:")
    for f1, thr, params in results[:N_TOP]:
        print(f"    {f1:.4f} @thr {thr:.3f}  {params}")
    return results


def _validate(kind: str, top: list[dict], anchor_f1_by_seed: np.ndarray, X, y) -> None:
    print(f"\n--- {kind.upper()} paired repeated-CV ({len(VALIDATION_SEEDS)} seeds) vs anchor ---")
    print(f"  anchor   {anchor_f1_by_seed.mean():.4f} +/- {anchor_f1_by_seed.std():.4f}")
    for i, params in enumerate(top):
        fs = np.array([_score(kind, params, X, y, s)[1] for s in VALIDATION_SEEDS])
        d = fs - anchor_f1_by_seed
        wins = int((d > 0).sum())
        print(f"  cand{i}    {fs.mean():.4f} +/- {fs.std():.4f}  dmean={d.mean():+.4f}  "
              f"wins {wins}/{len(VALIDATION_SEEDS)}")


def _tuned(estimator):
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return TunedThresholdClassifierCV(
        estimator=estimator, scoring="f1_macro", cv=cv, refit=True, random_state=config.SEED
    )


def main() -> None:
    config.set_seed()
    args = sys.argv[1:]
    submit = "--no-submit" not in args

    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  base default rate={y.mean():.1%}")

    do_hgb = "--rf-only" not in args
    do_rf = "--hgb-only" not in args
    families = []
    if do_hgb:
        hgb_anchor = np.array([_score("hgb", {}, X, y, s)[1] for s in VALIDATION_SEEDS])
        print(f"\nanchor: hgb_default={hgb_anchor.mean():.4f}")
        families.append(("hgb", _sample_hgb, N_SEARCH_HGB, hgb_anchor))
    if do_rf:
        rf_spec = config.MODEL_CONFIGS["rf_balanced"]["params"]
        rf_anchor = np.array(
            [_score("rf", rf_spec, X, y, s)[1] for s in VALIDATION_SEEDS]
        )
        print(f"\nanchor: rf_balanced={rf_anchor.mean():.4f}")
        families.append(("rf", _sample_rf, N_SEARCH_RF, rf_anchor))

    for kind, sampler, n_search, anchor in families:
        ranked = _search(kind, sampler, n_search, X, y)
        top = [p for _, _, p in ranked[:N_TOP]]
        _validate(kind, top, anchor, X, y)

        if submit:
            print(f"\n  emitting top {N_SUBMIT} {kind.upper()} candidates as submissions:")
            for i, params in enumerate(ranked[:N_SUBMIT]):
                f1, thr, p = params
                model = _tuned(_pipe(kind, p))
                model.fit(X, y)
                preds = model.predict(X_eval)
                emit_submission(
                    eval_ids, eval_df, preds,
                    description=f"Path X {kind} tuned cand{i} (OOF {f1:.4f}): {p}",
                    oof_f1=f1,
                )


if __name__ == "__main__":
    main()
