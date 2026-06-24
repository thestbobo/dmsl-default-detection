"""Path P14 / external-review item 5 — seed-averaging the deployed winner.

The LB deltas that decide this competition (0.001-0.002) are within run-to-run seed
variance. Averaging `predict_proba` across many seeds of the SAME model before thresholding
shrinks that variance — cheap insurance that de-risks the final 2-submission pick (and tests
whether the 0.719 FE-on-RF win is seed-stable rather than a lucky split).

Model = the deployed champion: rf_balanced + paysem_util_payratio (LB 0.719). For each seed
we (a) re-fit on full dev with that RF random_state -> eval proba, and (b) compute that seed's
OOF proba on dev. We average each across seeds, pick the macro-F1 threshold on the averaged
dev OOF (the deployed objective), and apply it to the averaged eval proba.

    python experiments/seed_average_submit.py            # seed-average + submit
    python experiments/seed_average_submit.py --no-submit # report only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

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

NOSCALE = config.ENCODING_CONFIGS["noscale"]
RF_PARAMS = dict(config.MODEL_CONFIGS["rf_balanced"]["params"])
GROUPS = config.FEATURE_CONFIGS["paysem_util_payratio"]  # the deployed winner's features
SEEDS = [42, 11, 7, 1, 2024, 99, 123, 2025]  # the proba-averaging ensemble of seeds


def _pipe(seed: int):
    params = {**RF_PARAMS, "random_state": seed}
    return make_pipeline(make_estimator("rf", params), feature_groups=GROUPS, encoding=NOSCALE)


def main() -> None:
    config.set_seed()
    submit = "--no-submit" not in sys.argv[1:]

    dev = load_development()
    X, y = split_xy(dev)
    eval_df = load_evaluation()
    X_eval = get_features(eval_df)
    eval_ids = get_eval_ids(eval_df)
    print(f"dev rows={len(y):,}  base default rate={y.mean():.1%}")
    print(f"model: rf_balanced + {GROUPS}; averaging proba over {len(SEEDS)} seeds {SEEDS}")

    oof_sum = np.zeros(len(y))
    eval_sum = np.zeros(len(X_eval))
    for i, seed in enumerate(SEEDS, 1):
        pipe = _pipe(seed)
        oof_sum += oof_proba(pipe, X, y, seed, cv_n_jobs=1)
        pipe.fit(X, y)
        eval_sum += pipe.predict_proba(X_eval)[:, 1]
        # Running view of the averaged OOF macro-F1 as seeds accumulate.
        thr_i, f1_i = best_threshold_macro_f1(oof_sum / i, y)
        print(f"  +seed {seed:>4}  ({i}/{len(SEEDS)})  avg-OOF macro-F1={f1_i:.4f} @thr {thr_i:.3f}")

    oof_avg = oof_sum / len(SEEDS)
    eval_avg = eval_sum / len(SEEDS)
    thr, f1 = best_threshold_macro_f1(oof_avg, y)
    preds = (eval_avg >= thr).astype(int)
    print(f"\nseed-averaged: OOF macro-F1={f1:.4f} @thr {thr:.3f}  "
          f"eval defaults={int(preds.sum())}/{len(preds)} ({preds.mean():.1%})")

    if submit:
        emit_submission(
            eval_ids, eval_df, preds,
            description=f"SEED-AVG ({len(SEEDS)} seeds) rf_balanced + paysem_util_payratio @thr {thr:.3f}",
            oof_f1=f1,
        )


if __name__ == "__main__":
    main()
