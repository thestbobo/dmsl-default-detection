"""Entry point — runs the full pipeline end to end and writes submission.csv.

    load development + evaluation
    -> preprocess + train the single chosen pipeline FROM SCRATCH
    -> predict on the evaluation set
    -> write outputs/submissions/submission.csv  (header: Id,Predicted)

Design constraints (exam rules):
- Must run in its entirety in <= 150 seconds.
- No pre-computed artifacts: everything is built at runtime.
- All randomness is seeded (see src.config.SEED).
- Hyperparameter tuning lives in experiments/, NOT here. This script only
  runs the one "best" configuration defined in src.models.
"""

from __future__ import annotations

import time

import numpy as np

from src import config
from src.data import (
    get_eval_ids,
    get_features,
    load_development,
    load_evaluation,
    split_xy,
)
from src.models import CHOSEN_MODEL_NAME, build_chosen_pipeline
from src.submission import validate_submission, write_submission


def main() -> None:
    t0 = time.perf_counter()
    config.set_seed()

    # 1. Load data.
    dev = load_development()
    X, y = split_xy(dev)

    eval_df = load_evaluation()
    eval_ids = get_eval_ids(eval_df)
    X_eval = get_features(eval_df)

    # 2. Train the chosen pipeline from scratch (incl. the threshold search).
    model = build_chosen_pipeline()
    model.fit(X, y)

    # 3. Predict on the evaluation set.
    preds = model.predict(X_eval)

    # 4. Write and validate the submission.
    n_rows = write_submission(eval_ids, preds)
    validate_submission(config.OUT_SUB, eval_df)

    # 5. Short summary for a sanity check.
    elapsed = time.perf_counter() - t0
    threshold = getattr(model, "best_threshold_", None)
    pos = int(np.sum(preds))
    print("=" * 60)
    print("DONE — submission written")
    print(f"  model            : {CHOSEN_MODEL_NAME}")
    if threshold is not None:
        print(f"  tuned threshold  : {threshold:.4f} (macro-F1)")
    print(f"  train rows       : {len(X):,}")
    print(f"  rows written     : {n_rows:,}  -> {config.OUT_SUB}")
    print(f"  predicted defaults: {pos:,} / {n_rows:,} ({pos / n_rows:.1%})")
    print(f"  runtime          : {elapsed:.1f} s (budget: 150 s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
