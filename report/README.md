# report/

The IEEE 4-page report is a **required deliverable** (separate from the code).
This folder is a placeholder — no `.tex` is generated here on purpose.

## Hard rules (from the exam rules)

- **Use the official IEEE conference LaTeX template** provided on the course
  website (locally or on Overleaf).
- **Maximum 4 pages**, excluding references.
- **Do NOT modify the template** — changing fonts, margins, or using a different
  template results in **0 points** for the report.
- LLMs may be used **only to help write the report**, *not* the implementation.
  Fill in the LLM-usage declaration form by the deadline (mandatory).

## Required sections (in this order)

1. **Problem Overview** — the task, the dataset, the imbalance, macro-F1.
2. **Proposed Approach**
   1. *Preprocessing* — cleaning (undocumented code folding), imputation,
      scaling, one-hot encoding.
   2. *Model Selection* — the baselines compared and why HistGradientBoosting.
   3. *Hyperparameter Tuning* — search space, CV, macro-F1 threshold tuning.
3. **Results** — CV macro-F1 per model, per-class metrics, ROC-AUC, confusion
   matrices, leaderboard score. **All results must already be in the report**
   (graders do not regenerate them).
4. **Discussion** — error analysis, limitations, what helped.
5. **References**

Generate the numbers/figures with `python experiments/tune_baseline.py`
(metrics + `outputs/figures/cm_*.png`) and copy them into the report.
