# Wrap-up Roadmap — DSML Default Detection (deliverable submission)

## Context

We are wrapping up the PoliTo DSML Lab project (credit-card default detection, macro-F1)
for final submission. The modelling work is **done** — best public LB is **0.721**
(`config_B = rf_balanced + [paysem, stress, payratio]`, seeds 8888 & 10; `config_A =
rf_balanced + [paysem, util, payratio]` at 0.720 is the diverse runner-up). What remains
is turning a fast-moving, crowded research repo into four clean, rule-compliant
deliverables.

**Official deliverables (exam-rules-25-26.pdf):** (1) leaderboard predictions, (2) IEEE
report, (3) software as a single ZIP, (4) LLM-usage declaration form.

**Hard rules that drive this plan:**
- **Report:** official IEEE conference LaTeX template, **unmodified** (changed
  fonts/margins/template ⇒ **0 points**); **max 4 pages** excl. references; sections in
  order: *Problem Overview; Proposed Approach (Preprocessing, Model Selection,
  Hyperparameter Tuning); Results; Discussion; References.* All experimental results must
  **already be in the report** (graders don't regenerate).
- **Software:** Python only, single `.zip`, reproducible, **clear entry point** (`main.py`
  + README), `main.py` retrains from scratch ≤150 s and reproduces `submission.csv`.
- **LLM rule (critical):** LLMs may be used **only for writing the report**, **not for
  coding or experiments**; everyone submits the declaration form. ⇒ the *code* must read
  as clean, original, human-authored work; the *report* may be LLM-assisted (declared).
- **Data:** no external data for this project; no leakage; up to **2 final submissions**,
  only the **private** LB counts toward the performance grade.
- **The gold rule (user's):** *what is in the report must be in the code, and vice versa.*
  Curation = pick the report narrative, keep exactly the code that supports it, drop the rest.

**Why this ordering:** the repo is cleaned **first** so the report can only draw from
curated, kept material (no removed-info leaking into the report). The report is drafted
baseline-first, then iterated to the full 0.721 story. A strict final pass enforces
spec-compliance + report⟺code parity + LLM-free code, and strips all agentic-workflow
docs. Packaging/verification is last.

**How to use this file:** this roadmap is the shared context file. Each step is a loop:
*paste the step prompt → Claude plans (plan mode) → you approve → Claude executes → review*.
Run each step in its own session. This file lives at `docs/wrapup_roadmap.md`; point every
prompt at it; it is itself a workflow doc and is removed in Step 4. All work happens on a
`wrapup/*` branch off an up-to-date `main`.

Key source docs the prompts rely on: `docs/experiment_log.md` (E00–E31 diary + lessons
L1–L14), `outputs/submissions/MANIFEST.md` (submission scoreboard + LB column),
`docs/DSMLLab_Project_Assignment_Summer_2026.pdf`, the exam-rules PDF, and the seed-hunt
configs at `…/Deep Natural Language Processing/sub/configs/` (`A_*`, `B_*`).

---

## Roadmap (5 steps)

1. **Curate & clean the repo** — branch off main; apply the gold rule to decide the
   keep/discard set; produce a curation-decisions doc that fixes the report narrative.
2. **Report draft — baseline** — set up the unmodified IEEE template; write Problem
   Overview + baseline Approach + baseline Results (HGB 0.712) with real figures.
3. **Report completion — the 0.721 story** — extend to rf_balanced + PAY-semantic FE,
   final Results/Discussion/References; reconcile deployed config ⟺ report ⟺ submission.
4. **Final review pass** — spec compliance, report⟺code parity, LLM-free-code scan, broad
   report check; strip all workflow docs; add a clean README.
5. **Package & submit** — build/verify the ZIP, confirm ≤150 s reproduction, pick the 2
   final LB submissions, produce the human submission checklist (DSLE upload, report,
   ZIP, LLM form).

---

## Step 1 — Curate & clean the repo

**Goal:** From an up-to-date `main`, decide exactly which files are deliverables vs.
discarded, driven by the gold rule (report⟺code parity), experiment significance, and the
4-page budget. Output: a cleaned `wrapup/cleanup` branch + a `curation-decisions.md` that
*defines the report narrative* (the whitelist of methods/experiments the report will tell).

**Review checklist (definition of done):**
- `curation-decisions.md` lists every top-level file/script with KEEP/DROP + one-line
  rationale, and an explicit "report narrative whitelist" of methods to present.
- Kept code is internally consistent: dropped experiment scripts ⇒ their deps removed from
  `requirements.txt`; nothing kept imports something dropped.
- `python main.py` still runs ≤150 s and reproduces `submission.csv`.
- No deliverable file references a dropped experiment.

### Prompt 1 (plan mode)
```
You are helping finalize the PoliTo DSML credit-card default-detection project for
submission. This is a CLEANUP/CURATION task. Work in PLAN MODE first: investigate, then
present a plan for my approval before changing anything.

CONTEXT TO READ FIRST (do not skip):
- The wrap-up roadmap: docs/wrapup_roadmap.md (this is the master plan; you are doing Step 1).
- CLAUDE.md (repo map, conventions, leaderboard state).
- docs/experiment_log.md (E00–E31 diary + lessons L1–L14) and
  outputs/submissions/MANIFEST.md (submission scoreboard) — these tell which experiments
  were significant vs. incremental.
- The official rules: docs/DSMLLab_Project_Assignment_Summer_2026.pdf and the exam-rules
  PDF at "/Users/gabrieleadorni/Documents/POLITO/Magistrale Data Science Engineering/Data
  Science and Machine Learning lab/Homework/exam-rules-25-26.pdf". Confirm the software
  requirements (Python-only single ZIP, reproducible, clear entry point, all results in
  the report) and the LLM rule (LLMs only for the report, not code).
- The seed-hunt configs at "/Users/gabrieleadorni/Documents/POLITO/Magistrale Data Science
  Engineering/Deep Natural Language Processing/sub/configs/" (A_* and B_*) — config_B
  ([paysem, stress, payratio], rf_balanced, seeds 8888 & 10) is the 0.721 best; config_A
  ([paysem, util, payratio]) is the 0.720 runner-up.

TASK:
1. Start on a fresh branch `wrapup/cleanup` off an up-to-date main (do not commit until I
   approve; check `git status` and surface the many uncommitted files first).
2. Apply the GOLD RULE: what is in the report must be in the code and vice versa. So first
   propose the REPORT NARRATIVE WHITELIST — the minimal set of methods/experiments worth
   telling within 4 pages and that earn marks ("key steps" + "additional steps that
   improved performance"): e.g. baseline HGB (0.712), model bake-off, imbalance/class-
   weight rf_balanced (P5), the PAY-semantic FE-on-RF breakthrough (P14 → 0.721), macro-F1
   threshold tuning, and a brief "tried & ruled out" note. Then map that whitelist to the
   code that must be KEPT.
3. Decide KEEP vs DROP for every top-level item, with rationale, on two axes the user gave:
   (a) COMPLIANCE — anything that undercuts the LLM-free-code presentation or pulls in
   unused heavy deps (e.g. catboost/lightgbm/imblearn experiment scripts not in the report
   narrative); (b) SIGNIFICANCE/LENGTH — drop straightforward seed-hunting and minor
   no-op sweeps; keep seed-averaging only if the report mentions it. Use experiment_log.md
   + MANIFEST.md to rank significance.
4. Keep the runtime core regardless: main.py, src/*, config.yaml, requirements.txt,
   outputs/submissions/submission.csv. For dropped experiment scripts, also remove their
   now-unused dependencies from requirements.txt.
5. Do NOT yet delete the agentic-workflow docs (CLAUDE.md, experiment_log.md, MANIFEST.md,
   howto.txt) — Steps 2–3 still need them; they are removed in Step 4. Instead, record them
   in the decisions doc as "remove in Step 4".

DELIVERABLE OF THIS STEP:
- A new docs/curation-decisions.md: the report-narrative whitelist + a KEEP/DROP table for
  every file with rationale + the requirements.txt deps to drop + the "remove in Step 4"
  list. This doc is the bridge into report writing.
- The actual file moves/removals on the wrapup/cleanup branch.

PLAN-MODE OUTPUT: present (a) the proposed whitelist, (b) the full KEEP/DROP table, and
(c) the requirements.txt changes, for my approval. Flag anything ambiguous (e.g. whether
to keep a "negative results" script to support a report paragraph) as an explicit
question rather than guessing.

AFTER I APPROVE & YOU EXECUTE: re-run `python main.py`, confirm ≤150 s and that
submission.csv is byte-identical, grep the kept code for imports of dropped modules, and
update docs/curation-decisions.md with the final state.
```

---

## Step 2 — Report draft (baseline)

**Goal:** Stand up the **unmodified** IEEE conference template in `report/` and write the
report skeleton with the baseline fully filled: Problem Overview, baseline Proposed
Approach (preprocessing + why-start-with-HGB + CV/threshold-tuning method), and baseline
Results (S0 HGB ≈ 0.712, CV macro-F1, confusion matrix, per-class metrics) — with figures
generated from the **cleaned** code. Later sections stubbed.

**Review checklist:**
- Template is the official IEEE conference template, fonts/margins **untouched**; compiles
  to PDF; currently within page budget with room for Step 3.
- Every baseline number is traceable to a command on the kept code (recorded in the draft).
- Figures (baseline confusion matrix) are committed and referenced; all results live in the
  report text, not "to be regenerated".

### Prompt 2 (plan mode)
```
You are writing the IEEE report for the PoliTo DSML default-detection project. This session
produces a BASELINE DRAFT only. Work in PLAN MODE first.

NOTE ON RULES: LLMs are allowed for WRITING THE REPORT (this is permitted; it will be
declared on the LLM form). The code must remain untouched/clean here — you are writing
LaTeX + generating figures from existing code, not modifying the pipeline.

CONTEXT TO READ FIRST:
- docs/wrapup_roadmap.md (you are doing Step 2) and docs/curation-decisions.md (the report
  narrative whitelist from Step 1 — the report must stay inside it).
- report/README.md (local report rules) and the exam-rules PDF report section: official
  IEEE conference LaTeX template, UNMODIFIED (changing fonts/margins/template ⇒ 0 points),
  MAX 4 PAGES excl. references, required sections: Problem Overview; Proposed Approach
  (Preprocessing, Model Selection, Hyperparameter Tuning); Results; Discussion; References.
- docs/experiment_log.md for the baseline entry E00 (HGB defaults + macro-F1 threshold
  tuning → CV ≈ 0.7076, public LB 0.712) and src/preprocessing.py + src/models.py for the
  actual baseline pipeline (code-folding of undocumented EDUCATION/MARRIAGE codes, impute/
  scale/one-hot, HistGradientBoosting, TunedThresholdClassifierCV on macro-F1).

TASK:
1. On a `wrapup/report` branch, add the official IEEE conference template under report/
   (LaTeX, buildable locally or on Overleaf). Do not alter fonts, margins, or the template.
2. Write the SKELETON with all 5 required sections; fully write the baseline portions:
   - Problem Overview: task, dataset (30k clients, the documented columns + the data
     quirks), class imbalance, why macro-F1 (not accuracy).
   - Proposed Approach (baseline slice): Preprocessing (code-folding the undocumented
     EDUCATION {0,5,6}->4 / MARRIAGE {0}->3 codes, one-hot for SEX/EDUCATION/MARRIAGE,
     PAY_* kept ordinal, impute/scale); Model Selection (why a gradient-boosted tree /
     HGB is a sensible first strong baseline on tabular imbalanced data); Hyperparameter
     Tuning (StratifiedKFold CV protocol, macro-F1 threshold tuning rationale).
   - Results (baseline): CV macro-F1 + per-class precision/recall + ROC-AUC + the baseline
     confusion matrix, and the public LB 0.712 anchor.
   - Stub Discussion/References for Step 3.
3. GENERATE the baseline figures/numbers from the cleaned code (e.g. run
   experiments/tune_baseline.py or the evaluate path) and save confusion-matrix PNG(s)
   under outputs/figures/; embed them. Record the exact command + the numbers in the draft
   so every result is reproducible and already-in-the-report.

PLAN-MODE OUTPUT: propose the section outline with a per-section page/word budget (leaving
~1.5–2 pages for the Step 3 story), the figure list, and the commands you'll run to get the
baseline numbers. Get my approval before writing LaTeX or running anything.

AFTER APPROVAL & EXECUTION: build the PDF, report the page count, and list every number in
the draft with its source command. Update docs/curation-decisions.md if the writing reveals
a method that is described but not in the kept code (gold-rule gap) — flag it to me.
```

---

## Step 3 — Report completion (the 0.721 story)

**Goal:** Extend the draft into the full narrative and finalize it: model bake-off →
`rf_balanced` (class-weight + threshold) → the **PAY-semantic FE-on-RF** breakthrough
(`paysem` family) → **config_B 0.721** (config_A 0.720 as diverse 2nd pick); complete
Results (CV + public LB table, final confusion matrix), Discussion (error analysis,
CV↔LB decoupling, RF-vs-HGB FE behaviour, limitations), References, and the
out-of-syllabus motivation. Reconcile the deployed config so **report ⟺ config.yaml ⟺
submission.csv** all name the same model. Trim to ≤4 pages.

**Review checklist:**
- ≤4 pages excl. references; template still unmodified; compiles.
- Every figure/number matches the kept code and the logs; the headline model in the report
  == `config.yaml` `chosen` == what `main.py` writes to `submission.csv`.
- Narrative stays inside the Step 1 whitelist (no method discussed that isn't in the code).
- The 0.721/0.720 final-pick story and the "additional steps that improved performance"
  (paysem semantics) are explicit — those earn marks.

### Prompt 3 (plan mode)
```
You are completing the IEEE report for the DSML default-detection project (the 0.721
story). The baseline draft from Step 2 exists. Work in PLAN MODE first.

CONTEXT TO READ FIRST:
- docs/wrapup_roadmap.md (Step 3), docs/curation-decisions.md (stay inside the whitelist),
  and the current report draft under report/.
- docs/experiment_log.md, esp. P5 (rf_balanced, S4 → 0.713) and P14/E27–E31 (FE-on-RF +
  PAY-semantic features → 0.719/0.720) plus lessons L1, L5/L6, L7, L13, L14.
- outputs/submissions/MANIFEST.md for the LB column.
- The configs at "/Users/gabrieleadorni/Documents/POLITO/Magistrale Data Science
  Engineering/Deep Natural Language Processing/sub/configs/": config_B
  ([paysem, stress, payratio], rf_balanced, seeds 8888 & 10) = 0.721 best; config_A
  ([paysem, util, payratio]) = 0.720. rf_balanced = RandomForest 300 trees,
  min_samples_leaf=20, class_weight="balanced", + macro-F1 threshold tuning.
- src/preprocessing.py for what the `paysem`/`stress`/`util`/`payratio` feature families
  actually compute (so the report describes the real code).

TASK:
1. Extend Proposed Approach: Model Selection (the bake-off: HGB vs RF vs LogReg; why
   rf_balanced won — class_weight balanced + threshold tuning beats the others on this
   imbalanced data). Hyperparameter Tuning: the paired repeated-CV protocol and, crucially,
   the lesson that dev-CV is DECOUPLED from the LB here (L13) so candidates were ranked by
   the leaderboard. Motivate the PAY-semantic insight: PAY_* codes are NOT a monotone scale
   (-2 no-consumption, -1 paid-in-full, 0 revolving = riskier, 1..9 late); the `paysem`
   family keeps those states distinct, which RF (axis-aligned splits) exploits where HGB
   did not — this is the "additional step that improved performance".
2. Results: a CV-macro-F1 + public-LB table across the milestones (HGB 0.712 → rf_balanced
   0.713 → +payratio 0.716 → +paysem combos 0.719/0.720 → config_B 0.721); final-model
   confusion matrix + per-class metrics; note the 2 final picks (A 0.720, B 0.721, chosen
   for diverse error modes) and that only the private LB counts.
3. Discussion: error analysis (where the model still confuses default/non-default), the
   CV↔LB decoupling and how we handled it, RF-vs-HGB feature behaviour, limitations, and a
   one-line out-of-syllabus motivation if any method is off-syllabus.
4. References + ensure the data-quirk handling and macro-F1 choice are cited/justified.
5. RECONCILE THE DEPLOYED CONFIG (gold rule): decide which single config main.py reproduces
   — recommend setting config.yaml `chosen` + `seed` to the headline config so that
   report == config.yaml == submission.csv. Surface this decision to me (config_A at its
   seed vs config_B 0.721 at seed 8888); whichever we pick, regenerate submission.csv and
   make the report's headline match it.
6. Generate the final-model figures from the cleaned code; trim the whole report to ≤4 pages.

PLAN-MODE OUTPUT: propose the additions per section with the page budget showing we land
≤4 pages, the final figure/table list, the milestone numbers (sourced from the logs), and
the deployed-config reconciliation decision. Wait for my approval.

AFTER APPROVAL & EXECUTION: build the PDF, confirm ≤4 pages, and produce a table mapping
every report claim/number/figure to its source (code command or log entry) so Step 4 can
verify report⟺code parity. Flag any claim not backed by kept code.
```

---

## Step 4 — Final review pass (repo + report)

**Goal:** One strict pass over the whole deliverable: spec compliance, **report⟺code
parity (gold rule)**, **LLM-free / clean-original code** scan (strict), broad report
naturalness check (LLM-assisted is allowed, declared), and **removal of all
agentic-workflow docs**, replaced by a single clean README (description + quickstart).

**Review checklist:**
- Specs: IEEE template unmodified, ≤4 pages; submission header exactly `Id,Predicted`;
  `main.py` reproduces `submission.csv` ≤150 s; no external data; no leakage; Python-only.
- Parity: every method/number/figure in the report exists in the kept code and vice versa;
  stale references fixed (e.g. any "HGB is the chosen model" leftovers, SEED mismatches
  42/2024/888 across docs, outdated CLAUDE.md LB numbers).
- Code reads as clean, consistent, original human work: no emojis, no "Here's…"/chatty
  comments, no agentic scaffolding, no over-commenting, consistent naming/voice, idiomatic
  and PEP-8-ish, no plagiarized blocks.
- Workflow docs removed from the deliverable; a single clean README remains.

### Prompt 4 (plan mode)
```
You are doing the FINAL REVIEW PASS on the DSML default-detection deliverable (code +
report) before packaging. Be strict. Work in PLAN MODE first: produce a findings report +
fix plan for my approval, then execute.

CONTEXT TO READ FIRST:
- docs/wrapup_roadmap.md (Step 4) and docs/curation-decisions.md (the intended whitelist).
- The exam-rules PDF + assignment PDF for the exact specs (4 pages, IEEE template
  unmodified, Python-only single ZIP, reproducible, clear entry point, all results in the
  report, no external data, LLM only for the report not the code + declaration form).
- The final report under report/ and the report-claim→source mapping produced in Step 3.

TASK (four checks, then cleanup):
1. SPEC COMPLIANCE: verify IEEE template untouched (fonts/margins) and ≤4 pages; submission
   header is exactly `Id,Predicted` (submission.validate_submission); `python main.py`
   reproduces submission.csv from scratch in ≤150 s; no external data; all transforms fit
   inside the Pipeline (no leakage); software is Python-only.
2. REPORT⟺CODE PARITY (gold rule, strict): cross-check every method, number, figure, and
   table in the report against the kept code/config and the logs. List mismatches: methods
   described but not in code, code paths not reflected in the report, numbers that don't
   match. Fix stale references across all kept files (config.yaml `chosen`, any leftover
   "HGB chosen" text, SEED consistency, CLAUDE.md/README leaderboard numbers).
3. CODE AUTHENTICITY (strict — code must be clean, original, human-authored per the rules):
   scan all kept .py for AI/agentic tells and quality issues — emojis, "Here's/Let's/I'll"
   or chatty/explanatory comments, over-commenting of trivial lines, inconsistent
   naming/voice, leftover TODO/scaffolding, dead code, anything referencing Claude/LLM/agent
   in code or comments. Rewrite to clean, consistent, idiomatic Python with appropriate
   (not excessive) comments. Do not change behavior; re-run main.py after to confirm
   identical submission.csv.
4. REPORT NATURALNESS (broad): the report may be LLM-assisted (allowed, declared), so just
   sanity-check for unnatural phrasing, hedging, repetition, and any unsupported claim;
   ensure an academic voice. Do not over-edit.
5. CLEANUP — remove ALL agentic/workflow docs from the deliverable (they stay on the
   branch history / your local copy): CLAUDE.md, docs/experiment_log.md,
   outputs/submissions/MANIFEST.md, howto.txt, docs/curation-decisions.md,
   docs/wrapup_roadmap.md, and any plan/AGENTS/scratch files. Replace with a single clean
   README.md: short project description + quickstart (setup, `python main.py`, where the
   submission lands, where the report is) per the software "clear entry point" rule.

PLAN-MODE OUTPUT: a findings report grouped by the 4 checks (with concrete file:line items)
+ the proposed cleanup list + the README outline. Wait for approval before editing.

AFTER APPROVAL & EXECUTION: re-run main.py (confirm ≤150 s + byte-identical submission.csv),
rebuild the report PDF (confirm ≤4 pages), and give me a final diff summary + a green/red
checklist against every spec above. Note: the LLM-usage declaration form is mine to fill
honestly per the rules — remind me, do not fill it for me.
```

---

## Step 5 — Package & submit

**Goal:** Produce and verify the final ZIP, confirm end-to-end reproduction, lock the 2
final LB picks, and hand me a precise human submission checklist.

**Review checklist:**
- Fresh-checkout reproduction: `pip install -r requirements.txt` then `python main.py`
  → byte-identical `submission.csv`, ≤150 s; `submission.validate_submission` passes.
- ZIP is a real `.zip` (not .rar/.7z), Python-only, with `main.py` + README + `submission.csv`
  and the kept scripts; opens to a clear entry point.
- 2 final picks identified (config_B 0.721 + config_A 0.720, diverse error modes); reminder
  that only the private LB counts.
- A checklist of the 4 deliverables with what's automated vs. what I must do by hand.

### Prompt 5 (plan mode)
```
You are packaging the final DSML default-detection deliverable for submission. Work in PLAN
MODE first.

CONTEXT TO READ FIRST:
- docs/wrapup_roadmap.md (Step 5) — note Step 4 already removed the workflow docs, so read
  whatever README now describes the repo.
- The exam-rules PDF deliverables + software requirements (single .zip, Python-only,
  reproducible, clear entry point; up to 2 final submissions; only the private LB counts).

TASK:
1. Do a clean reproduction check: from a clean state, `pip install -r requirements.txt`
   then `python main.py`; confirm it finishes ≤150 s and writes a submission.csv that is
   byte-identical to the committed one; confirm submission.validate passes (header exactly
   Id,Predicted, one row per eval id).
2. Build the SOFTWARE ZIP: a single .zip containing the Python deliverable (main.py, src/,
   config.yaml, requirements.txt, the kept experiment scripts, README, and
   outputs/submissions/submission.csv) — exclude .venv/, .git/, data/raw/ CSVs, .DS_Store,
   caches, and the (already-removed) workflow docs. Verify it unzips to a working,
   self-contained project with a clear entry point.
3. Lock the 2 final leaderboard picks: config_B ([paysem, stress, payratio], 0.721) +
   config_A ([paysem, util, payratio], 0.720) for diverse error modes; note the matching
   submission_*.csv files and that only the private LB contributes to the grade.
4. Produce the FINAL SUBMISSION CHECKLIST for the 4 deliverables, marking what is done vs.
   what I must do by hand: (a) upload the chosen predictions to DSLE and select the 2 final
   submissions; (b) submit the report PDF; (c) upload the software .zip; (d) fill & submit
   the LLM-usage declaration form (mine to do honestly per the rules).

PLAN-MODE OUTPUT: the ZIP manifest (include/exclude list), the reproduction-check plan, and
the final checklist, for my approval. Then execute the verifiable parts (repro check + ZIP
build + validation) and report results; leave the human upload/declaration steps as a
checklist for me.
```

---

## Verification (end-to-end)

After Step 5, the deliverable is correct when all of these hold:
- Clean checkout → `pip install -r requirements.txt` → `python main.py` reproduces
  `outputs/submissions/submission.csv` byte-identically in **≤150 s**;
  `submission.validate_submission` passes (header exactly `Id,Predicted`).
- Report PDF: official IEEE template **unmodified**, **≤4 pages** excl. references, all 5
  required sections present, every figure/number present in-text (nothing "to regenerate").
- **Report ⟺ code parity** holds: the headline model in the report == `config.yaml`
  `chosen` == what `main.py` writes; no method appears in one but not the other.
- Code is clean/original/human-authored (no LLM/agentic tells); no external data; no
  leakage; Python-only.
- All agentic-workflow docs removed; a single clean README provides the entry point.
- Software `.zip` is well-formed and self-contained; the 2 final LB picks are selected;
  the LLM-usage declaration form is filled by the user.

## Notes / open decisions to surface during execution
- **Deployed config choice (Step 3):** `main.py` can reproduce only one config; the report
  headline + `submission.csv` must match it. Recommend config_B (0.721, seed 8888) as the
  deployed headline, config_A (0.720) as the documented 2nd final pick.
- **"Tried & ruled out" scope (Step 1):** decide whether to keep one negative-results
  script (e.g. boosting/stacking) to back a short report paragraph, or drop them all and
  only mention the conclusion. Keeping any means keeping its dep and a report sentence.
- **Seed-hunt framing (Steps 1 & 3):** present seed sensitivity honestly (±0.001–0.002,
  config_B seed 8888 → 0.721) as variance, not as the main lever (the lever is the paysem
  features); drop the raw seed-hunting scripts as not report-worthy.
