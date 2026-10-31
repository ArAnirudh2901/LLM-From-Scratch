# Handoff (2026-10-26): continue here with any agent / account

Read in order: this file -> `2_Training_Tiny_LLM/JOURNEY.md` (story, intuition, results, lessons) -> `AGENTS.md`
(handbook + full log) -> `~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md` (website). Claude memory (if on this Mac):
`~/.claude/projects/-Users-anirudharavalli-Generative-AI-LLM-from-Scratch/memory/` (MEMORY.md index). On another machine
use the private bundle `~/kinthread_handoff.tgz` (memory, Gemini keys, real questions; never commit or share it).

## Completed final selection (Kaggle, owner's account)
- r8-final: `notebookf6d1e34e7f` version 6 (causal reader). r8p-final: `notebook99ae0586bd` version 7
  (prefix-LM). The final selection in `notebookf8798397a0`, run `362159765`, completed and wrote `FINAL.json`.
- Full dev (748 total; 714 fair / 650 verified): r8p base 73.0% fair / 76.5% verified; r7 72.7% / 76.5%; r8
  72.1% / 75.5%. The winner was `r8p@notebooks` with `all` reading settings (`--verifier verifier.json --cover
  --split_search --para 2 --k 16`), scoring 73.9% fair / 77.2% verified on dev.
- Locked test, scored once after the dev pick: winner 68.8% fair / 72.2% verified; r7 baseline 66.2% / 70.5%
  (+2.6 / +1.7 points). Do not re-score locked test for selection.

## Next steps
1. Candidate, verifier and final report are staged locally at `runs/v10-r8p-final/` (git-ignored). Exact checkpoint
   hash matches Kaggle. The downloaded `verifier.json` has 1,466 questions, 1,172 training examples and 22 weights.
2. In the canonical `LLM-From-Scratch/2_Training_Tiny_LLM` source, wire the selected setting as a model preset or
   defaults (`--adaptive 1.0 --hybrid --verifier runs/v10-r8p-final/verifier.json --cover --split_search --para 2
   --k 16`), then mirror the code here using the documented subtree workflow. Keep paths valid after the final
   artifacts are included. Do not commit or push unless asked.
3. For a release, split `best.pt` into <100 MB chunks, preserve `best.pt.sha256`, and update `.gitignore` explicitly;
   the current run folder is ignored. Website: follow `MODEL_UPGRADE_PLAN.md` (browser downloads the model; Cloudflare
   Pages only serves files).

## Rules that must hold
- Own from-scratch model only; fp32; no quantization/distillation; scraped real questions are evaluation only; locked
  test scored once for selection; decide on the full dev, never a subset.
- Commits authored by the owner (git config in repo); never add Claude co-author/session trailers. Push the model
  repo as the owner's GitHub account (see AGENTS.md "Owner's hard rules"); never act as another gh account.
- Never type passwords or create accounts; secrets (Gemini keys) only in `~/.config/kinthread/*.env`, never printed.
- Heavy jobs on Kaggle/Colab, not the 8 GB Mac. Keep tool calls and replies short (owner's usage limit).

## 2026-10-26 update
- Final selection completed in Kaggle notebook `notebookf8798397a0`, run `362159765`. Full dev (748 total; 714 fair / 650 verified): r8p base 73.0% fair / 76.5% verified; r7 72.7% / 76.5%; r8 72.1% / 75.5%. The dev winner is `r8p@notebooks` with setting `all`: fixed `--adaptive 1.0 --hybrid` plus `--verifier verifier.json --cover --split_search --para 2 --k 16`, scoring 73.9% fair / 77.2% verified.
- Locked test was scored once after the dev pick: winner 68.8% fair / 72.2% verified; r7 baseline 66.2% / 70.5%, improvement +2.6 / +1.7 points. Do not re-score this locked test for selection.
- `/Users/anirudharavalli/Downloads/best.zip` is v10-r8p step 6,000/15,000 (SHA-256 `38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52`), the selected Kaggle checkpoint, byte-identical to Kaggle `best.pt` (158,748,883 bytes, SHA-256 `38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52`). `FINAL.json` and the 628-byte `verifier.json` are in `~/Downloads`; all three artifacts are copied into local, git-ignored `runs/v10-r8p-final/` with `best.pt.sha256`. Remaining: decide and implement the CLI defaults in the canonical source repo, then mirror them here; do not commit or push unless asked.
- Chrome extension status checked: ChatGPT extension v1.26.1020.11845 is installed, enabled, and allowed on all sites. Codex's browser inventory still does not expose a Chrome surface, but native Chrome computer use works now.

## Update 2026-10-26
Final selection completed: v10-r8p, step 6,000, `all` boosters won full dev and was scored once on locked test. Details and integrity SHA are in AGENTS.md/JOURNEY.md. Export work has started in canonical LLM-From-Scratch source (`export_reader_web.py`), exact fp32, 39.50M parameters, 126.1 MB gzip. Website still runs v8 while v10 WASM/Pyodide parity and mobile budgets remain unverified. No commits or pushes requested.

## Update 2026-10-26 (v10 browser runtime prototype)
Canonical LLM source now has an exact-fp32 v10 export and separate SIMD/scalar WASM forward runtime. The golden test's maximum absolute difference was 1.15e-5 across prefix, cached decode, tied logits and copy head outputs for one short prompt. This is not full parity and has not been synced into Kinthread. Current site remains v8; follow `~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md` and do not deploy until pipeline/device tests pass.

## 2026-10-26 export follow-up
The canonical source exporter produced `/tmp/kinthread-v10-final` from `runs/v10-r8p-final/best.pt` plus the final settings, verifier, and name table. Manifest generated; do not treat this as a website release. Kinthread published the separate low-power v8 runtime update at `https://af2acf01.kinthread.pages.dev`; continue website v10 integration and parity before replacing v8.

## 2026-10-26 parity follow-up
The canonical LLM source has a 50-prompt deterministic C/WASM parity check passing in SIMD and scalar against PyTorch (13–101 tokens). Maximum: prefix 2.86e-5, decode 1.38e-5, logits 1.53e-5, copy keys 2.38e-5, mixed probabilities 7.82e-5. This does not validate the JS wrapper, beam search, end-to-end answers, or mobile budgets. Site's separate resource-aware v8 update is live at `https://kinthread.pages.dev`.

The 50-prompt C/WASM parity cases also now pass through `ReaderWasm` JS wrapper prefill/cache/decode in scalar and SIMD (max logp 2.27e-4; probability 7.82e-5). Full Python beam and answer parity remain pending.

## Website release verification (2026-10-26)
Kinthread (`~/Web_Dev/kinthread`) now has release-manifest checks and a returning-browser model update flow, but the
active identity is intentionally still v8 (`af70872adbce`). Chrome on a local preview verified successful cached
replacement and inference, rejected a same-size corrupted part while preserving the prior install, and blocked model
startup when a tampered manifest advertised `v9-test` instead of the app's expected v8. The update notice was checked
in dark and light themes; CSS stacks it below 560px, but a real mobile viewport was not tested. Website validation
passed (`bun run check`: lint, 134 tests, static build). The selected v10-r8p model is not releasable yet: full reader
grammar/beam, `kinlogic`, and exam pipeline parity remain incomplete. No deploy, commit, or push was made. See
`~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md` for details.

## Update 2026-10-27 — threaded v10 validation

RoPE precomputation and SIMD/scalar pthread reader builds now exist in canonical source and this mirror. The website selects threads only on isolated devices reporting at least 8 GB; low/unknown-memory profiles remain single-threaded. Threaded builds reserve 384 MiB because branching beam hypotheses consume distinct KV slots. 50-prompt wrapper parity passes. A Node/Pyodide end-to-end test and native Chrome local run return the expected three-link paternal-cousin answer; Chrome used SIMD/two threads and took 9,632 ms. Its score is tentative and uncalibrated. This is one smoke test, not broad parity. Production remains v8; locked-test accuracy is 68.8% fair / 72.2% verified (<85%). No deployment, commit, or push.

## Current handoff — 2026-10-27

The owner explicitly authorized pushing and publishing v10-r8p. Source, model mirror, and website commits are pushed: LLM-From-Scratch/main 11508ba; Kinthread-llm-model/v9-reader 57455fc; Kinthread/main 6f32924. Cloudflare Pages project kinthread now serves v10-r8p. The live no-cache release manifest matched checkpoint SHA-256 38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52, content hash c4a1d87737c6, and runtime f608eed15e9b. SHA-256 and size verification passed for all 66 model assets (127,187,553 bytes) and 20 runtime files (14,012,183 bytes). Selected weights remain exact fp32.

This publication was owner-authorized before all release gates were complete. Locked-test accuracy is 68.8% fair / 72.2% verified (<85%); broad browser/Python parity, confidence calibration, and low-power/mobile budgets remain open. The 9,632 ms Chrome paternal-cousin result is one tentative smoke check, not an accuracy claim.

- 2026-10-27 (v10-r8p accuracy check): checkpoint SHA-256 38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52 evaluated with eval.evaluate --sets unseen on frozen generated data SHA-256 42d1a89c1a6bc4215b0b1c9e8faf6e50be0a98a84e26f627db77c8e5a14bad0d. Result: 1,394/1,400 = 99.57%; misses were 5/95 story and 1/83 sufficiency questions. The manifest says question wording was excluded from training/tuning; this set was previously reported for v10-r3 (99.21%), so do not treat it as project-blind or real-world accuracy. The official locked real-exam result remains 474/689 fair = 68.8% and 462/640 verified = 72.2%, scored once for selection. No local real-question files are available to rerun; preserve the locked-test rule. Full JSON is in runs/v10-r8p-final/unseen_eval.json.

- 2026-10-27 (overfitting diagnostic): read checkpoint health history at steps 1,500/3,000/4,500/6,000: training loss 0.0893→0.0170; held-out loss 0.0127→0.0125 (intermediate max 0.0136); synthetic accuracy 96.7%→96.6%; real-dev score 71.2%→72.9%. This shows no clear classical train/held-out divergence in saved points. Re-evaluated separate available sets: fresh 38/40 (95.0%), gender 23/24 (95.8%), reworded 329/340 same-answer (96.8%) but only 258/340 exact fact parses (75.9%). These are small or synthetic/reworded benchmarks, not real-world exam accuracy. Combined with the generated-unseen 1,394/1,400 result, the near-perfect generated scores demonstrate strong generator-distribution performance; the official real-exam locked score is still 68.8% fair / 72.2% verified. Best diagnosis: synthetic-to-real distribution gap and template-family bias are evident risks; 99.57% alone does not prove classical overfitting. Full details: runs/v10-r8p-final/accuracy_diagnostic.json.

- 2026-10-28 (balanced question-format supplement): added a deterministic, code-generated candidate with 3,750 rows: 1,250 each direct speech, indirect speech, and long stories; 25 answer-relation buckets have exactly 50 rows in every format. Long stories span 80–187 words in this run, across the configured 80–240 word bands. Indirect wording now shifts first-person references to the speaker's name and possessive. All rows were independently parsed and solved; all 2,500 speech MCQs have five parsed choices with the gold answer exactly once, and each direct/indirect pair matches facts, truth, and answer. Data SHA-256: 315dccb5d2a9bdf0b3f32441601babb6b22d21eb73ad8b4545a51e7311e5cd88. No Gemini requests, training, checkpoint changes, commit, or push. The supplement is a candidate, not a model-accuracy claim.

- 2026-10-28 (long-story reader baseline): Long-story baseline diagnostic (same fixed 15 rows: five each from 80–120, 121–170, and 171–240 configured word bands): current v10-r8p checkpoint SHA remains 38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52. Python reader top-1 matched 5/15 gold answers at the default 96 generated-token cap and 12/15 when only the diagnostic cap was raised to 192. All 1,250 long-story prefixes fit within 236 tokens, below the 320-token prefix budget with max_new=192, so the test did not truncate their input. Three medium/extended cases still failed. This tiny same-generator sample is diagnostic, not an accuracy estimate; no model weights or production runtime changed.

## 2026-10-28 — current handoff for Claude's model-improvement brainstorm

### What changed in this code version

- `kinlogic/generate.py`: `q_speech` accepts an optional forced `direct` or `indirect` mode for controlled format generation. Forced examples use the same underlying graph and question direction, explicitly render either quoted speech or reported speech, convert first-person references to the speaker's name/possessive in reported speech, and add explicit gender evidence only when the text otherwise leaves it unstated. Existing stochastic generation remains available when no mode is forced.
- `kinlogic/speech.py`: the reported-speech reader can resolve an explicit entity/name at the start of a clause, which is needed by generated indirect forms.
- `generate_balanced_question_types.py`: deterministic graph-based builder for 3,750 candidates. There are 1,250 each of direct speech, indirect speech, and long story; exactly 50 rows for each of 25 canonical answer types in every format. Each speech item has five parsed options with the answer exactly once. Direct/indirect pairs have matching graph, facts, query, options, and answer. Stories are allocated across 80–120, 121–170, and 171–240 word acceptance bands; the realized set has 81–187 words (mean 127.04).
- `kinlogic/reword.py` and `generate_gemini_synthetic.py`: clarified high-precision rewrite instructions (all required names/letters, relation direction, speakers/pronouns, quantifiers/negation, exact fact graph); constrained JSON output, fact extraction and explicit rejection reasons; robust response text handling; an environment-controlled Gemini request interval (default 8 seconds); resumable source hashing, stratified source queues, acceptance audit and progress state. The Gemini key is read from `GEMINI_API_KEY`; no key is included in code or data.
- `SYNTHETIC_DATASET_PLAN.md`, README, this handoff, and the memory files record candidate boundaries, integrity data, diagnostic results, and the next research questions.

### Candidate datasets and verification

The graph-generated balanced candidate lives in the model mirror as `data/balanced_question_formats_v1.jsonl.gz` with a detailed manifest. SHA-256: `315dccb5d2a9bdf0b3f32441601babb6b22d21eb73ad8b4545a51e7311e5cd88`. It was produced without Gemini. Every row was parsed by the exact project reader and solved by `kinlogic.engine.solve`; an independent full audit reported no errors. It is a candidate, not trained data.

The separate Gemini pool is `data/gemini_synthetic_v1.jsonl.gz` (1,015 accepted / 2,069 attempted source rows; 49.1%), focus subset `data/gemini_synthetic_v1.focus.jsonl.gz` (204 rows), audit `data/gemini_synthetic_v1.audit.jsonl`, and resumable state `data/gemini_synthetic_v1.state.json`. This run tracked 275 API requests at an 8-second interval. It is not balanced by format or answer type. The audit files contain prompt/rewrite data and hashes, but no API secret. Keep this pool separate until label-depth distribution, train overlap, and relation-range suitability are explicitly checked.

### Model status and measured failure signal

No model training, checkpoint export, website change, Cloudflare deploy, or weight modification occurred in this work. Selected v10-r8p remains step 6,000, exact fp32, checkpoint SHA-256 `38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52`.

A fixed 15-item long-story smoke sample (five examples from each configured band) returned 5/15 top-1 answers with `max_new=96` and 12/15 when only `max_new` was raised to 192. The stories fit within the 236-token observed prefix maximum and the configured 320-token prefix budget. Three still failed. Treat this as evidence that decoding budget may contribute, not proof that it is the sole cause or an accuracy estimate. The official locked real-exam result remains 68.8% fair / 72.2% verified (<85%); the generated 99.57% score is not a project-blind real-world result.

### Recommended brainstorm starting point

Before training, use a stratified long-story development set to decompose exact fact reading vs. final answer and log prefix length, generated-token count, stop reason, and result under both output caps. Then inspect the 3 misses and the Gemini acceptance/relation-depth distributions. If Claude proposes model experiments, require one change at a time, a fixed development suite, overlap checks, and a decision rule that protects the current checkpoint; do not use or rescore the locked test for tuning. No experiment has yet shown that either new dataset improves v10-r8p.

### Cloud Claude context

This handoff is self-contained for Claude running in the cloud. Read this repository's `AGENTS.md`, `HANDOFF.md`, `SYNTHETIC_DATASET_PLAN.md`, `JOURNEY.md`, and `RESEARCH.md`; do not assume access to this Mac's `~/.claude` memory or local private files. Local Claude memory is supplemental and is not part of the GitHub handoff.

## 2026-10-28 — measure where dev points are lost (per-question trace)

`eval.run_real --trace` records, for every question: the route (an exact reader `codes` / `rules` / `speech`, or the
model), every reading the vote saw with the engine's answer and whether it reaches the key, and the final pick.
`--trace_shadow` also answers each exactly-read question with the exact readers off, which shows how often the rules
are confidently wrong where the model would be right. The hooks only observe: the dev score must equal FINAL.json's
528/714 fair; if it does not, the trace is invalid. Run on dev only (never `--split test`). Shadow roughly doubles
model time on exactly-read questions.

```bash
python -m eval.run_real --ckpt runs/v10-r8p-final/best.pt --split dev --adaptive 1.0 --hybrid \
  --verifier runs/v10-r8p-final/verifier.json --cover --split_search --para 2 --k 16 \
  --trace runs/v10-r8p-final/trace_dev.jsonl --trace_shadow --out runs/v10-r8p-final/real_dev_traced.json
python -m eval.trace_report runs/v10-r8p-final/trace_dev.jsonl --out runs/v10-r8p-final/trace_dev_report.json
```

Each miss lands in one bucket: exact reader wrong (fix or cross-check the rules), no reading right (reading coverage:
training data in that wording), right reading not picked (selection: verifier). A separate tag counts misses where the
engine found no family link although the key names a relation (solver conventions). Example: in "Barkha and Chandresh
are siblings. Manish has two children ... Hansa has only one son ...", the engine answers "not related" even from
correct facts, and "granddaughter" once everyone named is assumed to be one family (`members 5`). Decide the next
round of work from these counts, per category.

## 2026-10-28 — decoder parity fixes, answer-stage flags, balance and repeatability tools

Written in the cloud mirror (no torch, no private real questions). Nothing here changes the checkpoint or the default
answer path; every answer-stage change is a flag that is off until dev says otherwise.

**Answer-stage flags** (`eval.run_real`, env `KIN_*` in `kinlogic/exam.py`):
`--closure` ("not related" with unnamed relatives or `has` facts: retry with everyone named as one family, never when
"no relation" is an option), `--core_repair` (every reading contradicts: find a minimal conflicting set by deletion and
answer without each of its facts), `--lever` (verifier probability times the reader weight instead of replacing it),
`--trace_pow p` (vote weight times traceability^p: the share of facts whose two people share a sentence), `--max_new`
(output budget; 63% of long-story gold readings exceed 96 tokens) and `--reserve r` (within r tokens of the budget,
finish the statement and write the question: `kinlogic.facts.budget_filter`). Always on: "my. wife" -> "my wife"
(`prepare.normalize`), and `reader.encode_example` renumbers people by first appearance and drops rows with more than
32 (2.2% of mix_r8 rows were silently encoded with `<unk>` people).

Score them from one traced dev run without re-reading (dev only; the locked test is never used for this):

```bash
python -m eval.rescore_trace runs/v10-r8p-final/trace_dev.jsonl --verifier runs/v10-r8p-final/verifier.json --cover --all
```

It prints each setting's dev score and McNemar against `base`. Adopt a flag only if it wins on dev with p < 0.05.

**Browser decoder (Kinthread `runtime/v10-decoder.js`, runtime a171d14ed9a4).** Two parity bugs against
`reader.generate`, both fixed with tests: each decoded token did not attend to its own key/value slot (first-step
log-prob error 0.41; now 0), and `<eos>` was only allowed after a written `;`, so every reading took one extra forced
step. The beam now also stops at k finished readings, like Python. On 89 hard local rows (reworded and long stories),
top-1 engine answers were 54 / 52 / 53 (production / slot fix / both): no measurable accuracy change, mean decode
4.5 s -> 2.8 s. Still different from the measured pipeline: the site sends the v8 `forModel` rewording and strips the
exam question and options (so no option mapping), runs without `--adaptive`/`--para`, and lower tiers use beam 4/8.

**Balance.** `python -m eval.balance_report <file>` profiles any training file by question format (read from the
text), answer, question type, fact count, people, input/output length and constraint words. mix_r8: statements 28.3%,
coded 18.9%, direct speech 15.4%, family puzzle 13.7%, indirect speech 11.5%, long story 6.1%, possessive chain 2.4%,
nested clauses 2.3%, data sufficiency 1.5%; questions ask 81.7%, who 4%, gender 2.1%, pair 1.9%, count 1.8%; 7.1% of
gold readings exceed 96 tokens. `build_balanced.py` fills every format to the same count with answers (<= 1/12 of a
format), fact counts (<= 1/3) and source kinds (<= 1/2) spread inside it; caps relax x3 in a second pass for formats
that cannot reach them. Each row must parse, have <= 32 people, fit 190 output tokens, and (relation questions) read
to its stored answer, and only it, through the engine. Long stories come from `generate_balanced_question_types.py`.

**Detour (obfuscated) questions.** `kinlogic.generate.obfuscation_set` writes one family and pair with 0, 1 and 2
detours ("father" -> "mother's husband", "son" -> "wife's son", "brother" -> "father's only son"), same names, wording
and options; `q_obfuscated` is the opt-in kind (`KINDS['obfuscated']`, not in `MIX`, so `sample()` is unchanged).
After a detour, every step must name one person whatever the family (`_exact`): "the son of Urmila's husband's
mother" may be her husband, and the engine calls it unique, so such rows are rejected.

**Repeatability.** `python -m eval.consistency --groups 300 [--reader gold|exact | --ckpt ...]` scores answer
agreement and accuracy under renaming (to names the name table reads the same way), option shuffles, fact-sentence
reordering and 1-2 detours. Gold facts: 100% agreement (engine and answer mapping are invariant). Rule readers only:
96.7% of groups agree; detours flip 6.7% of answers. The model figure needs torch:

```bash
python -m eval.consistency --groups 300 --ckpt runs/v10-r8p-final/best.pt --k 16 --out runs/v10-r8p-final/consistency.json
```

Copy to LLM-From-Scratch: `build_balanced.py`, `eval/{balance_report,consistency,rescore_trace,trace}.py`,
`eval/run_real.py`, `kinlogic/{exam,facts,generate,prepare}.py`, `reader.py`, `tests/test_{answer_options,balance_consistency}.py`.

**Locked test v2 (`eval/locked.py`).** v1 halves one pool by hash, so site templates sit in dev and test alike. v2 is
frozen from new questions only (scrape new sites with `eval.sites` into `data/fresh_online/`), from sites with no
question in the v1 dev half, minus any question within `eval.leak.NEAR` (people-masked 3-gram Jaccard) of a dev
question. The manifest holds hashes only; dev and v1 runs exclude these questions; `--split locked2` refuses a partial
run and a second score without `--rescore_reason`, and appends checkpoint SHA-256, arguments and scores to
`data/locked_v2_ledger.jsonl`.

```bash
python -m eval.locked --freeze --pattern 'data/fresh_online/*.jsonl'     # once, before any v11 work looks at them
python -m eval.run_real --ckpt <selected>.pt --pattern 'data/fresh_online/*.jsonl' --split locked2 <FINAL.json settings>
```

## 2026-10-28 — gauntlet, deep lineages, twisted set, r9p notebook

**Gauntlet (`eval/gauntlet.py`, `data/gauntlet_v1.jsonl.gz`, evaluation only, leak-guarded).** 1,140 generated questions
in 19 categories (60 each: lines of descent of 3-5 and 6-10 generations, cousins up to the fifth degree, 1/2/3
detours, direct and indirect speech, introductions, identities, nested, shuffled statements, deep chains, coded,
puzzles, sufficiency, claims, 80-240 word stories, two-family stories), each with a renamed or option-shuffled twin,
plus 28 hand-written twisted questions (`eval/twisted_set.py`, all verified by `python -m eval.twisted_set`). Expected
answers come from the gold facts through the answer pipeline; gold scores 100%, rule readers alone 45.7%.

Production pipeline (Kinthread browser runtime with the fixed decoder, Node + Pyodide, beam 8, run on 667 questions:
30 per category, all generations rows and all twisted ones): **614/667 = 92.1%**. Weak: claims 63%, puzzles 70%, two
detours 77%, sufficiency 77%, three detours 83%, twisted 24/28 (missed: two families joined only by "friends" read as
cousins, one sufficiency, one claims, one deep in-law chain). Strong: 3-10 generations 97-98% (rules read only ~22% of
these: the model does), deep cousins 100%. Speech, coded and story rows are 100% read by the exact rules (the
generators write rule-parseable text), so those cells say little about the model. Twins: same answer 100% except deep
chains (92%) and sufficiency (93%). Generated questions are not real exams (real locked: 68.8%); use the gauntlet for
where the reader breaks, the real dev split for decisions.

**Deep lineages.** `kinlogic.generate.Lineage` / `q_generations` (opt-in kind): two lines of descent from one couple,
up to ten generations each, statements often shuffled, spouses as distractors. Fixed on the way: `World.father/mother/
spouse` return -1 for none, and the new detour/lineage code had tested `is not None`, which let -1 index the last
person (`test_generated_questions_are_true` caught it).

**Next run.** `colab/r9p/C_train` (was `colab/r9p_kaggle.ipynb`; r8p warm start checked by SHA-256, `balanced_mix_v1` added, block 512) and
`TRAINING_V11_CODEX.md` (the full train / select / locked procedure and the hypothesis order toward 85-90%).

**Training supplements (candidates, not trained on yet).** `data/balanced_mix_v1.jsonl.gz` (22,500 rows; manifest next to
it): 9 formats x 2,500; questions 79% ask, 7% none (claims, sufficiency), 5.6% who, 3.4% gender, 3.3% pair, 1.3% count
(mix_r8: 81.7% ask); 1,137 distinct answers, none above 4.6%; people <= 32; fact tokens <= 190 (2.3% above 128).
`data/detour_v1.jsonl.gz` (4,000 rows, `build_detours.py`): 1/2/3 detours equally, top answer 4.7%. Both are exact
generator facts, engine-checked; the leak guard keeps the gauntlet out at training time.

**Exam-pattern mining (`eval/mine_patterns.py`).** Counts what real dev questions look like (openings, question forms,
constructions, kin words, coded expression size, length, people, options, answers, masked skeletons shared by 3+
questions) next to generator or training-file output, with per-feature model accuracy from a dev trace. Writes counts
only, never question text; reads dev only and skips locked2. Needs the private real questions, so it runs on Kaggle or
the owner's Mac (TRAINING_V11_CODEX.md step 1b). Stand-in run on the 152 authored questions: generated questions are
far less often 4-option MCQ (16% vs 95%) and "How is X related to Y?" (25% vs 84%).

## 2026-10-29 — r9p run-book for Codex (start here for training)

`TRAINING_V11_CODEX.md` is now a run-book for a smaller model: a prompt to paste, fixed rules, setup, six steps each
run by one command (`colab/r9p/run_step.sh A|B1|C|B2|D|E`), what to check and copy after each, a STOP-message table
and a report template. The notebooks are generated by `colab/r9p/build_notebooks.py` (one folder per step with
`notebook.ipynb` and `kernel-metadata.json` for `kaggle kernels push`); every cell stops with `STOP: ...` on any
problem; outputs are split into `public/` (counts only, committed under `results/r9p/<step>/`) and `private/` (real
question text, kept on Kaggle and in `~/kaggle_r9p`). Decisions are computed, not judged: `eval.rescore_trace --out`
gives the adopted answer flags, `eval/compare_runs.py` the dev adoption rule (>= 2 points, more wins than losses,
McNemar p < 0.05), and step D adds the gauntlet and consistency checks into `decision.json`. Step E (locked test) runs
only after the owner writes "approve locked". The old `colab/r9p_kaggle.ipynb` is replaced by `colab/r9p/C_train`.

## 2026-10-29 — one branch per repo

All work is on `main` in Kinthread-llm-model, Kinthread and LLM-From-Scratch. `v9-reader`, the claude/* branches and
`kinship-qa-web` were merged (each fully contained in `main`) and are deleted on GitHub. LLM-From-Scratch
`2_Training_Tiny_LLM` was synced from this repo (60 files, none diverged). Notebooks and the r9p run-book clone `main`.
The answer-option tests now state the gender they need, so they pass without the name table.

## 2026-10-29 — r9p run stopped at step A

Pulled `origin/main`; local `HEAD` matched `2f7055e576ad4f40d89e164eb76b58e787786c35`. Uploaded the evaluation
archive, selected `FINAL.json`/`verifier.json`, and the SHA-matched r8p checkpoint to the private Kaggle dataset
`aranirudh/kinthread-real-questions`. Pushed step A as `aranirudh/kinthread-r9p-a-baseline`; Kaggle ended it with
`KernelWorkerStatus.ERROR`. The kernel log reports `NameError: name 'REPO' is not defined` in `In [1]`. Per the
run-book, stopped the run without retrying or changing notebook/runner code. No metrics or training checkpoint were
produced; no `results/r9p/A/` output, commit, or push was made. The A kernel metadata account/dataset references
were updated locally for this attempt and remain uncommitted.

### r9p run report (stopped)

| Measure | r8p | r9p |
| --- | --- | --- |
| Real dev fair, FINAL setting | Not run (A failed) | Not run |
| Gain (points), wins / losses, McNemar p | — | Not run |
| Dev fair with 192-token reading | Not run | Not run |
| Gauntlet correct | Not run | Not run |
| Consistency, all variants agree | Not run | Not run |
| Locked test fair / verified | 68.8% / 72.2% (scored 2026-10-26) | Not run |

- Flags adopted in step A: not run
- Step D decision: not run
- r9p checkpoint SHA-256: not produced
- Largest exam-pattern gaps (A): not available
- Most missed patterns (A): not available
- Problems: step A — `KernelWorkerStatus.ERROR`; `NameError: name 'REPO' is not defined` in `In [1]`.

## 2026-10-30 — r9p notebooks fixed, steps A and C started

The `NameError` came from a truncated line in `colab/r9p/build_notebooks.py` (`REPO, X`; `BRANCH` was never defined),
shared by all six notebooks. Fixed (`REPO, BRANCH = '/kaggle/temp/Kinthread-llm-model', 'main'`), notebooks regenerated,
every notebook's code checked with pyflakes (no undefined names). The pipeline now runs under the owner's Kaggle account
`aranirudh`: the r8p and final-selection notebooks belong to `anirudharavalli` and cannot be attached there, so the private
dataset `aranirudh/kinthread-real-questions` holds `real_questions_full.tgz`, the SHA-checked r8p `best.pt`, `FINAL.json`
and `verifier.json`; steps A, B1 and C attach only that dataset. Steps A and C (training) were pushed at the same time (C
does not use A's output); then B1, B2, D. Locked test (E) only after "approve locked".

## r9p run (2026-10-31)

| Measure | r8p | r9p |
| --- | --- | --- |
| Real dev fair, FINAL setting | 528/714 = 73.9% | 525/714 = 73.5% |
| Gain (points), wins / losses, McNemar p | | -0.42, 13 / 16, p=0.71 |
| Dev fair with 192-token reading | 527/714 = 73.8% | 525/714 = 73.5% (config final_plus; 12 / 15, p=0.70) |
| Gauntlet correct | 0.9495 | 0.9615 |
| Consistency, all variants agree | 0.9400 | 0.9867 |
| Locked test fair / verified | 68.8% / 72.2% (scored 2026-10-26) | not run |

- Flags adopted in step A: none
- Step D decision: adopt false (kept r8p), config final, extra flags none
- r9p checkpoint SHA-256: 2aa1474a75a2acad6e852b1d9c2e89fa9041e81b76f420ef07969e3061c3b68b
- Largest exam-pattern gaps (A): options <= 4 (real 66.8%, generated 16.4%); "how is X related to Y" (58.2% vs 24.4%);
  kin word brother (48.1% vs 16.8%); single letters as names, 3+ (57.1% vs 26.3%); "only" (33.6% vs 9.1%)
- Most missed patterns (A): single letters as names, 3+ (117 missed, 73%); father (113, 73%); options <= 4 (110, 78%);
  son (102, 74%); daughter (102, 68%)
- Problems: the first C run used pre-review code (fp16-overflow crash risk) and was deleted and restarted once; A took
  6.5 h (trace + shadow), so B1/B2/D now run their two jobs on both T4s at once (`par()` in `build_notebooks.py`).
- Reading: r9p is more robust to rewording (consistency +4.7 points, gauntlet +1.2) but not more accurate on real dev.
  The added balanced / detour data taught invariance, not new real wording. Next ideas: `RESEARCH.md` 2026-10-31.
