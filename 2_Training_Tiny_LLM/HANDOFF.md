# Handoff (2026-10-25 night): continue here with any agent / account

Read in order: this file -> `2_Training_Tiny_LLM/JOURNEY.md` (story, intuition, results, lessons) -> `AGENTS.md`
(handbook + full log) -> `~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md` (website). Claude memory (if on this Mac):
`~/.claude/projects/-Users-anirudharavalli-Generative-AI-LLM-from-Scratch/memory/` (MEMORY.md index). On another machine
use the private bundle `~/kinthread_handoff.tgz` (memory, Gemini keys, real questions; never commit or share it).

## Running now (Kaggle, owner's account, Save & Run All)
- r8-final: notebook `notebookf6d1e34e7f` version 6 (causal reader). r8p-final: `notebook99ae0586bd` version 7
  (prefix-LM). Same data/code: per-type balanced `data/mix_r8` (build_mix.py), no-relation rows, leak guard, OOM
  fallback. Started ~2026-10-25 20:00 IST, ~4 h each. Output: `runs/v10-r8[p]/best.pt`, `real_dev.json`,
  `construction_suite.json`. A notebook shows "Successful" even if training crashed: read its log.
- Earlier results (old 573 dev subset, adaptive+hybrid): r7 72.9, r8 v5 73.5, r8p v5 74.3 (best so far).
- Kaggle GPU quota 30 h/week: ~17.6 h left before these runs, ~8.6 h after (the final run needs ~7 h).

## Next steps (exact)
1. When both runs finish: open notebook `notebookf8798397a0` (inputs already: private dataset
   `kinthread-real-questions`, r8 and r8p outputs) -> File > Import Notebook > Link:
   `https://raw.githubusercontent.com/ArAnirudh2901/Kinthread-llm-model/v9-reader/colab/final_kaggle.ipynb` -> OK ->
   Settings > Accelerator = GPU T4 x2 -> Save Version > Save & Run All. It fetches r8p v5, r8p v7, r8 v6 by version
   (kagglehub), picks the checkpoint on the full real dev, trains the verifier, sweeps boosters on dev, scores the
   locked test once for the pick and for r7, writes `/kaggle/working/FINAL.json` (~7 h).
2. Read FINAL.json (checkpoint, flags, locked_test fair/verified, improvement over r7). Put the numbers in
   JOURNEY.md (timeline "final" row) and AGENTS.md handoff log.
3. Release: download the winner `best.pt` (+ verifier.json if picked) from the Kaggle output, split
   (`split -b 90m best.pt best.pt.part-`; `shasum -a 256 best.pt > best.pt.sha256`), commit to Kinthread-llm-model
   `runs/<final>/`, set the chosen flags as defaults in `eval/run_real.py`/`kinlogic/exam.py`.
4. Website: follow `MODEL_UPGRADE_PLAN.md` (model downloads to the device; Cloudflare Pages only serves files).

## Rules that must hold
- Own from-scratch model only; fp32; no quantization/distillation; scraped real questions are evaluation only; locked
  test scored once for selection; decide on the full dev, never a subset.
- Commits authored by the owner (git config in repo); never add Claude co-author/session trailers. Push the model
  repo as the owner's GitHub account (see AGENTS.md "Owner's hard rules"); never act as another gh account.
- Never type passwords or create accounts; secrets (Gemini keys) only in `~/.config/kinthread/*.env`, never printed.
- Heavy jobs on Kaggle/Colab, not the 8 GB Mac. Keep tool calls and replies short (owner's usage limit).

## Current status (2026-10-27)

RoPE precomputation and SIMD/scalar pthread v10 reader builds are now in canonical `web/`; the model-repo mirror contains the same source and binaries. The website only selects threaded WASM on isolated devices reporting at least 8 GB and uses single-thread WASM on low/unknown-memory profiles. The pthread build reserves 384 MiB after smaller builds exhausted KV slots during branching beam search. 50-prompt JS-wrapper parity passes. The website now has a passing Node/Pyodide end-to-end three-link answer regression and a local Chrome smoke check returning paternal cousin in `9,632 ms` with SIMD/two threads; the score is tentative and uncalibrated. This is not full parity. The site stays on production v8; locked test remains 68.8% fair / 72.2% verified (<85%). Do not deploy, commit, or push until the documented browser parity and device gates pass.

## 2026-10-27 — v10-r8p production publication

Owner-authorized deployment supersedes the prior “keep production on v8” status. The exact fp32 v10-r8p checkpoint is active on Cloudflare Pages. Canonical source commit 11508ba is pushed to main; model mirror merge 57455fc and website commit 6f32924 are pushed as well. Production manifest identity: checkpoint SHA-256 38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52, content c4a1d87737c6, runtime f608eed15e9b. Live verification matched 66 model assets and 20 runtime files by byte count and SHA-256.

Accuracy remains 68.8% fair / 72.2% verified on the locked test, below 85%; broad browser parity, calibration, and mobile/low-memory budgets remain unfinished. The Chrome three-link answer is one tentative smoke check only.

## 2026-10-28 — balanced speech and long-story candidate

Mirrored the deterministic `generate_balanced_question_types.py` builder, its speech generator changes, plan, and `data/balanced_question_formats_v1.jsonl.gz` plus manifest from Kinthread-llm-model. The candidate has 3,750 rows: 50 per each of 25 canonical answer-relation buckets in each of direct speech, indirect speech, and long-story formats. Indirect wording shifts first-person references to the speaker's name and possessive. All are parsed and solved by the exact project readers and engine; every speech item has five parsed choices and exactly one correct option. Long-story lengths in this run are 80–187 words (configured bands 80–240). No Gemini usage or model training; checkpoint unchanged. This is a verified data candidate, not evidence of improved checkpoint accuracy. Data SHA-256: 315dccb5d2a9bdf0b3f32441601babb6b22d21eb73ad8b4545a51e7311e5cd88.

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
