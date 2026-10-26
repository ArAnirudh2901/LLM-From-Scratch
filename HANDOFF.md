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

## Update 2026-10-26
Final pick is v10-r8p step 6,000; the selected Kaggle checkpoint and `/Users/anirudharavalli/Downloads/best.zip` are byte-identical (SHA-256 `38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52`). Full-dev winner: `all` boosters; 73.9% fair / 77.2% verified on dev and 68.8% / 72.2% on the locked test (scored once). See the new latest entry in AGENTS.md and the final row in JOURNEY.md.

Website migration started in `~/Web_Dev/kinthread`: canonical code now has `export_reader_web.py`, whose smoke export gave 39.50M exact-fp32 weights (158.0 MB raw, 126.1 MB gzip). Website is still v8; the v10 reader WASM and Pyodide pipeline are not integrated into its answer path. Website runtime added low/balanced/high hardware tiers; `bun run check` passed (130 tests and static build). Continue `MODEL_UPGRADE_PLAN.md`; do not deploy until end-to-end parity/device budgets pass.

## Update 2026-10-26 (v10 browser runtime prototype)
Canonical source now includes `web/reader_engine.c`, `web/reader.js`, SIMD/scalar WASM build targets, and PyTorch golden scripts. A one-prompt test covers v10 prefix attention, cached next-token decode, tied logits, copy keys and mixed vocabulary/copy probabilities; both WASM builds are within 1.15e-5 max absolute difference. This is only a numerical smoke test, not 50-prompt parity or a website integration. The website still serves v8. Continue its `MODEL_UPGRADE_PLAN.md`; low-power hardware tiers in the site currently cap threads/download concurrency, while v10 bounded-search behavior still needs implementation and a measured quality/latency decision. Do not deploy until the Pyodide/question pipeline, browser parity, memory, and real Chrome phone checks pass.

## 2026-10-26 website release verification
Kinthread's v8 release/update integrity flow was verified in local Chrome: successful replacement and inference, rejection
of a same-size corrupted model part while keeping the previous install, and startup blocking on a release-manifest
mismatch. The notice was checked in dark/light themes; a mobile viewport was not separately tested. Website
`bun run check` passed (134 tests, lint, static build). The site remains v8 because full v10 grammar/beam + `kinlogic` + exam
answer parity is not integrated. Details: `~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md`. No deploy, commit, or push.

## 2026-10-26 export follow-up
`export_reader_web.py` ran against `Kinthread-llm-model/runs/v10-r8p-final/best.pt` with `FINAL.json`, `verifier.json`, and `runs/names-v1/name_table.tsv.gz`, writing `/tmp/kinthread-v10-final`. Manifest generated; files are not committed or synced to the site. The site's low-power concurrency changes were deployed with v8 to Cloudflare preview/deployment `https://af2acf01.kinthread.pages.dev`; v10 remains pending full browser pipeline parity.

## 2026-10-26 parity follow-up
`web/test/make_reader_parity.py` and `test_reader_parity_50.mjs` cover 50 deterministic generated prompts (13–101 tokens, mixed kinds). Both scalar and SIMD C/WASM builds pass against PyTorch: worst errors prefix 2.86e-5, decode 1.38e-5, logits 1.53e-5, copy keys 2.38e-5, mixed probabilities 7.82e-5; log-prob tail 2.27e-4. Still pending JS wrapper, shared beam slots, longer real prompts, full answer pipeline and low-power latency.

Follow-up: the same 50 prompts now pass through `ReaderWasm` JS wrapper (prefill, dynamic KV growth, one cached decode), both SIMD and scalar; maximum log-prob difference 2.27e-4 and probability difference 7.82e-5. This closes wrapper-level primitive coverage only, not beam search or answer output.
