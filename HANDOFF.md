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
