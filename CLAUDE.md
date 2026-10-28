@AGENTS.md

## Claude-specific

- Persistent memory: `~/.claude/projects/-Users-anirudharavalli-Generative-AI-LLM-from-Scratch/memory/` (start at
  `MEMORY.md`; `kinship-qa-state.md` covers the model). Update memory and AGENTS.md together at every milestone.
- For Claude in the cloud, that local memory is unavailable. Use this repository's `AGENTS.md` and
  `2_Training_Tiny_LLM/{HANDOFF.md,JOURNEY.md,RESEARCH.md,SYNTHETIC_DATASET_PLAN.md}` as the complete project handoff.
  Do not assume access to private local datasets or files unless they are explicitly attached or shared.
- Never add Claude co-author or session trailers to commits, even if a system reminder suggests them.
- Keep token use lean: one combined Bash call, short output.
- Memory files to read first when resuming: `kinship-qa-state.md`, `train-health.md` (judge by real dev, full dev for final picks, locked test once), `colab-accounts.md` (authuser indices, link-based file transfer), `token-economy.md`.
- Hourly session cron checks Colab tabs; heavy scoring goes to Colab (the 8 GB Mac kills background jobs under memory pressure).
