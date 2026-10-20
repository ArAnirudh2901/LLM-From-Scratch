# LLM from Scratch: handbook for any agent or model

Keep this file current. When a checkpoint, result, command or decision changes, update it here (and the Claude
memory noted in CLAUDE.md) and add a line to the handoff log, so the next model can continue cold.

## Layout

- `1_Transformer_arch/`: attention and transformer building blocks with tests (part 1).
- `2_Training_Tiny_LLM/`: the owner's 13M-parameter GPT (`model_gpt.py`) trained from scratch, and the kinship QA
  project built on it (part 2). Canonical source of the model code.
- `Architecture/`: diagrams used by the README.

## Related repos

| Repo | Local path | Holds |
| --- | --- | --- |
| [LLM-From-Scratch](https://github.com/ArAnirudh2901/LLM-From-Scratch) (this) | `~/Generative_AI/LLM_from_Scratch` | everything; code only (checkpoints, data, exported weights are git-ignored here) |
| [Kinthread-llm-model](https://github.com/ArAnirudh2901/Kinthread-llm-model) | `~/Generative_AI/Kinthread-llm-model` | `2_Training_Tiny_LLM` with history (subtree) + final checkpoints, data, exported model |
| [Kinthread](https://github.com/ArAnirudh2901/Kinthread) | `~/Web_Dev/kinthread` | the website; final model in `vendor/kinship/`, synced from `2_Training_Tiny_LLM/web` |

Publishing a code change to the model repo:
`git subtree split --prefix=2_Training_Tiny_LLM -b model-split`, then in the model repo
`git pull --no-rebase $HOME/Generative_AI/LLM_from_Scratch model-split`. After a retrain also copy the checkpoints,
`web/model/*` and data there and run `python verify_model.py`.

## Kinship QA project (`2_Training_Tiny_LLM/`)

Answers "how is B related to A?" about a short family story (CLUTRR-style). Names become entity tokens (`@0`, `@1`,
…); the model writes a chain of facts before the answer, found with a constrained beam search; a step checker
(`<nofact>`) and a co-mention rule let it say "unknown".

| File | Role |
| --- | --- |
| `qa_data.py` | data, tokens, constrained beam search (`beam_predict`) |
| `train_qa.py`, `ask_qa.py` | training; asking questions (`--query A B`: "B is A's ___"; defaults point at the best checkpoints) |
| `name_tagger.py` | finds person names in plain text (same GPT class) |
| `fast_qa.py` | reference for the browser engine: same beam search with a shared key/value slot pool |
| `eval_pipeline.py` | end-to-end evaluation |
| `export_web.py` | exports both models for the browser: exact fp32, byte-shuffled gzip; writes `model_manifest.json` |
| `verify_model.py` | checks model files against `model_manifest.json` (exit 1 on any mismatch) |
| `web/` | browser engine: `engine.c` (WASM via `build.sh`, needs emcc; six builds), `kinship.js`, `model/`, `test/test_node.mjs` |

Best checkpoints: QA `runs/clutrr-qa-v8/best.pt` (val 89.8%, test 44.8%, val with distractors 78.2%); tagger
`runs/name-tagger-v4/tagger.pt`. Weak spots: long chains, friend/stranger detection, ambiguous places.
Python with torch: `.venv/bin/python` in this repo (needs pandas + pyarrow for the parquet data).

## Owner's hard rules

- No quantization, no distillation or smaller model: keep the v8 weights and exact fp32; optimise the runtime only.
- Weights, checkpoints and data stay git-ignored here; they are published only in Kinthread-llm-model (and the
  exported model in Kinthread).
- Commit or push only when asked. Default branch `main` (includes the kinship work); `kinship-qa-web` mirrors it.
- Commits are authored as `Anirudh Aravalli <anirudh30012005@gmail.com>` (ArAnirudh2901; set in git config). Never
  add `Co-Authored-By: Claude` or `Claude-Session` lines. If `gh` is signed in as another account, push with the
  owner's token (see Kinthread's AGENTS.md) and never act as the other account.
- The machine is an 8 GB Mac: keep beam and mining batches at 64; use `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35`.
- Production quality; suggest improvements and flag risks unasked. Code comments brief.

## Handoff log

- 2026-10-19: v8 QA model and v4 tagger; browser engine (WASM, 26 ms per question on an M2 in Chrome) matches Python.
- 2026-10-20: website renamed Kinthread; exam-style input handled on the website side. `main` fast-forwarded to
  include the kinship project; Kinthread-llm-model created (subtree + checkpoints + data + exported model); all
  history rewritten so only ArAnirudh2901 is a contributor.
