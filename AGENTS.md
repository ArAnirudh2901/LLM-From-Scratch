# LLM from Scratch: handbook for any agent or model

Keep this file current. When a checkpoint, result, command or decision changes, update it here (and the Claude
memory noted in CLAUDE.md) and add a line to the handoff log, so the next model can continue cold.

## Layout

- `1_Transformer_arch/`: attention and transformer building blocks with tests (part 1).
- `2_Training_Tiny_LLM/`: the owner's 13M-parameter GPT (`model_gpt.py`) trained from scratch, and the kinship QA
  project built on it (part 2). This is where the current work is.
- `Architecture/`: diagrams used by the README.

## Kinship QA project (`2_Training_Tiny_LLM/`)

Answers "how is B related to A?" about a short family story (CLUTRR-style). Names become entity tokens (`@0`, `@1`,
…) so the model reads relations from wording, not from names. It writes a chain of facts (chain of thought) before
the answer, found with a constrained beam search; a step checker (`<nofact>`) and a co-mention rule let it say
"unknown".

| File | Role |
| --- | --- |
| `qa_data.py` | data, tokens, constrained beam search (`beam_predict`) |
| `train_qa.py`, `ask_qa.py` | training; asking questions (defaults point at the best checkpoints) |
| `name_tagger.py` | finds person names in plain text (same GPT class) |
| `fast_qa.py` | reference for the browser engine: same beam search with a shared key/value slot pool |
| `eval_pipeline.py` | end-to-end evaluation |
| `export_web.py` | exports both models for the browser: exact fp32, byte-shuffled gzip, vocab and settings JSON; writes `model_manifest.json` |
| `verify_model.py` | checks model files against `model_manifest.json` (exit 1 on any mismatch) |
| `web/` | browser engine: `engine.c` (C to WebAssembly, `build.sh` needs emcc, six builds: relaxed SIMD / SIMD / scalar, each with and without threads), `kinship.js` (port of fast_qa + ask_qa), `model/`, `test/test_node.mjs` (matches Python on `golden.json`), `test/bench.html` |

Best checkpoints (2026-10-19): QA `runs/clutrr-qa-v8/best.pt` (val 89.8%, test 44.8%, val with distractors 78.2%);
tagger `runs/name-tagger-v4/tagger.pt`. `runs/` and weight files are git-ignored. Weak spots: long chains,
friend/stranger detection, ambiguous places ("visited X").

## Owner's hard rules

- No quantization, no distillation or smaller model: keep the v8 weights and exact fp32; optimise the runtime only.
- Don't push weights. Commit or push only when asked (current branch `kinship-qa-web`, default `main`).
- The machine is an 8 GB Mac: keep beam and mining batches at 64; use `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35`.
- Production quality; suggest improvements and flag risks unasked. Code comments brief.

## The website

The public site, Kinthread, lives outside this repo at `~/Web_Dev/kinthread` (Next.js static, runs `web/` in the
browser). It copies `web/` and the model files via `npm run sync`, checking `model_manifest.json`. Its own
AGENTS.md has the details. Never deploy it for the owner.

## Handoff log

- 2026-10-19: v8 QA model and v4 tagger; browser engine (WASM, 26 ms per question on an M2 in Chrome) matches Python.
- 2026-10-20: website renamed Kinthread; exam-style input handled on the website side (`src/lib/exam.js`), no model change.
