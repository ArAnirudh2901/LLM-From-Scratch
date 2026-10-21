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

### v9: reader + exact engine (in progress, 2026-10-21)

The model no longer guesses the relation. The **reader** (same 13M GPT, warm-started from v8) turns the whole question
into family facts; the **engine** (`kinlogic/`, pure Python, exact) builds every family that fits them and answers.

| File | Role |
| --- | --- |
| `kinlogic/facts.py` | the fact language (`@1 mother #1 ; #1 brother #2 ; #2 son @0 ; ask @1 @0`), parser, decoding grammar |
| `kinlogic/kin.py` | relation words as moves; names every relation (paternal/maternal, great-*, in-laws, co-brother, Hindi) |
| `kinlogic/engine.py` | the solver: conventions, "only"/no-sibling identities, negations, counts, closed-family puzzles, claims, name-gender priors, fallback readings |
| `kinlogic/prepare.py` | question text -> tokens with placeholders (names, letters, "a man"); options split off; uses `kinlogic/data/lexicon.txt` |
| `kinlogic/generate.py` | synthetic exam questions with exact facts (statements, nested, speech, identity, introduce, coded, puzzle, sufficiency, claims, story) |
| `kinlogic/clutrr.py` | CLUTRR stories -> facts (`data/clutrr_facts/`) |
| `kinlogic/exam.py`, `explain.py` | whole exam question -> option + worked solution |
| `reader.py`, `train_reader.py` | vocab, encoding, cached constrained greedy/beam/sampling; supervised training (fp16 on CUDA, resume, eval) |
| `train_rl.py` | GRPO after supervised training: 8 sampled readings per question, engine-scored reward, KL to the supervised model |
| `eval/` | `exam_set.py` (hand-written, never trained on), `evaluate.py` (exam, CLUTRR test, RiddleBench, generated) |
| `eval/scrape_real.py`, `eval/run_real.py` | real exam questions from sites whose robots.txt allows it (IndiaBix, Examveda, prep articles); kept in git-ignored `data/real_questions/`, evaluation only, never training text |
| `colab/train_reader_colab.ipynb` | free-T4 training: clones branch `v9-reader` of Kinthread-llm-model |

Tests: `.venv/bin/python -m pytest 2_Training_Tiny_LLM/tests -q` (run from `2_Training_Tiny_LLM`). With gold facts the
engine answers 66/66 exam questions (`python -m eval.run_exam gold`) and matches every consistent CLUTRR label
(87% of CLUTRR test; the rest are CLUTRR labelling errors or ambiguous stories).

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
- 2026-10-20: no model changes. Kinthread (website) now ships a neural-network-cone loader driven by an exact
  progress follower, resumable downloads, share links and contradiction checks; its `vendor/kinship/` equals
  `2_Training_Tiny_LLM/web` here (SHA-256 manifest).
- 2026-10-21: v9 pipeline built (kinlogic engine, generator, reader training, eval, Colab notebook); not trained yet.
- 2026-10-21: audit fixes: RiddleBench 'A.V | B.O' options when the text starts with 'A family'; generations check with unlinked groups; puzzle member list vs stray placeholders; pair and data-sufficiency answers in eval/RL rewards; lexicon coverage test (rebuild with `python -m kinlogic.lexicon` after generator changes); new generated forms (addressee 'your', 'said to her husband', 'this man's', relative clauses, plural children, elder/younger); readings vote by confidence; exam set split dev/test (only dev guides choices).
- 2026-10-21: 263 real scraped questions (evaluation only). CPU check model (1% of planned training): 90% on generated questions, 24% on real ones; gaps found (missing words, 'A, B and C are sisters', 'cannot be definitely true') added to the generator.
- 2026-10-21: first Colab run (SFT `best.pt`): exam 51/66 (dev 80.6%, test 74.3%). Fixed `eval_clutrr`: a reading with no answer counted as right whenever the gold had no neutral word (inflated `clutrr_acc`), and the ceiling crashed on `None`; with gold facts accuracy now equals the ceiling (87.2%).
