# Kinthread kinship reader: the whole journey (read this first)

Goal: answer exam-style blood-relation questions ("A is the father of B ... How is C related to A?", coded "A $ B means",
speech "Pointing to a man, she said ...", family puzzles) with our own from-scratch model, about 80% on real questions.
Owner rules: own model only (no pretrained third-party weights, no distillation, no quantization, fp32), free compute,
scraped evaluation questions never used for training, locked test scored once.

## Intuition (why the system looks like this)
- Split reading from reasoning. A small model is bad at multi-step kinship logic but good at turning text into facts.
  So the model only **reads**: question text -> fact language (`@1 mother #1 ; #1 brother @0 ; ask @1 @0`). An exact
  engine (`kinlogic/engine.py`) builds every family that fits and names the relation (paternal/maternal, in-laws,
  great-*, cousins, puzzles with member/couple/generation counts). Reasoning errors are then impossible; only reading
  errors remain.
- Names become placeholders (`@0`, `@1`), so the model never memorizes names; genders come from words, pronouns, and a
  name->gender table/model (`kinlogic/names.py`).
- Formulaic text is read exactly in code first (`kinlogic/codes.py` coded questions, `rules.py`, `speech.py`); the model
  reads the rest. Readings vote through the engine (8 beams + 32 extra when unsure).
- Every training row is engine-verified synthetic text (generator, rewrites, STaR on real-like keyed questions). Real
  scraped questions are evaluation only.

## Pipeline (inference)
`prepare` (names -> placeholders, options split) -> exact readers -> v10 reader (16x384, RoPE, RMSNorm, SwiGLU, copy
head, 39.5M params, sub-word vocab 28,836) with grammar-constrained beam -> engine -> exam layer (vote, conventions,
option matching, "not related", "Either X or Y"). Answer-time boosters (`eval/run_real.py` flags): `--adaptive 1.0`
(re-read when unsure), `--hybrid` (exact clause facts merged), `--cover` (readings must use every named person),
`--para N` (vote over renamed copies), `--split_search` (sentence-level beam checked by engine), `--verifier`
(LEVER-style learned reranker, `kinlogic/verifier.py`, trained by `train_verifier.py` on keyed non-eval questions).

## Timeline (real-question accuracy, fair items)
| step | what changed | result |
| --- | --- | --- |
| v8 QA (Oct 19) | 13M GPT answers directly (CLUTRR) | CLUTRR test 44.8%; still the website model |
| v9/v10 r1-r3 | reader + exact engine; verified pool 1.36M; rewrites | real 38% -> 42% |
| pipeline | exact readers, conventions, scoring fixes | r3 -> 50.5% locked |
| pre-v1 + r4 | own English pretraining (TinyStories+FineWeb-Edu), fine-tune | locked test 61.2% |
| r5 / r6a / r7 | fact grammar growth, STaR, teacher rows, RL tried | r5 locked 62.7%; r7 dev 72.1% (717) |
| r8 / r8p v5 | dataset v8 balance, prefix-LM option, no-relation | old dev subset: r7 72.9, r8 73.5, r8p 74.3 |
| final (Oct 25-26) | per-type balanced mix (build_mix), leak guard, verifier + boosters, picked on dev | see `FINAL.json` (Kaggle final run) |

## Data (how to rebuild)
- Verified pool `data/dataset_v7` (Drive / model repo); extras: rewrites, STaR r2/r3, teacher rows, `pairs_v1`
  (+8k no-relation rows from `build_norel.py`).
- `python build_mix.py --out data/mix_r8.jsonl.gz --strict`: rakes every verified row to real exam shares (kinds, 45
  phrase types, fact types, 43 named answers, people, length), each type has a minimum count, fills gaps with
  engine-checked rewrites/joins, drops eval copies, gates. `mixture_audit.py` audits the whole training mixture.
- `eval/leak.py`: training drops any row matching an eval question (masked exact or 3-gram Jaccard >= 0.8);
  `data/eval_fingerprint.json.gz` = hashes only.
- Eval sets: real questions (`data/real_questions`, private; dev/test split by hash, test locked), keys checked by
  Gemini (`eval/verify_keys.py`: 1,235 confirmed, 35 corrected, 22 ambiguous), construction suite (25 sentence types),
  held-out rewrites, unseen synthetic set.

## Lessons (do not repeat)
- Consistency checks alone never pick the right reading (engine accepts consistent-but-wrong readings): use a learned
  verifier. Model ensembles did not help. RL/GRPO never beat supervised. Decide on the full dev only (300-question
  subsets misled by 2 points). Exact readers fit dev wording ~13 points better than unseen wording: confirm on locked test.
- A correct reading exists for ~75% of dev questions but the vote picks it ~70%: selection is the lever.
- Remaining errors: long puzzles (7+ family words, <40%), ~10% wrong/ambiguous keys, unusual phrasing.
- Kaggle notebooks report "Successful" even when a `!python` cell crashed: always read the log.

## Infra
- Kaggle (account anirudharavalli, 30 GPU h/week): r8 notebook `notebookf6d1e34e7f`, r8p `notebook99ae0586bd`
  (Save & Run All; clone Kinthread-llm-model `v9-reader` HEAD), final selection `colab/final_kaggle.ipynb` (import into
  notebook `notebookf8798397a0`; inputs: private dataset `kinthread-real-questions`), writes `FINAL.json`.
- Colab accounts by authuser index (memory `colab-accounts.md`); Gemini keys `~/.config/kinthread/gemini*.env` (10).
- Repos: LLM-From-Scratch (code), Kinthread-llm-model (code subtree + weights/data), Kinthread (website).

## Website (not yet on the new model; detailed plan: ~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md)
Kinthread (`~/Web_Dev/kinthread`, Cloudflare Pages) still runs the v8 QA model in WASM. To ship the new reader:
(a) browser: export v10 reader to ONNX fp32 (~160 MB, split < 25 MB parts with SHA manifest as the site already does),
onnxruntime-web for the grammar-constrained beam, kinlogic (pure Python) via Pyodide or a JS port; or (b) a hosted
Python API (container/Space) with the Pages site as frontend. Weights: Kaggle output of the final pick -> model repo
`runs/<final>/best.pt.part-*` (+ verifier.json, FINAL.json). R2 needs billing enabled on the Cloudflare account.
