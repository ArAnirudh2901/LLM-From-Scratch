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
| final (Oct 25-26) | per-type balanced mix (build_mix), leak guard, verifier + boosters, picked on full dev | r8p + all boosters: dev 73.9% fair / 77.2% verified; locked test 68.8% / 72.2% vs r7 66.2% / 70.5% (+2.6 / +1.7 points) |

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
- Kaggle (account anirudharavalli): r8 notebook `notebookf6d1e34e7f`, r8p `notebook99ae0586bd`; final selection
  `notebookf8798397a0`, run `362159765`, completed with `FINAL.json`. Full dev: 748 questions (714 fair, 650 verified);
  r8p base 73.0% fair / 76.5% verified, r7 72.7% / 76.5%, r8 72.1% / 75.5%. The `all` setting won dev at 73.9% /
  77.2%. Locked test, scored once after selection: winner 68.8% fair / 72.2% verified; r7 baseline 66.2% / 70.5%.
- Colab accounts by authuser index (memory `colab-accounts.md`); Gemini keys `~/.config/kinthread/gemini*.env` (10).
- Repos: LLM-From-Scratch (code), Kinthread-llm-model (code subtree + weights/data), Kinthread (website).

## Website migration (2026-10-26)
Kinthread (`~/Web_Dev/kinthread`, Cloudflare Pages) still runs the v8 QA model in WASM. The final pick is v10-r8p,
step 6,000 (SHA-256 in AGENTS.md). The fixed deployment route remains browser-only: exact fp32 weights served in
small static parts, custom WASM reader forward pass, and the Python `kinlogic`/reader pipeline in Pyodide. No hosted
inference service or model compression. The website implementation plan is `~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md`.

Started Phase 1 with `export_reader_web.py`: exact fp32 export is 39.50M weights, 158.0 MB raw and 126.1 MB gzip;
SHA-256 manifest generated. Site resource tiers are implemented and checked: low-power devices use one WASM thread
and serial downloads; balanced devices cap at two; high-capacity devices at four. `bun run check` passed, 130 tests
plus static build. Prototype WASM reader (`web/reader_engine.c`, `web/reader.js`) now supports prefix attention,
cached decoding and the copy head in exact fp32, with independent SIMD/scalar builds. One PyTorch golden prompt
(8-token prefix, one decode) gave <=1.15e-5 max absolute difference across hidden states, tied logits, copy keys and
mixed probabilities, both builds. This is not the 50-prompt parity suite; no v10 browser runtime has shipped. Continue
parity, memory, latency and real Chrome checks before Cloudflare deployment. Constrained devices should use
worker-based bounded search; high-cost `para=2`/`k=16` is reserved for high tier or explicit deep-check, with the
quality trade-off visible.

- 2026-10-26: v10 browser-forward prototype extended from one prompt to 50 deterministic generated prompts (13–101 tokens); scalar and SIMD WASM both match PyTorch within 2.86e-5 on prefix/decode/logits/copy-key surfaces and 7.82e-5 on mixture probabilities. This is primitive parity, not end-to-end browser answer parity.

- 2026-10-26 (continued): same 50-prompt fixtures pass through `ReaderWasm` wrapper on SIMD and scalar, including prefill, dynamic cache growth and one-token decode; maximum log-prob difference 2.27e-4 and probability difference 7.82e-5. Grammar-constrained generation remains the next major phase.

- 2026-10-26 (website release verification): Kinthread's local Chrome preview passed success, corrupted-model-file, and mismatched-release-manifest scenarios for its release/update plumbing. The website is still pinned to v8; v10-r8p forward/decode parity does not establish end-to-end answer parity. Website details and the remaining v10 gate are in `~/Web_Dev/kinthread/MODEL_UPGRADE_PLAN.md`.
- 2026-10-27 (threaded v10): canonical WASM reader now precomputes RoPE values and builds SIMD/scalar pthread modules; website gates threads to isolated devices with >=8 GB reported memory and keeps low/unknown-memory devices single-threaded. 384 MiB fixed memory is needed for branching beam KV slots (192/256 MiB variants exhausted memory on a deep story). The 50-prompt JS wrapper parity suite passes. Node/Pyodide end-to-end and native Chrome local smoke checks both answer the three-link cousin case correctly; Chrome runtime `f608eed15e9b` took 9,632 ms using SIMD/two threads (tentative, uncalibrated score). This is one case, not broad parity. Full website check passes 139 tests with one intentional skip, lint and static build. Production stays v8; locked-test accuracy remains 68.8% fair / 72.2% verified, below 85%; no deploy, commit or push.
