# Research behind the reader (2026-10-23)

Goal: our own from-scratch reader (no third-party weights) at ~90% on real exam questions. r3 (12M, trained only on
generated questions) scores 45.5% on the locked half of the real questions; its misses are reading errors on phrasing
the generator never writes. The papers below decide the next steps.

| Source | Finding | What we do |
| --- | --- | --- |
| Furrer et al. 2020, *Compositional Generalization in Semantic Parsing: Pre-training vs. Specialized Architectures* (arXiv 2007.08970) | On the harder compositional tasks, pre-training gives significant gains over comparable non-pre-trained models; specialized architectures do not | Pretrain our reader on English first (`pretrain_reader.py`) |
| Eldan & Li 2023, *TinyStories* (2305.07759) | Models of a few million parameters learn fluent English from simple, varied text | TinyStories V2 is half the pretraining text |
| BabyLM Challenge 2023-2025 (ACL Anthology) | 10M-100M words of pretraining already give usable language skill | 400M tokens is enough for a ~40M model on free GPUs |
| MobileLLM (Liu et al., ICML 2024, 2402.14905) | Below 1B parameters, deep-and-thin beats wide-and-shallow; share embeddings; grouped-query attention | 16 layers x 384 (39.5M), tied embeddings |
| SmolLM2 (2502.02737), SmolLM3 (HF blog), MiniCPM (2404.06395), OLMo 2 (2501.00656), Llama 3 (2407.21783) | Warmup-stable-decay schedule; high-quality domain data mixed into the decay (annealing / mid-training); no weight decay on embeddings | WSD; verified kinship text is half of each batch in the decay (`--anneal`); embeddings and norms undecayed |
| Xu et al. 2020, *AutoQA* (2010.04806) | A parser trained only on synthetic data plus filtered paraphrases reaches 62.9% on natural questions, 6.4 points below one trained on expert annotations | Verified rewordings (`kinlogic/reword*.py`) |
| SymGen (2305.13917); Phi-1 *Textbooks Are All You Need* (2306.11644); Phi-4 (2412.08905) | A large model generating whole new structured training data, checked by an agreement verifier, trains strong small models | New exam-style questions written by an LLM, kept only when independent answers agree |
| STaR (2203.14465); Llama 3 rejection sampling; *A Minimalist Approach to LLM Reasoning* (RAFT, 2504.11343); DeepSeek-R1 (2501.12948) | Training on the model's own samples that reach the right answer works as well as GRPO; GRPO's edge is mostly dropping prompts it never gets right | For answer-keyed questions without facts: sample readings, keep those whose engine answer matches the key, train on them; our engine is an exact verifier |
| Kimi K2 (2507.20534) | Rephrasing high-value text in many styles, each rewrite checked for fidelity, beats repeating it for more epochs | Scale verified rewording from 7.8k to 100k+ questions (Gemini and a Qwen3 rewriter on free T4s); our fidelity check is exact (the engine must give the same answer and facts) |
| Qwen2.5-Math (2409.12122), DeepSeekMath (2402.03300) | Self-improvement loop: the model writes candidate solutions, a verifier keeps the right ones, retrain, repeat; verifier-ranked best-of-N at inference; RL mostly turns "right in some samples" into "right first time" | STaR rounds on answer-keyed questions (`star.py`); more sampled readings at test time ranked by the engine (consistency, every name read, an option matched) |
| DeepSeekMath (2402.03300) | A classifier mines domain text from the web; domain tokens count more than total tokens | Mine family-relation passages from FineWeb for the kinship anneal |
| Qwen3 (2505.09388), DeepSeek-R1 | Small models gain most from a strong teacher's data; RL with verifiable rewards needs only a few thousand hard question-verifier pairs | Teacher data stays data only (SymGen questions, rewrites), never weights; GRPO after STaR on the hardest answer-keyed questions |
| MiniCPM4 (2506.07900) | Data quality judged by short annealing runs on a small model before full training | Check each new data source with a short fine-tune and the dev split before adding it |
| DeepSeek-V4.1-Flash (2609.19969) | Post-training gains came "almost entirely" from what the model trains on: synthesized verifiable tasks with reference answers, rigorous dedup and difficulty calibration, under a plain SFT + RL recipe; model-written text that only restates known content is filtered as "implicit duplication" | Keep the plain recipe; spend effort on verified, diverse, difficulty-calibrated questions; dedupe rewrites by meaning (canonical facts), not by text |
| GLM-5 (2602.15763) | RL data is difficulty-filtered: problems the current model rarely or never solves but a stronger solver can; final on-policy distillation from earlier stages' checkpoints undoes forgetting between stages | Keep answer-keyed questions our reader gets wrong in most samples yet right in some (the learnable frontier); after STaR/RL rounds, distill from our own previous best checkpoint so skills already learned stay |
| Kimi K3 (2607.24653) | Tasks synthesized from a hierarchical concept graph so coverage is controlled, not left to chance; data mixture rates set by small ablations; Per-Head Muon optimizer (Muon for matrices, AdamW for embeddings, head and norms, as in DeepSeek-V4) | Question synthesis driven by a feature grid (relation words x phrasing style x structure x distractors x name style), weighted by the dev error rate of each cell; try Muon on a short fine-tune ablation before any long run |
| Our additions | Their verifiers compare meaning approximately; ours is exact (the engine must give the same facts and answer). Reader errors are often one wrong word, so we add minimal pairs (his/her, son/only son, of/'s) that flip the answer, and distill the exact readers (codes, rules, speech: 100% precise where they apply) into the model on new, never-tested text | |
| Looped / recursive models: Universal Transformers (1807.03819), PonderNet (2107.05407), Looped Transformers (2301.13196, 2409.15647), HRM (2506.21734), TRM (2510.04871), Huginn recurrent depth (2502.05171), Ouro LoopLM (2510.25741), Mixture-of-Recursions (2507.10524) | One block reused many times gives depth without parameters; a halting signal (ACT, PonderNet's lambda prior, TRM's halting head trained on "is the current answer right", Ouro's exit gate) decides when to stop. TRM: a 2-layer, 7M network refining its answer up to 16 times with a loss at every step (deep supervision) beats far larger LLMs on Sudoku/maze/ARC from ~1,000 examples; looped models generalize to longer inputs and manipulate knowledge better | Our engine already does the exact reasoning, so recursion goes where we fail: reading. (1) Reading refinement: the reader sees the question, its last reading and the engine's verdict, and writes a corrected reading, trained with deep supervision on corrupted readings; the engine is the halting check. (2) A looped reader (shared blocks, adaptive exit) for long multi-clause puzzles, tested as an ablation at equal parameters |
| Open-model architectures (configs: Qwen3, Llama 3.1, DeepSeek-V3, Kimi-K2, GLM-4.5, SmolLM2/3, MiniCPM4, OLMo 2, Phi-4, Mistral, Falcon-H1, Granite 4, LFM2, Kimi-Linear, Ouro, Huginn) | Small models converge on deep-thin decoders (SmolLM2-135M: 30 x 576), RMSNorm, SwiGLU, RoPE with high base, grouped-query attention, tied embeddings, QK-norm (OLMo 2, Qwen3); large ones add MoE + latent attention (efficiency, not reasoning); newer ones mix linear/state-space layers or loop their blocks | pre-v1 (16 x 384, RMSNorm, SwiGLU, RoPE, tied) already matches; candidates: QK-norm, grouped-query attention, a looped variant |
| Execution-verified LLM labels: Learning from Executions (Wang et al. 2021), PaD program-aided distillation (NAACL 2024), SQuaD-SQL, LEVER, MBR-exec | A small parser trained on real-distribution questions labelled by a large model, kept only when execution matches the known answer, closes most of the domain gap | `teach.py`: a frontier model writes facts for answer-keyed questions; kept only when our engine reaches the key (wrong keys and wrong readings both drop out); never on evaluation questions |
| Not used | Distillation from big teachers (Gemma 3, Qwen3), multi-token prediction and mixture-of-experts (DeepSeek-V3), selective tokens (Rho-1) | Against the from-scratch rule, or no gain at our scale |

Measurement: real questions split into dev (error analysis only) and a locked test half (`eval.run_real`), broken
scrapes reported apart (`eval/real_clean.py`); synthetic unseen set (`eval/unseen_set.py`) and held-out rewrites.

## 2026-10-24: why r5 did not beat r4 on real questions, and what the literature says to do

Measured: synthetic held-out 96-99.8% but real 62-67% (r4 and r5 alike): the reader fits the generator's style
(template memorization), not real wording. On dev, exact readers cover 54% at 89%; the model covers 46% at 42%.
- Chu et al. 2025, *SFT Memorizes, RL Generalizes* (ICML): SFT memorizes training rules and styles; RL with an
  outcome reward generalizes to unseen variants. -> GRPO with the engine's answer as reward on real-style keyed questions
  (train_rl.py), evaluated on real dev, after SFT.
- Saparina & Lapata 2024 (EACL), *Improving Generalization in Semantic Parsing by Increasing Natural Language Variation*:
  LLM-written variation of training questions fixes brittleness to rewording. -> reword --styles (9 exam styles).
- NAACL 2024, *Simple and effective data augmentation for compositional generalization*: sample meaning representations
  broadly (uniform), back-translate to text; beats sampling from the training distribution.
- Data Factors for Better Compositional Generalization (2023): diverse and harder examples beat more examples.
- Consistency Regularization Training for Compositional Generalization (ACL 2023): consistent predictions across
  perturbed inputs; word dropout (--unk_p) is the cheap form.
- ReST^EM / STaR (Zelikman 2022; Singh et al. 2023): self-training filtered by a verifiable answer: star.py at scale.
- Prefix LM / encoder-decoder (T5, UL2; Furrer et al. 2020 on compositional semantic parsing): the input read
  bidirectionally -> model_reader prefix_lm option.
- Constrained decoding and output close to the input order (Shin et al. 2021; Rongali et al. 2022) -> --of_form.
- Architecture surveys (Hymba 2024 meta tokens + SSM heads, Differential Transformer 2024, MobileLLM deep-thin): gains
  are in pretraining-scale language modelling; none targets style memorization in a small fine-tuned parser.
