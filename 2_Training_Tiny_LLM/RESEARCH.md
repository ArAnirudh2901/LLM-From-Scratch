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
| Not used | Distillation from big teachers (Gemma 3, Qwen3), multi-token prediction and mixture-of-experts (DeepSeek-V3), selective tokens (Rho-1) | Against the from-scratch rule, or no gain at our scale |

Measurement: real questions split into dev (error analysis only) and a locked test half (`eval.run_real`), broken
scrapes reported apart (`eval/real_clean.py`); synthetic unseen set (`eval/unseen_set.py`) and held-out rewrites.
