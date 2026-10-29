# Synthetic data candidates: balanced formats and Gemini rewrites

## Purpose

Add question forms that the current training mix underrepresents: direct speech, indirect/reported speech, and
long multi-clue family stories. Keep the three forms equally represented and balance the answer relation types
inside each form. This is a training candidate, not an accuracy claim and not a replacement for real-exam evaluation.

## Quotas

The generator defaults to 50 accepted examples in every `format × relation_type` cell:

| Relation type | Rows per format |
| --- | ---: |
| Father, mother, son, daughter | 50 each |
| Brother, sister, husband, wife | 50 each |
| Grandfather, grandmother, grandson, granddaughter | 50 each |
| Uncle, aunt, nephew, niece | 50 each |
| Cousin | 50 |
| Father-in-law, mother-in-law, brother-in-law, sister-in-law, son-in-law, daughter-in-law | 50 each |
| Other affinal relation (co-relations and extended by-marriage paths) | 50 |
| No family link | 50 |

There are 25 canonical answer types. The complete output contains 1,250 direct-speech, 1,250 indirect-speech, and
1,250 long-story questions (3,750 total). Direct and indirect questions are paired examples over the same graph,
names, question direction, answer options, facts, and gold answer; only the speech form changes. Indirect speech shifts
first-person wording into the speaker's name and possessive (for example, “I am ... of my father” becomes “Ravi is ...
of Ravi's father”). No-link examples
also have true facts about both people in separate family components, so the answer depends on recognizing that the
stories do not connect them. In no-link indirect speech, the speaker is named explicitly in the reported clause; this
avoids assigning an ambiguous third-person pronoun to the person being pointed at.

Each long-story relation type is split into three length bands: 17 rows at 80–120 words, 17 at 121–170 words, and 16
at 171–240 words. The final word count includes the question. In the checked 1,250-story output, the realized
range is 81–187 words; the extended band is an acceptance range of 171–240, not a promise that every upper bound
is reached. Long stories contain multiple connected family facts;
some also include only/negation constraints when those are true in the source family graph. No-link stories describe
two independent families.

## Generation and acceptance

Run from the model repository:

```bash
python3 generate_balanced_question_types.py --per-cell 50
```

The generator uses the repository's family-graph generator, not a language model, to determine ground truth. It
spends zero Gemini requests and therefore does not use the free-tier quota. It draws varied Indian, international,
mixed, and letter names from `kinlogic.Names`.

Every candidate must pass the text reader and exact family solver before it can count toward a quota:

- Direct and indirect speech are independently parsed by `kinlogic.speech.read`.
- Long stories are independently parsed by `kinlogic.rules.read`.
- Parsed facts must be valid fact-language input and solve to exactly one answer in the required relation type.
- Speech MCQs must include the exact solver answer exactly once. Paired direct/indirect rows must have identical
  parsed facts and gold answers.
- Duplicate question text is rejected. Long stories outside their length band are rejected.
- Only verified accepted rows count; candidate attempts do not count toward balance.

## Optional surface-rewrite prompt

The current generator does not call Gemini. If a later run uses Gemini to increase wording variety, use it only to
rewrite a code-verified source graph. Never ask it to invent the family graph, answer, options, or quota. Then send
each rewrite through the same exact readers and solver; reject the whole pair if any fact or answer changes.

```text
You are rewriting a verified competitive-exam family-relation question. The supplied facts and query are the full
ground truth. You are not solving the question and must not add, remove, reverse, weaken, or strengthen any fact.

Requirements:
1. Keep every supplied person name exactly as written and use no additional people.
2. Preserve each relation, direction, gender, maternal/paternal side, only/exactly-one constraint, and negation.
3. Preserve the exact ordered query: how TARGET is related to REFERENCE. Do not state or hint at the answer.
4. Produce one direct-speech version with a quoted utterance and one indirect/reported-speech version. The two
   versions must express identical facts and ask the same query. Keep first/third-person pronouns anchored correctly.
5. Produce one long-story version of 80–240 words. Include all supplied facts, distribute them through a coherent
   family scenario, and do not add relationship claims or unsupported filler events. The story must still leave the
   answer to be inferred from the facts.
6. Do not write answer choices; the dataset builder creates and verifies choices separately.
7. Output valid JSON only, with exactly these keys:
   {"direct_speech":"...", "indirect_speech":"...", "long_story":"..."}

Verified names: {{NAMES_JSON}}
Verified facts in text order: {{FACTS_JSON}}
Exact query target: {{TARGET}}
Exact query reference: {{REFERENCE}}
Canonical answer type (for quota routing only; do not reveal): {{RELATION_TYPE}}
```

Use this prompt only for surface variation. The programmatic generator is the default because it is cheaper and
keeps graph truth independent from model-generated text.

The output is `data/balanced_question_formats_v1.jsonl.gz`, with an adjacent JSON manifest containing exact format
and answer-type counts, answer-label counts, story-length counts, constraint coverage, rejection reasons, and SHA-256.
The data stays separate from the existing Gemini reword pool so its balance is auditable. Combine it into a training
run only after reviewing this manifest and a development-set evaluation; do not use the locked test to select or tune
synthetic data.

## Limits

Equal quotas guarantee representation, not 85% real-world accuracy. The exact answer types are balanced at the
canonical level above; side-specific wording (for example, maternal versus paternal aunt) remains visible in the
manifest's exact-label counts. The selected v10-r8p checkpoint is not retrained by this generator and remains
unchanged until a separately authorized training run.

## Separate Gemini rewrite candidate (not format-balanced)

The model repository also contains `data/gemini_synthetic_v1.jsonl.gz`, a separate pool of **1,015 accepted
Gemini rewrites** and `data/gemini_synthetic_v1.focus.jsonl.gz`, a 204-row high-priority subset. These were generated
by `generate_gemini_synthetic.py`, which samples eligible sources from balanced strata, deduplicates by source
hash, verifies the rewritten facts and answer with the project's exact readers and engine, checkpoints progress,
and reads the Gemini key from the environment. The accompanying JSONL audit and state files record attempts,
rejections, focus, and request pacing; no API key is stored in the outputs.

This run accepted 1,015 of 2,069 attempted source rows (49.1%) across 275 tracked API requests, with an 8-second
minimum request interval. The pool is **not** balanced to the 25 answer buckets or to the three formats above:
it is a filtered paraphrase candidate, and rare/extreme relation labels remain in it. Do not combine it and then
describe the mixture as balanced. Inspect its relation-depth and label distribution, deduplicate against the final
training pool, and run development-only checks before any training use. The balanced 3,750-row graph-generated
candidate was produced separately and made no Gemini calls.

## Model baseline and next investigation

The selected checkpoint remains v10-r8p, step 6,000, SHA-256
`38c289d351d6517b5cde1bc9fb64ea97bd33eee193639d6351edda52fdfd7a52`; neither candidate above has been used to
train it. The official locked exam result remains 68.8% fair / 72.2% verified, and the generated 99.57% result is
not a project-blind real-world estimate. A fixed 15-story smoke sample scored 5/15 at the normal 96-token output
cap and 12/15 when only the diagnostic cap was raised to 192; this is a hypothesis-generating result, not an
accuracy estimate or proof that changing the cap fixes the model.

For the next model-improvement brainstorm, first separate reading failures from answer/decoder truncation on a
stratified long-story development set: record prefix length, generated-token count, stop reason, exact fact-read
match, and final answer at both caps. Then compare one controlled training change at a time against the same
untouched development suite. Preserve the checkpoint and locked test; do not train or tune from these notes alone.

## 2026-10-28 — format-balanced supplement and detour questions

`build_balanced.py` writes a supplement with every question format at the same count (formats are read from the text
by `eval.balance_report.fmt`; mix_r8 has 1.5–2.4% data sufficiency, nested clauses and possessive chains against 28%
plain statements). Inside each format no answer exceeds 1/12, no fact-count bucket 1/3 and no source kind 1/2 (relaxed
x3 only where a format cannot reach them). Every row is exact generator output that the engine reads back to its
answer, with at most 32 people and at most 190 output tokens, so it also exercises `--max_new 192`. New source:
`kinlogic.generate.q_obfuscated` (LogiQAte-style detours, arXiv 2602.01132), where every detour step names exactly one
person. Train on it only as a mixed-in share (it is not a replacement for mix_r8), and judge it on dev, the consistency
suite (`eval.consistency`) and long-story rows, not on generated sets alone.
