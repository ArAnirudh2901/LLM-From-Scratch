"""How far our engine and conventions can go with a near-perfect reader (measurement only, never training data).

A frontier model reads each real question (the placeholder text our reader sees) into the fact language, with the
spec and worked examples; the exact readers (codes, rules, speech) still read what they cover, as in the pipeline;
the engine answers. Separately the frontier model answers each question directly, which estimates how often the
answer keys themselves are wrong. Run on the dev half only.

    GEMINI_API_KEY=... python -m eval.ceiling --split dev --out runs/ceiling_dev.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re

SPEC = __import__('kinlogic.facts', fromlist=['x']).__doc__

READ = """You translate family-relation exam questions into a fact language. People are already replaced by @0, @1, ...
Specification:
{spec}
Relation words allowed: {words}.
Rules: write every fact the text states, in text order; use #1, #2 for unnamed people; add "x male"/"x female" only
when the text shows the gender (pronoun, man/woman, husband/wife words imply it already); end with the question
(ask/who/count/...). For "which statement is true" questions write each option as "claim N" followed by its facts.
Examples:
{examples}
Translate each numbered question. Return JSON only: {{"facts": ["...", "...", ...]}} with exactly {n} strings in order.

{questions}"""

SOLVE = """Answer each blood-relation multiple-choice question. Think carefully, then give only the option letter.
Return JSON only: {{"answers": ["a", "c", ...]}} with exactly {n} letters in order.

{questions}"""


def examples(k_per_kind: int = 2, seed: int = 0) -> str:
    from kinlogic import generate as gen
    rng, out = random.Random(seed), []
    for kind in ('statements', 'nested', 'speech', 'introduce', 'identity', 'puzzle', 'claims', 'sufficiency'):
        got = 0
        while got < k_per_kind:
            ex = gen.KINDS[kind](rng)
            if ex and len(ex['tokens']) < 90:
                out.append(f"Question: {' '.join(ex['tokens'])}\nFacts: {ex['lf']}")
                got += 1
    return '\n\n'.join(out)


def main():
    from eval.real_clean import broken, category
    from eval.run_real import as_text, correct, load_real, split_of
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.exam import answer
    from kinlogic.kin import KIN
    from kinlogic.prepare import prepare
    from kinlogic.reword import call_rotating
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--shard', default='0/1', help='i/n: this process takes every question whose index % n == i')
    ap.add_argument('--models', default=os.environ.get('CEILING_MODELS', 'gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-3-flash-preview'))
    a = ap.parse_args()
    models = a.models.split(',')
    si, sn = map(int, a.shard.split('/'))
    rows = [r for j, r in enumerate(x for x in load_real() if split_of(x) == a.split and not broken(x)) if j % sn == si]
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    ex = examples()
    words = ', '.join(sorted(KIN))
    todo = [r for r in rows if r['question'] not in done]
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        preps = [prepare(as_text(r)) for r in chunk]
        qs = '\n\n'.join(f"Question {j + 1}: {' '.join(p['tokens'])}" for j, p in enumerate(preps))
        full = '\n\n'.join(f'Question {j + 1}: {as_text(r)}' for j, r in enumerate(chunk))
        try:
            lfs = json.loads(re.search(r'\{.*\}', call_rotating(READ.format(spec=SPEC, words=words, examples=ex, n=len(chunk), questions=qs),
                                                                 models, temperature=0.0, json_out=True), re.S).group(0))['facts']
            direct = json.loads(re.search(r'\{.*\}', call_rotating(SOLVE.format(n=len(chunk), questions=full), models, temperature=0.0,
                                                                   json_out=True), re.S).group(0))['answers']
        except Exception as e:
            print('call failed', type(e).__name__, str(e)[:80], flush=True)
            continue
        if len(lfs) != len(chunk) or len(direct) != len(chunk):
            continue
        for r, p, lf, d in zip(chunk, preps, lfs, direct):
            exact = bool(read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens'])))
            res = answer(as_text(r), lambda b, lf=str(lf), toks=p['tokens']: [[(lf if t is toks or t == toks else '', 0.0)] for t in b])
            done[r['question']] = {'cat': category(r), 'exact': exact, 'pipeline_ok': bool(correct(r, res)),
                                   'direct': str(d).strip().lower()[:1], 'key': str(r.get('answer', '')).strip().lower()[:1], 'facts': str(lf)}
        json.dump(done, open(a.out, 'w'))
        print(f'{min(i + a.batch, len(todo))}/{len(todo)}', flush=True)
    by = collections.defaultdict(collections.Counter)
    for v in done.values():
        c = by[v['cat']]
        c['n'] += 1
        c['frontier reader + our engine'] += v['pipeline_ok']
        c['frontier direct = key'] += v['direct'] == v['key']
        c['both disagree with key'] += (not v['pipeline_ok']) and v['direct'] != v['key']
    tot = sum(by.values(), collections.Counter())
    for k, c in [x for x in sorted(by.items()) + [('ALL', tot)] if x[1]['n']]:
        print(f"{k:11s} n={c['n']:3d} | reader+engine {c['frontier reader + our engine'] / c['n']:.1%} | direct=key {c['frontier direct = key'] / c['n']:.1%} | both disagree with key {c['both disagree with key'] / c['n']:.1%}")


if __name__ == '__main__':
    main()
