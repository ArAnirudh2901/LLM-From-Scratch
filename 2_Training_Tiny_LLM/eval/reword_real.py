"""Reworded copies of real exam questions (evaluation only, never training text): a frontier model rewrites each
question in new words (people, relations, genders, code definitions and what is asked kept; the options are not shown
to it and stay unchanged), and a copy is kept only when the same model, answering the original and the copy as separate
questions, picks the keyed option for both. The pipeline is then scored on the kept originals and their copies side by
side: how much of a score survives wording it was never tuned on.

    GEMINI_API_KEY=... python -m eval.reword_real --split dev            # build data/real_questions/reworded_dev.jsonl
    python -m eval.reword_real --split dev --score                       # exact readers only (no model)
    python -m eval.run_real --ckpt ... --split dev --reworded data/real_questions/reworded_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

REWRITE = """Rewrite each numbered blood-relation question below the way a different exam paper might word it. Change the
sentence structure and wording freely, but keep its meaning exactly:
- keep every person's name or letter exactly as written, and every family relationship and stated gender (pronouns,
  "a man", "his wife"); never add or drop a fact,
- keep code definitions exact ("A $ B means A is the father of B" may be reworded, but its symbol, letters and relation
  stay the same) and copy every expression like "P $ Q # R" character for character,
- keep the question asking for the same thing (the same two people, or "which of the following is true" and so on),
  since its answer options will be shown unchanged after your text,
- plain text only; no options, no answer.
Return JSON only: {{"rewrites": ["...", "...", ...]}} with exactly {n} items in order.

{questions}"""

SOLVE = """Answer each blood-relation multiple-choice question. Think carefully, then give only the option letter.
Return JSON only: {{"answers": ["a", "c", ...]}} with exactly {n} letters in order.

{questions}"""

MODELS = os.environ.get('REWORD_MODELS', 'gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-3-flash-preview')


def _json(raw: str, key: str) -> list:
    return json.loads(re.search(r'\{.*\}', raw, re.S).group(0))[key]


def build(rows, out: str, batch: int = 6, pause: float = 6.0):
    from eval.run_real import as_text
    from kinlogic.reword import call_rotating
    models = MODELS.split(',')
    done = {json.loads(l)['reworded_from'] for l in open(out)} if os.path.exists(out) else set()
    todo = [r for r in rows if r['question'] not in done]
    kept = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            rw = _json(call_rotating(REWRITE.format(n=len(chunk), questions='\n\n'.join(f'Question {j + 1}: {r["question"]}' for j, r in enumerate(chunk))),
                                     models, temperature=0.9, json_out=True), 'rewrites')
            if len(rw) != len(chunk):
                continue
            copies = [{**r, 'question': str(t).strip(), 'reworded_from': r['question']} for r, t in zip(chunk, rw)]
            solve = lambda rs: _json(call_rotating(SOLVE.format(n=len(rs), questions='\n\n'.join(f'Question {j + 1}: {as_text(r)}' for j, r in enumerate(rs))),
                                                   models, temperature=0.0, json_out=True), 'answers')
            g0, g1 = solve(chunk), solve(copies)   # separate calls: the copy is judged without its original in view
        except Exception as e:
            print('call failed', type(e).__name__, str(e)[:80], flush=True)
            time.sleep(60)
            continue
        if len(g0) != len(chunk) or len(g1) != len(chunk):
            continue
        with open(out, 'a') as f:
            for k, c in enumerate(copies):
                ok = str(g0[k]).strip().lower()[:1] == c['answer'] == str(g1[k]).strip().lower()[:1]
                kept += ok
                f.write(json.dumps({**c, 'kept': ok}, ensure_ascii=False) + '\n')
        print(f'{min(i + batch, len(todo))}/{len(todo)} | kept {kept}', flush=True)
        time.sleep(pause)


def pairs_of(path: str, rows) -> tuple[list, list]:
    """(originals, copies) for the kept copies whose original is among rows."""
    by = {r['question']: r for r in rows}
    cs = [c for c in map(json.loads, open(path)) if c.get('kept') and c['reworded_from'] in by]
    return [by[c['reworded_from']] for c in cs], cs


def compare(ok_orig, ok_copy) -> dict:
    n = len(ok_orig)
    a, b = sum(ok_orig), sum(ok_copy)
    return {'n': n, 'original': f'{a / max(1, n):.1%}', 'reworded': f'{b / max(1, n):.1%}',
            'right, then wrong after rewording': sum(x and not y for x, y in zip(ok_orig, ok_copy)),
            'wrong, then right after rewording': sum(y and not x for x, y in zip(ok_orig, ok_copy))}


def main():
    from eval.real_clean import broken
    from eval.run_real import load_real, run, split_of
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev')
    ap.add_argument('--out', default=None)
    ap.add_argument('--score', action='store_true', help='score the exact readers (no model) on kept pairs')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    out = a.out or f'data/real_questions/reworded_{a.split}.jsonl'
    rows = [r for r in load_real() if split_of(r) == a.split and not broken(r) and r.get('options') and r.get('answer')][:a.limit]
    if not a.score:
        build(rows, out)
        return
    orig, copies = pairs_of(out, rows)
    none = lambda b: [''] * len(b)
    s0, _ = run(none, orig)
    s1, _ = run(none, copies)
    print(json.dumps(compare(s0['ok'], s1['ok']), indent=1))


if __name__ == '__main__':
    main()
