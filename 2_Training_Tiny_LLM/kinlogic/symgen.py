"""New answer-keyed exam questions written by an LLM (SymGen, Phi-style synthetic data): one call writes a batch of
questions in a given exam style with options and an answer key, a second, independent call solves them; a question is
kept only when both give the same option (agreement verifier). Questions whose wording (names masked) matches a
scraped real question are dropped, so the real test stays unseen. The kept questions have answers but no facts: STaR
/ rejection sampling (our reader samples readings, the engine keeps those reaching the key) turns them into training
rows. Free tier, paced; resumable (one gzip block per batch).

    GEMINI_API_KEY=... python -m kinlogic.symgen --out data/symgen/part-k1.jsonl.gz --n 4000
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import time

from .reword import call, call_rotating

WRITERS = os.environ.get('SYMGEN_WRITERS', 'gemini-3.5-flash-lite,gemini-3.1-flash-lite-preview,gemini-flash-latest,gemini-3.6-flash').split(',')
SOLVERS = os.environ.get('SYMGEN_SOLVERS', 'gemini-flash-lite-latest,gemini-3.7-flash,gemini-3.8-flash').split(',')

STYLES = {
    'puzzle': "a closed-family puzzle: 'There are N members P, Q, R, ... in a family of two or three generations' with "
              "facts such as 'there are two married couples', 'no single parent', 'X is unmarried', followed by 5-9 short "
              "clues (often with 'who is' clauses) and one question about two members",
    'count': "a closed-family puzzle (as above) whose question asks how many male or female members there are, or how many "
             "children someone has",
    'speech': "a speech question: 'Pointing to a photograph / a man / a lady, X said, \"...\"' or 'Introducing Y, X said ...' "
              "with a nested description such as 'his mother's only daughter's husband'",
    'coded': "a coded relation question: 'A + B means A is the father of B; A - B means ...' (3-5 symbols, sometimes with "
             "quotes, sometimes 'B is A's ...'), then an expression like 'P + Q - R x S' and a question about two letters",
    'statements': "a chain of 3-7 relation statements about named people (mixed Indian and international names, sometimes "
                  "letters), some phrased 'X's mother is Y', 'Y, whose husband is Z', 'X and Y are siblings', then 'How is A "
                  "related to B?'",
    'claims': "statements about a family followed by 'Which of the following is definitely true?' with options that are "
              "statements ('M is the aunt of R')",
    'sufficiency': "a data-sufficiency question: 'How is A related to B? Statement I: ... Statement II: ...' with the standard "
                   "five options (I alone, II alone, either, both, neither)",
    'story': "a short everyday story (3-6 sentences: visits, gifts, trips, friends) that mentions relations in passing, then "
             "a question about two people in it",
    'gender': "a question where the answer depends on a gender given only by a pronoun or a word like 'Mrs.' or 'a lady', with "
              "'Cannot be determined' among the options when the gender is open",
}

GRID = {   # coverage is planned, not left to chance (Kimi K3): each batch draws one value per axis
    'relations': ['in-laws (son-in-law, sister-in-law, co-brother)', 'grand- and great-grand relations', 'uncle/aunt/nephew/niece on a stated side',
                  'only son / only daughter / only child', 'step and half relations avoided; plain blood relations', 'spouse chains (husband\'s sister\'s son)',
                  'cousins', 'father-in-law / mother-in-law chains'],
    'wording': ['possessive chains ("X\'s mother\'s brother")', '"the X of the Y of Z" chains', 'pronouns (he, she, his, her) that must be resolved',
                'who-clauses ("Q, whose sister is R, ...")', 'plural and list forms ("A, B and C are sisters")', 'negations ("X is not the father of Y")',
                'descriptions instead of names ("a lady", "the boy in the photograph")', 'counts ("has two sons", "has no brother")'],
    'trap': ['a gender left open by the text (answer may be "cannot be determined")', 'an extra person who is irrelevant', 'two people with similar roles',
             'an answer that depends on one small word ("only", "his" vs "her")', 'none'],
}
WEIGHTS = {'puzzle': 3.0, 'count': 1.5, 'speech': 2.5, 'statements': 2.5, 'claims': 1.0, 'coded': 0.8, 'sufficiency': 1.0, 'story': 1.0,
           'gender': 1.5}   # roughly the error rate of each kind on the real dev half


def features(rng):
    return '; '.join(f'{k}: {rng.choice(v)}' for k, v in GRID.items())


WRITE = """You write practice questions for Indian competitive exams (SSC, IBPS, RRB, CAT) on blood relations.
Write {n} NEW questions, all different from each other, in this style: {style}.
Across the batch, use these features: {features}.
Each question must follow logically from its text, have 4 or 5 options and exactly one correct option.
Use varied names ({names}); use plain text (no markdown, no LaTeX).
Return only JSON: a list of objects {{"question": "...", "options": ["...", "..."], "answer": <index of the correct option, 0-based>}}."""

SOLVE = """Solve each blood-relation question carefully, step by step in your head, then give only the index (0-based) of the
correct option. Return only JSON: a list of integers, one per question, in order.

{qs}"""

NAMES = ['Indian names', 'Indian and international names', 'single capital letters as names', 'South Indian names',
         'North Indian names', 'Muslim and Christian names', 'international names']


def _key(text):
    from .prepare import prepare
    return hashlib.sha1(' '.join(prepare(text)['tokens']).encode()).hexdigest()


def real_keys():
    """Masked wordings of the scraped real questions (evaluation only; kept out of anything we train on)."""
    try:
        from eval.run_real import as_text, load_real
        return {_key(as_text(r)) for r in load_real()}
    except Exception:
        return set()


def as_text(q):
    return q['question'].strip() + ' ' + ' '.join(f'({"abcde"[i]}) {o}' for i, o in enumerate(q['options'][:5]))


def _json(text):
    m = re.search(r'\[.*\]', text, re.S)
    return json.loads(m.group(0)) if m else None


def batch(style, n, rng):
    raw = call_rotating(WRITE.format(n=n, style=STYLES[style], names=rng.choice(NAMES), features=features(rng)), WRITERS, temperature=1.0,
                        json_out=True)
    qs = [q for q in (_json(raw) or []) if isinstance(q, dict) and isinstance(q.get('question'), str)
          and isinstance(q.get('options'), list) and 4 <= len(q['options']) <= 5 and isinstance(q.get('answer'), int)
          and 0 <= q['answer'] < len(q['options'])]
    if not qs:
        return [], 0
    solved = _json(call_rotating(SOLVE.format(qs='\n\n'.join(f'Q{i + 1}. {as_text(q)}' for i, q in enumerate(qs))), SOLVERS,
                                 temperature=0.0, json_out=True)) or []
    keep = [dict(q, style=style) for q, s in zip(qs, solved) if s == q['answer']]
    return keep, len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4000, help='kept questions wanted in this file (counting earlier runs)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--pause', type=float, default=8.0, help='seconds between calls (free-tier pacing)')
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    seen, have = set(), 0
    if os.path.exists(a.out):
        try:
            for line in gzip.open(a.out, 'rt'):
                seen.add(_key(as_text(json.loads(line))))
                have += 1
        except (EOFError, OSError):
            pass
    blocked = real_keys()
    rng = random.Random(a.seed if a.seed is not None else int(time.time()))
    styles = list(STYLES)
    tried = kept = waits = 0
    while have < a.n:
        style = rng.choices(styles, weights=[WEIGHTS.get(x, 1.0) for x in styles])[0]
        try:
            got, n = batch(style, a.batch, rng)
            waits = 0
        except Exception as e:   # 429 / 503: back off exponentially, then continue
            waits += 1
            wait = min(900, 30 * 2 ** min(waits, 5))
            print(f'call failed: {type(e).__name__} {str(e)[:80]} - waiting {wait}s', flush=True)
            time.sleep(wait)
            continue
        tried += n
        new = []
        for q in got:
            k = _key(as_text(q))
            if k in seen or k in blocked:
                continue
            seen.add(k)
            new.append(q)
        if new:
            with gzip.open(a.out, 'at') as f:   # one complete gzip block per batch
                f.write(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in new))
        have += len(new)
        kept += len(new)
        print(f'{style}: {len(new)}/{n} agreed | {have} in file | agreement {kept / max(1, tried):.0%}', flush=True)
        time.sleep(a.pause * 2)   # two calls per batch


if __name__ == '__main__':
    main()
