"""Teacher rows written directly in our fact language (teach v2). teach.py's prompt let the teacher say only
"X is the R of Y", so puzzles, counts, negations, generations, professions, claims and data sufficiency never agreed
with their keys (about 20% kept). Here a frontier model reads the prepared question (people already @k placeholders:
the reader's exact input) with the language spec and generator examples of every feature, and writes the facts
itself. A row is kept only when the facts parse and our engine reaches the key: two independent signals agree.
Rows are the training format {kind, text, tokens, lf, truth}. Never run on evaluation questions (refused by wording).

    GEMINI_API_KEY=... python teach2.py --questions 'data/symgen_todo_*.jsonl.gz' --out data/teach2_k4.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
import re
import time

SPEC = """Convert each blood-relation exam question into our fact language. People are already placeholders @0, @1, ...;
write #1, #2, ... for people the text mentions without a name. One statement per ";", in the text's order.
  x REL y            y is x's REL ("@0 's mother is @1" -> @0 mother @1; "@1 is the mother of @0" -> @0 mother @1)
  x not REL y        the text denies it;  x only REL y: the only one ("only son")
  x paternal REL y | x maternal REL y   sided relations ("paternal uncle")
  x male | x female  gender the text gives (he/she, his/her, man, woman, Mr/Mrs); never from a name
  x no-sibling | no-brother | no-sister | no-child | unmarried | married
  x opposite y | x same y       genders differ / are the same
  x has N REL        exactly N such relatives ("has two sons" -> @0 has 2 son, then one fact per named son)
  members N | couples N | generations N | males N | females N | singles N   closed-family puzzles ("no single parent" -> singles 0)
  x gen N            x is in generation N (1 = oldest);  x prof P   profession (doctor, lawyer, ...)
  fathers N | mothers N | brothers N | sisters N   members who are that to another member
Questions: ask x y (how is y related to x?) | who x REL (who is x's REL?) | count male|female|people|couples |
  count x REL | gender x | pair male|female|couple|REL
Groups: part N (facts after it are statement N, data sufficiency) | claim N (facts after it are claim N, checked not assumed;
  only for claims written inside the question as i., ii., ...). For "which of the following is true?" with statement
  options, write only the facts (no ask); the options are checked separately.
Relation words: {words}
Examples:
{examples}
Return JSON only: {{"facts": ["...", ...]}} with exactly {n} strings in order.

{questions}"""

FEATURES = ['not', 'only', 'has', 'gen', 'singles', 'couples', 'generations', 'prof', 'count', 'who', 'gender', 'pair', 'part',
            'claim', 'no-sibling', 'opposite', 'paternal', 'maternal', 'male', 'female', 'members', 'brothers', 'unmarried', 'married']


def examples(pool='data/dataset_v6/kinds', per=1, seed=0):
    """Short generator questions covering every feature of the language, plus one per kind."""
    rng = random.Random(seed)
    rows = []
    for f in sorted(glob.glob(f'{pool}/*.jsonl.gz')):
        got = []
        for i, l in enumerate(gzip.open(f, 'rt')):
            r = json.loads(l)
            if len(r['tokens']) <= 90:
                got.append(r)
            if i > 3000:
                break
        rows += got
    rng.shuffle(rows)
    chosen, seen_kind = [], set()
    for feat in FEATURES:
        for r in rows:
            if feat in r['lf'].split() and r not in chosen:
                chosen.append(r)
                break
    for r in rows:
        k = r.get('kind', '')
        if k not in seen_kind and r not in chosen:
            seen_kind.add(k)
            chosen.append(r)
    return '\n'.join(f"Q: {' '.join(r['tokens'])}\nF: {r['lf']}" for r in chosen[:34])


def main():
    from kinlogic.exam import _answer_one
    from kinlogic.facts import RELS, parse
    from kinlogic.prepare import prepare
    from kinlogic.reword import call_rotating
    from kinlogic.symgen import as_text, real_keys
    from kinlogic.codes import _fits, read as read_code
    from kinlogic.rules import read as read_rules
    from kinlogic.speech import read as read_speech
    MODELS = os.environ.get('TEACH_MODELS', 'gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-3-flash-preview').split(',')
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--pause', type=float, default=5.0)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    import hashlib
    blocked = real_keys()
    qs = []
    for p_ in a.questions:
        for f in glob.glob(p_):
            try:
                for l in gzip.open(f, 'rt'):
                    qs.append(json.loads(l))
            except (EOFError, OSError):
                pass
    done = set()
    if os.path.exists(a.out):
        try:
            done = {json.loads(l)['text'] for l in gzip.open(a.out, 'rt')}
        except (EOFError, OSError):
            pass
    tried = set(open(a.out + '.tried').read().split()) if os.path.exists(a.out + '.tried') else set()
    h = lambda t: hashlib.sha1(t.encode()).hexdigest()
    todo = []
    for q in qs:
        t = as_text(q)
        if t in done or h(t) in tried:
            continue
        p = prepare(t)
        if hashlib.sha1(' '.join(p['tokens']).encode()).hexdigest() in blocked:
            continue   # never a question worded like an evaluation question
        try:
            if read_code(p['tokens'], p['options'], p['ents']) or _fits(read_rules(p['tokens']) or read_speech(p['tokens']), p['ents']):
                continue   # exact readers read it: star.py already distils those
        except Exception:
            pass
        todo.append((q, p))
    todo = todo[:a.limit]
    ex = examples()
    words = ', '.join(sorted(RELS)[:400])
    st = collections.Counter()
    print(f'{len(todo)} questions to teach', flush=True)
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        prompt = SPEC.format(words=words, examples=ex, n=len(chunk),
                             questions='\n\n'.join(f"Question {j + 1}: {' '.join(p['tokens'])}" for j, (_, p) in enumerate(chunk)))
        try:
            raw = call_rotating(prompt, MODELS, temperature=0.2, json_out=True)
            got = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))['facts']
        except Exception as e:
            st['call failed'] += 1
            print('call failed', type(e).__name__, str(e)[:80], flush=True)
            time.sleep(30)
            continue
        if len(got) != len(chunk):
            st['wrong count'] += 1
            continue
        out = []
        for (q, p), lf in zip(chunk, got):
            lf = re.sub(r'\s+', ' ', str(lf)).strip().strip(';').strip()
            stmts, bad = parse(lf)
            if bad or not stmts:
                st['unparsable'] += 1
                continue
            key = 'abcde'[int(q['answer'])]
            try:
                res = _answer_one(p, lf, lambda b: [''] * len(b), None)
            except Exception:
                st['engine error'] += 1
                continue
            if res.get('option') != key:
                st['no agreement'] += 1
                continue
            st['kept'] += 1
            out.append({'kind': 'teach2-' + q.get('style', 'q'), 'text': as_text(q), 'tokens': p['tokens'], 'lf': lf, 'truth': q['options'][int(q['answer'])]})
        with gzip.open(a.out, 'at') as f:   # one gzip block per batch: a stopped run keeps its rows
            f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
        with open(a.out + '.tried', 'a') as f:
            f.write(''.join(h(as_text(q)) + '\n' for q, _ in chunk))
        print(f'{min(i + a.batch, len(todo))}/{len(todo)} | {dict(st)}', flush=True)
        time.sleep(a.pause)
    print('done', dict(st))


if __name__ == '__main__':
    main()
