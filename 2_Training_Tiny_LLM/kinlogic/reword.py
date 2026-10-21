"""Frontier-model rewording with rejection: more natural, varied phrasings of verified questions, kept only when an
independent reading of the new text gives the same answer.

For each verified question (build_dataset.py output):
  1. a frontier model rewrites it in another exam style, keeping every name;
  2. a separate call, which never sees the original, extracts the facts of the rewrite as JSON;
  3. the engine solves those facts; the rewrite is kept only when every name survives and the answer is the
     constructed truth (gender-neutral path when the text leaves a gender open, as in kinlogic.deep.check).
The extracted facts, in the rewrite's own order, become the training target.

Keys come from the environment only (never pasted into a chat or a file): GEMINI_API_KEY or ANTHROPIC_API_KEY.
    python -m kinlogic.reword --src data/dataset/deep --out data/reworded.jsonl.gz --n 2000 [--provider gemini]
Resumable (skips questions already in --out); one request at a time with a pause (free-tier rate limits).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import re
import time
import urllib.request

from .deep import _neutral
from .engine import claim_holds, solve
from .facts import parse
from .kin import KIN
from .prepare import prepare

REWRITE = """Rewrite this blood-relation reasoning question as it might appear in a different competitive exam
(bank PO, SSC, CAT, UPSC CSAT). Change the wording and sentence structure freely, but:
- keep every person's name or letter exactly as written,
- keep every family relationship and every stated gender exactly (no new facts, none dropped),
- keep the final question asking about the same two people,
- do not add answer options or an answer.
Output only the rewritten question.

Question: {q}"""

EXTRACT = """Read this blood-relation question and list its facts as JSON, nothing else:
{{"facts": [["X", "relation", "Y"], ...], "genders": [["X", "male" or "female"], ...], "ask": ["A", "B"]}}
A fact ["X", "relation", "Y"] means "Y is X's relation". Use only these relation words: {words}.
"ask" ["A", "B"] means the question asks how B is related to A. Use the names exactly as written.

Question: {q}"""


def _post(url, body, headers, timeout=90):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json', **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call(prompt: str, provider: str) -> str:
    if provider == 'gemini':
        key = os.environ['GEMINI_API_KEY']
        model = os.environ.get('REWORD_GEMINI_MODEL', 'gemini-2.5-flash')
        r = _post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                  {'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': {'temperature': 0.9}},
                  {'x-goog-api-key': key})
        return r['candidates'][0]['content']['parts'][0]['text'].strip()
    key = os.environ['ANTHROPIC_API_KEY']
    model = os.environ.get('REWORD_CLAUDE_MODEL', 'claude-haiku-5-5')
    r = _post('https://api.anthropic.com/v1/messages',
              {'model': model, 'max_tokens': 800, 'messages': [{'role': 'user', 'content': prompt}]},
              {'x-api-key': key, 'anthropic-version': '2023-06-01'})
    return r['content'][0]['text'].strip()


def to_facts(extracted: dict, ents: list[dict]) -> str | None:
    """The extraction as fact language, names placed the way prepare() numbered them; None if anything is off."""
    by = {e['text'].lower(): e['id'] for e in ents}
    place = lambda n: by.get(str(n).strip().lower())
    out = []
    for x, r, y in extracted.get('facts', []):
        r = str(r).strip().lower().replace(' ', '-') if str(r).strip().lower() not in KIN else str(r).strip().lower()
        if r not in KIN or place(x) is None or place(y) is None:
            return None
        out.append(f'{place(x)} {r} {place(y)}')
    for x, g in extracted.get('genders', []):
        if place(x) is None or g not in ('male', 'female'):
            return None
        out.append(f'{place(x)} {g}')
    a, b = extracted.get('ask', [None, None])
    if place(a) is None or place(b) is None:
        return None
    out.append(f'ask {place(a)} {place(b)}')
    return ' ; '.join(out)


def accept(lf: str, truth: str, tokens) -> bool:
    st, bad = parse(lf)
    if bad:
        return False
    r = solve(st, people=sorted({t for t in tokens if t[0] == '@'}), max_steps=20000)
    if not r['ok'] or not r['questions'] or not r['questions'][-1]['answers']:
        return False
    q = r['questions'][-1]
    got = {a.get('key') for a in q['answers']} | {a.get('base') for a in q['answers']} | {q.get('neutral')}
    return truth in got or _neutral(truth) in {_neutral(k) for k in got if k}


def entails(lf_a: str, lf_b: str, people) -> bool:
    """Every family fact of b (relations and genders) holds in every family a allows."""
    st_a, _ = parse(lf_a)
    r = solve(st_a, people=people, max_steps=20000)
    if not r['ok'] or not r['worlds']:
        return False
    facts = [f for f in parse(lf_b)[0] if f['t'] in ('rel', 'gender') and not f.get('part') and not f.get('claim')]
    return all(claim_holds(w, [f], r['ids']) is True for w in r['worlds'] for f in facts)


def same_meaning(original: dict, lf: str, ents) -> bool:
    """The extraction and the original facts entail each other (names matched across the two texts)."""
    by = {e['text'].lower(): e['id'] for e in ents}
    move = {pid: by.get(name.lower()) for pid, name in (original.get('names') or {}).items()}
    if not move or None in move.values():
        return False
    orig = ' '.join(move.get(t, t) for t in original['lf'].split())
    people = sorted(set(move.values()))
    return entails(orig, lf, people) and entails(lf, orig, people)


def reword_one(ex: dict, provider: str) -> dict | None:
    names = [e['text'] for e in prepare(ex['text'])['ents'] if e['kind'] in ('name', 'letter')]
    new = call(REWRITE.format(q=ex['text']), provider)
    if not new or any(n not in new for n in names):
        return None
    p = prepare(new)
    raw = call(EXTRACT.format(q=new, words=', '.join(sorted(KIN))), provider)
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None
    try:
        lf = to_facts(json.loads(m.group(0)), p['ents'])
    except (ValueError, TypeError):
        return None
    if lf is None or not accept(lf, ex['truth'], p['tokens']) or not same_meaning(ex, lf, p['ents']):
        return None
    return {'kind': 'reworded-' + ex['kind'], 'text': new, 'tokens': p['tokens'], 'lf': lf, 'truth': ex['truth'],
            'depth': ex.get('depth'), 'source': hashlib.sha1(ex['text'].encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/dataset/deep')
    ap.add_argument('--out', default='data/reworded.jsonl.gz')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--provider', default='gemini' if os.environ.get('GEMINI_API_KEY') else 'anthropic')
    ap.add_argument('--pause', type=float, default=4.0, help='seconds between questions (free-tier rate limits)')
    a = ap.parse_args()
    rows = [json.loads(l) for f in sorted(glob.glob(f'{a.src}/**/*.jsonl.gz', recursive=True)) for l in gzip.open(f, 'rt')]
    rows = [r for r in rows if isinstance(r.get('truth'), str) and "'s " not in r['truth'] and r.get('names')]
    random.Random(0).shuffle(rows)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['source'] for l in gzip.open(a.out, 'rt')}
    kept = tried = 0
    with gzip.open(a.out, 'at') as f:
        for ex in rows:
            if kept + len(done) >= a.n:
                break
            if hashlib.sha1(ex['text'].encode()).hexdigest() in done:
                continue
            tried += 1
            try:
                got = reword_one(ex, a.provider)
            except Exception as e:   # rate limit or network: wait and go on
                print('call failed:', type(e).__name__, str(e)[:120], flush=True)
                time.sleep(30)
                continue
            if got:
                f.write(json.dumps(got, ensure_ascii=False) + '\n')
                f.flush()
                kept += 1
            if tried % 20 == 0:
                print(f'{tried} tried, {kept} kept ({kept / tried:.0%})', flush=True)
            time.sleep(a.pause)
    print(f'done: {kept} new rewrites kept of {tried} tried; {kept + len(done)} in {a.out}')


if __name__ == '__main__':
    main()
