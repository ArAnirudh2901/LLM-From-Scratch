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
- keep every family relationship and every stated gender exactly (no new facts, none dropped; keep sentences like
  "X is a woman" or pronouns that show a gender),
- keep the final question asking about the same two people,
- do not add answer options or an answer,
- plain text only: no LaTeX, markdown or bullet points; copy code symbols (like $, #, @, ×) exactly as written.
Output only the rewritten question.

Question: {q}"""

EXTRACT = """List the family facts of this question as JSON, nothing else:
{{"facts": [{{"person": "X", "is_the": "relation", "of": "Y"}}, ...], "genders": [{{"person": "X", "gender": "male" or "female"}}, ...],
 "ask": {{"how_is": "B", "related_to": "A"}}}}
Write each fact exactly as the text states it: "X is the son of Y" -> {{"person": "X", "is_the": "son", "of": "Y"}};
"Y's daughter is X" -> {{"person": "X", "is_the": "daughter", "of": "Y"}}. Keep the exact relation word the text uses
(son, not child). List a gender only when the text states it (he/she, his/her, a man, a woman). Relation words allowed:
{words}. Use the names exactly as written.

Question: {q}"""


def _post(url, body, headers, timeout=90):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json', **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call(prompt: str, provider: str, temperature: float = 0.9, json_out: bool = False, model: str | None = None) -> str:
    if provider == 'gemini':
        key = os.environ['GEMINI_API_KEY']
        model = model or os.environ.get('REWORD_GEMINI_MODEL', 'gemini-3.5-flash-lite')   # ~1 s a call; 3.x flash thinks for ~30 s
        cfg = {'temperature': temperature, 'thinkingConfig': {'thinkingLevel': 'minimal'},
               **({'responseMimeType': 'application/json'} if json_out else {})}
        r = _post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                  {'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': cfg},
                  {'x-goog-api-key': key})
        return r['candidates'][0]['content']['parts'][0]['text'].strip()
    key = os.environ['ANTHROPIC_API_KEY']
    model = os.environ.get('REWORD_CLAUDE_MODEL', 'claude-haiku-5-5')
    r = _post('https://api.anthropic.com/v1/messages',
              {'model': model, 'max_tokens': 800, 'temperature': temperature, 'messages': [{'role': 'user', 'content': prompt}]},
              {'x-api-key': key, 'anthropic-version': '2023-06-01'})
    return r['content'][0]['text'].strip()


def clean(text: str) -> str:
    """The rewritten question without a preamble ("Rewritten question:") or wrapping quotes."""
    t = text.strip()
    t = re.sub(r'^(?:here is|here\'s)?[^\n:]{0,40}(?:question|version)[^\n:]{0,20}:\s*', '', t, flags=re.I)
    return t.strip().strip('"“”').strip()


ARTICLES = re.compile(r'^(?:the|a|an)\s+')


def _head(text: str) -> str:
    """'the man in the photograph' -> 'man'; 'a lady' -> 'lady'."""
    return (ARTICLES.sub('', str(text).strip().lower()).split() or [''])[0]


def placer(ents):
    """name -> placeholder: exact text, else a description's head noun when it picks out one entity."""
    by = {e['text'].lower(): e['id'] for e in ents}
    heads = {}
    for e in ents:
        if e['kind'] == 'desc':
            heads.setdefault(_head(e['text']), []).append(e['id'])
    return lambda n: by.get(str(n).strip().lower()) or (heads.get(_head(n), [None, None]) + [None])[0] \
        if len(heads.get(_head(n), [])) <= 1 else None


def needed(ex) -> list[str]:
    """Names and letters of the people in the original's facts (filler sentences may be dropped by a rewrite)."""
    ids = set(re.findall(r'@\d+', ex['lf']))
    return [e['text'] for e in prepare(ex['text'])['ents'] if e['id'] in ids and e['kind'] in ('name', 'letter')]


def to_facts(extracted: dict, ents: list[dict]) -> str | None:
    """The extraction as fact language, names placed the way prepare() numbered them; None if anything is off."""
    place = placer(ents)
    out = []
    for f in extracted.get('facts', []):   # "person is the R of Y" -> fact "Y R person"
        x, y, r = f.get('person'), f.get('of'), str(f.get('is_the', '')).strip().lower()
        r = r if r in KIN else r.replace(' ', '-')
        if r not in KIN or place(x) is None or place(y) is None:
            return None
        out.append(f'{place(y)} {r} {place(x)}')
    for g in extracted.get('genders', []):
        if place(g.get('person')) is None or g.get('gender') not in ('male', 'female'):
            return None
        out.append(f"{place(g['person'])} {g['gender']}")
    ask = extracted.get('ask') or {}
    b, a = place(ask.get('how_is')), place(ask.get('related_to'))
    if a is None or b is None:
        return None
    out.append(f'ask {a} {b}')
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
    place = placer(ents)
    ids = set(re.findall(r'@\d+', original['lf']))
    names = original.get('names') or {}
    move = {pid: place(names.get(pid, '')) for pid in ids}
    new_desc = [e['id'] for e in ents if e['kind'] == 'desc']
    old_desc = [pid for pid in ids if move[pid] is None and pid in names and not names[pid][:1].isupper()]
    if len(old_desc) == 1 and len(new_desc) == 1:   # one description on each side ("a man" / "the gentleman")
        move[old_desc[0]] = new_desc[0]
    if not move or None in move.values():
        return False
    orig = ' '.join(move.get(t, t) for t in original['lf'].split())
    people = sorted(set(move.values()))
    return entails(orig, lf, people) and entails(lf, orig, people)


def reword_one(ex: dict, provider: str) -> dict | None:
    names = needed(ex)
    new = clean(call(REWRITE.format(q=ex['text']), provider))
    if not new or any(n.lower() not in new.lower() for n in names) or re.search(r'\\[a-z]+|\$[A-Za-z]|\*\*|^\s*[-*] ', new, re.M):
        return None
    p = prepare(new)
    raw = call(EXTRACT.format(q=new, words=', '.join(sorted(KIN))), provider, temperature=0.0, json_out=True,   # a faithful reading
               model=os.environ.get('REWORD_EXTRACT_MODEL', 'gemini-flash-lite-latest') if provider == 'gemini' else None)   # its own free quota
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
    ap.add_argument('--src', default='data/dataset_v2')
    ap.add_argument('--out', default='data/reworded.jsonl.gz')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--provider', default='gemini' if os.environ.get('GEMINI_API_KEY') else 'anthropic')
    ap.add_argument('--pause', type=float, default=4.0, help='seconds between questions (one call per model each; free-tier rate limits)')
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(f'{a.src}/**/*.jsonl.gz', recursive=True)) if '/test/' not in f]   # never the locked test
    rng = random.Random(0)
    per = max(5, 4 * a.n // max(1, len(files)))   # a sample from every file (all kinds and depths), parsed lazily: the set is 1.36M
    rows = []
    for f in files:
        lines = gzip.open(f, 'rt').readlines()
        rows += [json.loads(l) for l in rng.sample(lines, min(per, len(lines)))]
    rows = [r for r in rows if isinstance(r.get('truth'), str) and "'s " not in r['truth'] and r.get('names')]
    rng.shuffle(rows)
    done = set()
    if os.path.exists(a.out):   # a killed run can leave a truncated last block: keep what reads
        try:
            for l in gzip.open(a.out, 'rt'):
                done.add(json.loads(l)['source'])
        except (EOFError, OSError, ValueError):
            pass
    kept = tried = 0
    if True:   # each kept row is appended as its own complete gzip block: a killed run never damages earlier rows
        for ex in rows:
            if kept + len(done) >= a.n:
                break
            if hashlib.sha1(ex['text'].encode()).hexdigest() in done:
                continue
            tried += 1
            try:
                got = reword_one(ex, a.provider)
            except Exception as e:   # rate limit or network: back off and go on
                waits = getattr(main, 'waits', 0) + 1
                main.waits = waits
                print('call failed:', type(e).__name__, str(e)[:120], flush=True)
                time.sleep(min(300, 15 * waits))
                continue
            main.waits = 0
            if got:
                with gzip.open(a.out, 'at') as f:
                    f.write(json.dumps(got, ensure_ascii=False) + '\n')
                kept += 1
            if tried % 20 == 0:
                print(f'{tried} tried, {kept} kept ({kept / tried:.0%})', flush=True)
            time.sleep(a.pause)
    print(f'done: {kept} new rewrites kept of {tried} tried; {kept + len(done)} in {a.out}')


if __name__ == '__main__':
    main()
