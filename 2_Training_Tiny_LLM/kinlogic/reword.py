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
- keep every person's name or letter exactly as written: each one must appear in your version,
- keep every relation word exactly (never replace 'son' with 'child' or 'mother' with 'parent'),
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
(son, not child). List a gender whenever the text shows it: he/she, his/her, a man, a woman.
Example: "Vani and her son Irfan went to the temple. How is Irfan related to Vani?" ->
{{"facts": [{{"person": "Irfan", "is_the": "son", "of": "Vani"}}], "genders": [{{"person": "Vani", "gender": "female"}}],
 "ask": {{"how_is": "Irfan", "related_to": "Vani"}}}}
Example: "Pointing to Ravi, Asha said, 'His father is my brother.' How is Ravi related to Asha?" ->
{{"facts": [{{"person": "#1", "is_the": "father", "of": "Ravi"}}, {{"person": "#1", "is_the": "brother", "of": "Asha"}}],
 "genders": [{{"person": "Ravi", "gender": "male"}}], "ask": {{"how_is": "Ravi", "related_to": "Asha"}}}}
Use "#1", "#2" for people the text mentions without a name. Relation words allowed: {words}. Use names exactly as written.

Question: {q}"""


SKIP = {'coded', 'deep-coded'}   # read exactly by kinlogic/codes.py at answer time; rewriters also 'decode' them wrongly


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
    place0 = placer(ents)
    place = lambda n: str(n).strip() if re.fullmatch(r'#\d+', str(n).strip()) else place0(n)   # unnamed people
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


REWRITE_BATCH = """Rewrite each numbered blood-relation question below as it might appear in a different competitive exam
(bank PO, SSC, CAT, UPSC CSAT). For each one, change the wording and sentence structure freely, but:
- keep every person's name or letter exactly as written: each one must appear in your version,
- keep every relation word exactly (never replace 'son' with 'child' or 'mother' with 'parent'),
- keep every family relationship and every stated gender (sentences like "X is a woman", pronouns),
- keep the final question asking about the same two people; no answer options, no answer,
- plain text only: no LaTeX, markdown or bullet points.
Return JSON only: {{"rewrites": ["rewrite of question 1", "rewrite of question 2", ...]}} with exactly {n} items in order.

{questions}"""

EXTRACT_BATCH = """For each numbered blood-relation question below, list its family facts. Return JSON only:
{{"readings": [{{"facts": [...], "genders": [...], "ask": {{...}}}}, ...]}} with exactly {n} items in order, each in the form
{{"facts": [{{"person": "X", "is_the": "relation", "of": "Y"}}, ...], "genders": [{{"person": "X", "gender": "male" or "female"}}, ...],
 "ask": {{"how_is": "B", "related_to": "A"}}}}
Write each fact as the text states it ("X is the son of Y" -> person X, is_the son, of Y; "Y's daughter is X" -> person X,
is_the daughter, of Y), keeping the exact relation word. List a gender whenever the text shows it (he/she, his/her, a man,
a woman): "Vani and her son Irfan" -> Vani female. Use "#1", "#2" for people the text mentions without a name; facts of one
question never mention people of another. Relation words allowed: {words}. Use names exactly as written.

{questions}"""


def _numbered(texts):
    return '\n\n'.join(f'Question {i + 1}: {t}' for i, t in enumerate(texts))


def reword_batch(exs: list[dict], provider: str) -> list[dict | None]:
    """Rewrites and readings for several questions in two requests; the same checks as reword_one for each."""
    raw = call(REWRITE_BATCH.format(n=len(exs), questions=_numbered([e['text'] for e in exs])), provider, temperature=0.7, json_out=True)
    try:
        news = [clean(str(t)) for t in json.loads(re.search(r'\{.*\}', raw, re.S).group(0))['rewrites']]
    except (AttributeError, ValueError, KeyError, TypeError):
        return [None] * len(exs)
    if len(news) != len(exs):
        return [None] * len(exs)
    keep = [i for i, (ex, new) in enumerate(zip(exs, news))
            if new and all(n.lower() in new.lower() for n in needed(ex)) and not re.search(r'\\[a-z]+|\$[A-Za-z]|\*\*|^\s*[-*] ', new, re.M)]
    out = [None] * len(exs)
    if not keep:
        return out
    raw = call(EXTRACT_BATCH.format(n=len(keep), words=', '.join(sorted(KIN)), questions=_numbered([news[i] for i in keep])),
               provider, temperature=0.0, json_out=True,
               model=os.environ.get('REWORD_EXTRACT_MODEL', 'gemini-flash-lite-latest') if provider == 'gemini' else None)
    try:
        readings = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))['readings']
    except (AttributeError, ValueError, KeyError, TypeError):
        return out
    if len(readings) != len(keep):
        return out
    for i, reading in zip(keep, readings):
        ex, new = exs[i], news[i]
        p = prepare(new)
        try:
            lf = to_facts(reading, p['ents'])
        except (ValueError, TypeError, AttributeError):
            continue
        if lf and accept(lf, ex['truth'], p['tokens']) and same_meaning(ex, lf, p['ents']):
            out[i] = {'kind': 'reworded-' + ex['kind'], 'text': new, 'tokens': p['tokens'], 'lf': lf, 'truth': ex['truth'],
                      'depth': ex.get('depth'), 'source': hashlib.sha1(ex['text'].encode()).hexdigest()}
    return out


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
    ap.add_argument('--pause', type=float, default=6.0, help='seconds between batches (two requests each, one per model)')
    ap.add_argument('--batch', type=int, default=10, help='questions per request (1 = one request per question)')
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(f'{a.src}/**/*.jsonl.gz', recursive=True)) if '/test/' not in f]   # never the locked test
    rng = random.Random(0)
    per = max(5, 4 * a.n // max(1, len(files)))   # a sample from every file (all kinds and depths), parsed lazily: the set is 1.36M
    rows = []
    for f in files:
        lines = gzip.open(f, 'rt').readlines()
        rows += [json.loads(l) for l in rng.sample(lines, min(per, len(lines)))]
    rows = [r for r in rows if isinstance(r.get('truth'), str) and "'s " not in r['truth'] and r.get('names')
            and r['kind'] not in SKIP]
    rng.shuffle(rows)
    done = set()
    if os.path.exists(a.out):   # a killed run can leave a truncated last block: keep what reads
        try:
            for l in gzip.open(a.out, 'rt'):
                done.add(json.loads(l)['source'])
        except (EOFError, OSError, ValueError):
            pass
    kept = tried = 0
    todo = [ex for ex in rows if hashlib.sha1(ex['text'].encode()).hexdigest() not in done]
    waits = 0
    i = 0
    while i < len(todo) and kept + len(done) < a.n:
        batch = todo[i:i + a.batch]
        try:
            got = reword_batch(batch, a.provider) if a.batch > 1 else [reword_one(batch[0], a.provider)]
        except Exception as e:   # rate limit or network: back off (longer each time), retry the same batch
            waits += 1
            print('call failed:', type(e).__name__, str(e)[:100], f'- waiting {min(600, 20 * 2 ** min(waits, 5))}s', flush=True)
            time.sleep(min(600, 20 * 2 ** min(waits, 5)))
            continue
        waits = 0
        i += len(batch)
        tried += len(batch)
        for g in got:
            if g:
                with gzip.open(a.out, 'at') as f:   # its own complete block: a killed run never damages earlier rows
                    f.write(json.dumps(g, ensure_ascii=False) + '\n')
                kept += 1
        print(f'{tried} tried, {kept} kept ({kept / tried:.0%}), {len(got and [g for g in got if g])}/{len(batch)} this batch', flush=True)
        time.sleep(a.pause)
    print(f'done: {kept} new rewrites kept of {tried} tried; {kept + len(done)} in {a.out}')


if __name__ == '__main__':
    main()
