"""Answer-key check for the real exam questions (evaluation only). Published keys are sometimes wrong; a score on
wrong keys caps every model. Our own pipeline never judges a key (that would grade itself): an independent frontier
model does, in two stages.

1. Screen: every question is solved (high reasoning, exam conventions stated). Agreeing with the key confirms it.
2. Review, only where the screen disagrees: the question is solved again without the key, then the key is reviewed in
   a separate call. Both pointing to the same other option (and the review not calling it ambiguous) corrects the key;
   a review that upholds the key confirms it; anything else marks the question ambiguous (left out of verified scores).

Results: data/real_questions/keys_verified.json {question hash: {status, key, was}}. eval.run_real reports
'<split> verified' next to '<split> fair'. Resumable; run again to continue.

    GEMINI_API_KEY=... python -m eval.verify_keys
"""
from __future__ import annotations

import hashlib
import glob
import json
import os
import re
import time
import urllib.error
import urllib.request

OUT = 'data/real_questions/keys_verified.json'
SHARD = os.environ.get('VERIFY_SHARD', '')   # 'i/n': this process checks every n-th question (one Gemini key each)
MODELS = os.environ.get('VERIFY_MODELS', 'gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview').split(',')
REVIEW_MODELS = os.environ.get('REVIEW_MODELS', 'gemini-3.1-pro-preview,gemini-pro-latest').split(',') + MODELS
CONVENTIONS = """Conventions of these exams: a person whose name is clearly male or female has that gender; pronouns and
words like man, lady, husband give gender; 'cannot be determined' is right only when the answer truly depends on
information the question does not give; in coded questions symbols mean exactly what their definitions say."""

SCREEN = """Solve each blood-relation (family relation) multiple-choice question from Indian competitive exams.
{conv}
Think carefully. Return JSON only: {{"answers": ["a", "c", ...]}} with exactly {n} option letters in order.

{qs}"""

SOLVE = """Solve this blood-relation multiple-choice question from an Indian competitive exam.
{conv}
Think step by step. Return JSON only: {{"answer": "<option letter>", "ambiguous": true or false}}
(ambiguous: two or more options are defensible, or none is right).

{q}"""

REVIEW = """An Indian competitive exam published this blood-relation question with answer key ({key}).
{conv}
Check the key independently and carefully. Return JSON only:
{{"key_correct": true or false, "answer": "<the option letter you find correct>", "ambiguous": true or false, "why": "<one short sentence>"}}

{q}"""

_REST: dict[str, float] = {}


def qhash(r) -> str:
    return hashlib.sha1(re.sub(r'\s+', ' ', r['question']).strip().lower().encode()).hexdigest()[:16]


def gemini(prompt: str, models: list[str]) -> dict:
    """JSON reply from the first model not resting (429/503 rest it 10 min; 400/404 skip it today)."""
    key = os.environ['GEMINI_API_KEY']
    while True:
        live = [m for m in models if _REST.get(m, 0) < time.time()]
        if not live:
            time.sleep(max(5, min(_REST.values()) - time.time()))
            continue
        m = live[0]
        body = {'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {'temperature': 0.0, 'responseMimeType': 'application/json', 'thinkingConfig': {'thinkingLevel': 'high'}}}
        req = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent',
                                     json.dumps(body).encode(), {'Content-Type': 'application/json', 'x-goog-api-key': key})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            text = ''.join(p.get('text', '') for p in r['candidates'][0]['content']['parts'] if not p.get('thought'))
            return json.loads(re.search(r'\{.*\}', text, re.S).group(0))
        except urllib.error.HTTPError as e:
            _REST[m] = time.time() + (86400 if e.code in (400, 404) else 600)
        except Exception:
            _REST[m] = time.time() + 120


def letter(x) -> str:
    return str(x or '').strip().lower().strip('().')[:1]


def main():
    from eval.real_clean import broken
    from eval.run_real import as_text, load_real
    out = OUT.replace('.json', f".{SHARD.replace('/', 'of')}.json") if SHARD else OUT
    done = json.load(open(out)) if os.path.exists(out) else {}
    rows = [r for r in load_real() if not broken(r) and r.get('options') and r.get('answer')]
    if SHARD:
        i, n = map(int, SHARD.split('/'))
        others = {k: v for f in glob.glob(OUT.replace('.json', '*.json')) if f != out for k, v in json.load(open(f)).items()}   # already checked by another run
        rows = [r for j, r in enumerate(rows) if j % n == i and qhash(r) not in others]
    save = lambda: json.dump(done, open(out, 'w'), indent=0)
    todo = [r for r in rows if qhash(r) not in done]
    print(f'{len(rows)} keyed questions, {len(todo)} to check', flush=True)
    for i in range(0, len(todo), 8):   # 1: screen
        chunk = todo[i:i + 8]
        got = gemini(SCREEN.format(conv=CONVENTIONS, n=len(chunk), qs='\n\n'.join(f'Question {j + 1}: {as_text(r)}' for j, r in enumerate(chunk))), MODELS).get('answers', [])
        if len(got) != len(chunk):
            continue
        for r, g in zip(chunk, got):
            done[qhash(r)] = {'status': 'confirmed' if letter(g) == r['answer'] else 'disputed', 'key': r['answer'], 'was': r['answer'], 'screen': letter(g)}
        save()
        print(f'screen {min(i + 8, len(todo))}/{len(todo)}', flush=True)
    for r in rows:   # 2: review the disputed ones
        v = done.get(qhash(r))
        if not v or v['status'] != 'disputed':
            continue
        s = gemini(SOLVE.format(conv=CONVENTIONS, q=as_text(r)), REVIEW_MODELS)
        rv = gemini(REVIEW.format(conv=CONVENTIONS, key=r['answer'], q=as_text(r)), REVIEW_MODELS)
        a, b = letter(s.get('answer')), letter(rv.get('answer'))
        if rv.get('key_correct') is True and not rv.get('ambiguous'):
            v.update(status='confirmed')
        elif a == b and a != r['answer'] and a in 'abcde'[:len(r['options'])] and not (s.get('ambiguous') or rv.get('ambiguous')):
            v.update(status='corrected', key=a, why=str(rv.get('why', ''))[:200])
        else:
            v.update(status='ambiguous', why=str(rv.get('why', ''))[:200])
        save()
    import collections
    print('done', dict(collections.Counter(v['status'] for v in done.values())))


if __name__ == '__main__':
    main()
