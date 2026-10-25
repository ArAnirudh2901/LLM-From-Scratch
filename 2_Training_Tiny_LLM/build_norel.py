"""No-relation rows: two verified questions joined into one text of two separate families. Every other training row
asks about a linked pair, so a reader never learns to leave two people unlinked; these rows teach it:
  cross     - ask across the two families: answer "not related"
  stranger  - also "X is a stranger to Y" / "X and Y are not related" (fact: Y stranger X)
  distract  - ask inside the first family, the second is a distractor: same answer as the first question
Each row is re-solved by the engine. Appended to data/pairs_v1.jsonl.gz (the r8 notebooks read it).

    python build_norel.py --n 8000
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re

SKIP = re.compile(r'\b(?:members|couples|generations|singles|claim|part|count|who|pair|gen|has|only|not|same|opposite|prof|male|female)\b')
QFORMS = [['how', 'is', 'X', 'related', 'to', 'Y', '?'], ['who', 'is', 'X', 'to', 'Y', '?'], ['what', 'is', 'X', 'to', 'Y', '?'],
          ['how', 'is', 'X', 'related', 'with', 'Y', '?'], ['what', 'is', 'the', 'relation', 'of', 'X', 'to', 'Y', '?']]
STRANGER = [['X', 'is', 'a', 'stranger', 'to', 'Y', '.'], ['X', 'is', 'not', 'related', 'to', 'Y', '.'],
            ['X', 'and', 'Y', 'are', 'strangers', '.'], ['X', 'and', 'Y', 'are', 'not', 'related', '.']]


def split(r):
    """(body tokens up to the last '.', facts without ask, ask) or None for rows this does not fit."""
    t, lf = r.get('tokens'), r.get('lf') or ''
    if not isinstance(t, list) or 'means' in t or t[-1] != '?' or '.' not in t or SKIP.search(lf):
        return None
    facts = [f.strip() for f in lf.split(';') if f.strip()]
    asks = [f for f in facts if f.startswith('ask ')]
    if len(asks) != 1:
        return None
    cut = len(t) - 1 - t[::-1].index('.')
    from kinlogic.kin import KIN
    if any(w in KIN for w in t[cut + 1:]):   # the question itself states a relation: its facts are not in the body
        return None
    body, rest = t[:cut + 1], [f for f in facts if not f.startswith('ask ')]
    named = {w for f in rest for w in f.split() if w.startswith('@')}
    if not named or not named <= set(body):   # someone only named in the question
        return None
    return body, rest, asks[0]


def shift(tokens_or_fact, n, m):
    """@k -> @(k+n), #k -> #(k+m)."""
    f = lambda s: re.sub(r'@(\d+)', lambda x: f'@{int(x.group(1)) + n}', re.sub(r'#(\d+)', lambda x: f'#{int(x.group(1)) + m}', s))
    return [f(w) for w in tokens_or_fact] if isinstance(tokens_or_fact, list) else f(tokens_or_fact)


def make(a, b, rng):
    from train_reader import answers
    (ba, fa, qa), (bb, fb, _) = a, b
    na = 1 + max(int(w[1:]) for w in ba if re.fullmatch(r'@\d+', w))
    bb, fb = shift(bb, na, 50), [shift(f, na, 50) for f in fb]
    pa = sorted({w for w in ba if w.startswith('@')}, key=lambda w: int(w[1:]))
    pb = sorted({w for w in bb if w.startswith('@')}, key=lambda w: int(w[1:]))
    people = pa + pb
    kind = rng.choices(['cross', 'stranger', 'distract'], [0.5, 0.25, 0.25])[0]
    extra, facts = [], fa + fb
    if kind == 'distract':
        y, x = qa.split()[1:3]
    else:
        x, y = (rng.choice(pa), rng.choice(pb)) if rng.random() < 0.5 else (rng.choice(pb), rng.choice(pa))
        if kind == 'stranger':
            s, o = (x, y) if rng.random() < 0.5 else (y, x)
            extra = [s if w == 'X' else o if w == 'Y' else w for w in rng.choice(STRANGER)]
            facts = facts + [f'{o} stranger {s}']
    q = [x if w == 'X' else y if w == 'Y' else w for w in rng.choice(QFORMS)]
    tokens = ba + bb + extra + q if rng.random() < 0.7 else bb + ba + extra + q
    order = {}
    for w in tokens:   # number people by first appearance, as prepare does
        if re.fullmatch(r'@\d+', w) and w not in order:
            order[w] = f'@{len(order)}'
    ren = lambda s: re.sub(r'@\d+', lambda m: order.get(m.group(0), m.group(0)), s)
    tokens = [order.get(w, w) for w in tokens]
    lf = ' ; '.join([ren(f) for f in facts] + [f'ask {ren(y)} {ren(x)}'])
    want = answers(' ; '.join(fa + [qa]), pa) if kind == 'distract' else None
    try:
        got = answers(lf, sorted(order.values(), key=lambda w: int(w[1:])))
    except Exception:
        return None
    if kind == 'distract' and got != want:
        return None
    if kind != 'distract' and any(k not in ('none',) and not str(k).startswith('not family') for k in got):
        return None
    return {'kind': f'norel-{kind}', 'text': ' '.join(tokens), 'tokens': tokens, 'lf': lf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/dataset_v8_extra.jsonl.gz')
    ap.add_argument('--out', default='data/pairs_v1.jsonl.gz')
    ap.add_argument('--n', type=int, default=8000)
    a = ap.parse_args()
    rng = random.Random(41)
    rows = []
    for line in gzip.open(a.src, 'rt'):
        s = split(json.loads(line))
        if s and len(s[0]) < 60:
            rows.append(s)
        if len(rows) >= 60000:
            break
    out, tries = [], 0
    while len(out) < a.n and tries < a.n * 5:
        tries += 1
        r = make(*rng.sample(rows, 2), rng)
        if r:
            out.append(r)
    old = [l for l in gzip.open(a.out, 'rt') if '"norel-' not in l]
    with gzip.open(a.out, 'wt') as f:
        f.write(''.join(old) + ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    print(len(out), 'no-relation rows from', len(rows), 'sources;', len(old), 'rows kept ->', a.out)


if __name__ == '__main__':
    main()
