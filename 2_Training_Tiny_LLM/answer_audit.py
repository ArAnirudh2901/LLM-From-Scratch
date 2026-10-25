"""Answer-label balance: how often each relation answer (father, maternal uncle, co-brother, great-grand-niece, ...)
is the asked answer in training versus real exams. Rare answers in training are answers the reader rarely practised
reading the facts for; build_dataset_v8.py flattens them.

    python answer_audit.py --pool data/dataset_v6 --n 40000
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random


def label(lf, people):
    from kinlogic.engine import solve
    from kinlogic.facts import parse
    st, bad = parse(lf)
    if bad:
        return None
    r = solve(st, people=people)
    qs = r.get('questions') or []
    if not r.get('ok') or not qs or not qs[0].get('answers'):
        return None
    a = qs[0]['answers']
    return a[0].get('base') if len(a) == 1 else 'several'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--out', default='data/answer_audit.json')
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')) + sorted(glob.glob(f'{a.pool}/deep/*.jsonl.gz')):
        rows += [json.loads(l) for _, l in zip(range(a.n // 10), gzip.open(f, 'rt'))]
    random.Random(0).shuffle(rows)
    c = collections.Counter()
    for r in rows[:a.n]:
        if not r.get('lf') or 'ask ' not in r['lf']:
            continue
        people = sorted({t for t in r['tokens'] if t[0] == '@'})
        lb = label(r['lf'], people)
        if lb:
            c[lb] += 1
    tot = sum(c.values())
    print(f'{len(c)} distinct answers in {tot} asked questions')
    for k, v in c.most_common(12):
        print(f'  common {k:28s} {v / tot:6.2%}')
    print('  rarest:', ', '.join(f'{k} ({v})' for k, v in sorted(c.items(), key=lambda x: x[1])[:25]))
    json.dump(dict(c), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
