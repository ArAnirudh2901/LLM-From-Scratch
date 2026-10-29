"""How balanced a training file is: rows per question format, answer, question type, chain depth, people, input and
output length, and constraint words, with the cells that fall under a floor. Formats are read from the text itself, so
any jsonl(.gz) with `text`/`tokens`/`lf` rows works (mix_r8, rewrites, STaR rows, generated candidates).

    python -m eval.balance_report data/mix_r8.jsonl.gz [--floor 0.02] [--limit 100000]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re

QSTART = ('ask', 'who', 'count', 'gender', 'pair', 'prof')


def fmt(r) -> str:
    """The question's surface format (one label; the first rule that applies)."""
    t = (r.get('text') or ' '.join(r.get('tokens') or [])).lower()
    words = len(t.split())
    if re.search(r'\b(means?|denotes?|indicates?|represents?|implies)\b', t) and re.search(r'[^\w\s,.?!\'"()-]|\b[a-z] [+\-*/%$#@&×÷] [a-z]\b', t):
        return 'coded'
    if re.search(r'statement\s*(i|1)\b|statements?\s*:', t):
        return 'data sufficiency'
    if re.search(r'\b(members?|couples?|generations?)\b', t) or re.search(r'\b(count|members|couples|generations)\b', r.get('lf', '').split(';')[-1]):
        return 'family puzzle'
    if '"' in t or '“' in t:
        return 'direct speech'
    if re.search(r'\b(said|says|told|tells|pointing|introduc\w*)\b', t):
        return 'indirect speech'
    if words >= 80:
        return 'long story'
    if re.search(r'\b(who|whom|whose|which)\b.*\b(is|was)\b', t) and t.count(',') >= 2:
        return 'nested clauses'
    if re.search(r"'s\b", t) and not re.search(r'\bof\b', t):
        return 'possessive chain'
    return 'statements'


def depth(lf: str) -> int:
    return sum(1 for s in lf.split(';') if s.split() and s.split()[0] not in QSTART)


def bucket(x: int, edges) -> str:
    for e in edges:
        if x <= e:
            return f'<={e}'
    return f'>{edges[-1]}'


def profile(rows):
    c = {k: collections.Counter() for k in ('format', 'kind', 'answer', 'question', 'facts', 'people', 'input tokens',
                                            'fact tokens', 'constraints', 'options')}
    for r in rows:
        lf = r.get('lf') or ''
        toks = r.get('tokens') or []
        c['format'][fmt(r)] += 1
        c['kind'][r.get('kind') or r.get('format') or '?'] += 1
        c['answer'][str(r.get('truth') or r.get('answer') or '?')] += 1
        q = [s.split()[0] for s in lf.split(';') if s.split() and s.split()[0] in QSTART]
        c['question'][q[-1] if q else 'none'] += 1
        c['facts'][bucket(depth(lf), (1, 2, 3, 4, 6, 8, 12))] += 1
        c['people'][bucket(len({t for t in toks + lf.split() if re.fullmatch(r'@\d+', t)}), (2, 3, 4, 6, 8, 12, 20, 32))] += 1
        c['input tokens'][bucket(len(toks), (20, 40, 80, 160, 240, 320, 416))] += 1
        c['fact tokens'][bucket(len(lf.split()) + 1, (16, 32, 64, 96, 128, 192))] += 1
        for w in ('only', 'not', 'has', 'paternal', 'maternal', 'gen', 'unmarried', 'married', 'prof'):
            if re.search(rf'\b{w}\b', lf):
                c['constraints'][w] += 1
        c['options']['mcq' if re.search(r'\(\s*[a-e1-5]\s*\)', r.get('text') or ' '.join(toks)) else 'open'] += 1
    return c


def show(c, n: int, floor: float, top: int = 25) -> str:
    out = []
    for name, cnt in c.items():
        items = cnt.most_common()
        if name in ('answer', 'kind') and len(items) > top:
            head = items[:top]
            low = [(k, v) for k, v in items if v / n < floor]
            out.append(f'\n{name}: {len(items)} distinct; top {top}:')
            out += [f'  {k:32s} {v:8d} {v / n:7.2%}' for k, v in head]
            out.append(f'  ... {len(low)} {name} values below {floor:.0%} each (together {sum(v for _, v in low) / n:.1%})')
            continue
        out.append(f'\n{name}:')
        out += [f'  {k:32s} {v:8d} {v / n:7.2%}' + ('   <- below floor' if v / n < floor else '') for k, v in items]
    return '\n'.join(out)


def rows_of(path, limit=None):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--floor', type=float, default=0.02, help='flag cells under this share of rows')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    for f in a.files:
        rows = list(rows_of(f, a.limit))
        print(f'=== {f}: {len(rows)} rows')
        print(show(profile(rows), len(rows), a.floor))


if __name__ == '__main__':
    main()
