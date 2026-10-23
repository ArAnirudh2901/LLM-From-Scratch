"""Scores a reader on scraped real exam questions (data/real_questions, private): right option, or answer text.

    python -m eval.run_real --ckpt runs/reader-v9/best.pt [--out results.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

from kinlogic.exam import _relation_option, answer, option_moves
from eval.sites import complete
from kinlogic.explain import match_option


def load_real(pattern='data/real_questions/*.jsonl'):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r.get('url', '').lower().endswith('.pdf'):   # page breaks scramble PDF questions
                continue
            r['options'] = [[l, re.sub(r'^\s*\(?[A-Ea-e1-5][).]\s*\.?\s*', '', t)] for l, t in r.get('options') or []]
            if r['options'] and all(l.isdigit() for l, _ in r['options']):   # "1) ... 5)" with the answer as a letter
                r['options'] = [['abcde'[int(l) - 1], t] for l, t in r['options'] if 1 <= int(l) <= 5]
                if r.get('answer') and r['answer'].isdigit():
                    r['answer'] = 'abcde'[int(r['answer']) - 1]
            key = re.sub(r'[^a-z0-9]+', ' ', (r['question'] + ' ' + ' '.join(t for _, t in r['options'])).lower()).strip()
            if key in seen or complete(r):   # once across sites; only questions answerable from their own text
                continue
            seen.add(key)
            rows.append(r)
    return rows


def split_of(r) -> str:
    """'dev' (error analysis, fixes) or 'test' (locked: scores only), half each by question hash."""
    import zlib
    return 'dev' if zlib.crc32(re.sub(r'\s+', ' ', r['question']).strip().lower().encode()) % 2 == 0 else 'test'


def as_text(r):
    opts = ' '.join(f'({l}) {t}' for l, t in r['options'])
    return f"{r['question']} {opts}".strip()


SYNONYMS = [(r'\bgrand[- ]?(uncle|aunt|nephew|niece)\b', r'great-\1'), (r'\bgrand (father|mother|son|daughter)\b', r'grand\1'),
            (r'\bspouse\b', 'husband or wife')]


def _same_words(t):
    t = t.lower().strip()
    for a, b in SYNONYMS:
        t = re.sub(a, b, t)
    return t


def correct(r, res):
    if r.get('answer') and r['options']:
        return res['option'] == r['answer']
    want = r.get('answer_text') or ''
    got = res.get('answer') or ''
    if not want or not got:
        return False
    if match_option(got, [('x', want)]) or match_option(want, [('x', got)]) or _same_words(got) == _same_words(want):
        return True
    mv_w, mv_g = option_moves(re.sub(r"^.*?\bis\s+\w+'s\s+", '', want)), option_moves(got)
    return bool(mv_w and mv_g and [m for m, _ in mv_w] == [m for m, _ in mv_g])


def run(read, rows):
    from eval.real_clean import broken
    by = collections.defaultdict(lambda: [0, 0])
    sp = collections.defaultdict(lambda: [0, 0])
    misses, oks = [], []
    for r in rows:
        res = answer(as_text(r), read)
        ok = correct(r, res)
        oks.append(ok)
        for k in (split_of(r), f'{split_of(r)} fair' if not broken(r) else None):
            if k:
                sp[k][0] += ok
                sp[k][1] += 1
        k = by[r['site']]
        k[0] += ok; k[1] += 1
        if not ok:
            misses.append({'split': split_of(r), 'site': r['site'], 'q': r['question'][:300], 'want': r.get('answer') or r.get('answer_text'),
                           'got': res.get('option') or res.get('answer'), 'facts': res.get('facts', '')})
    total = [sum(v[i] for v in by.values()) for i in range(2)]
    return {'real_acc': total[0] / max(1, total[1]), 'n': total[1],
            'splits': {k: f'{a}/{b} = {a / b:.1%}' for k, (a, b) in sorted(sp.items())},
            'by_site': {k: f'{a}/{b}' for k, (a, b) in sorted(by.items())}, 'ok': oks}, misses


if __name__ == '__main__':
    import torch
    from eval.evaluate import load
    from reader import generate
    from train import get_device
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None, help='first N questions only (quick comparisons)')
    ap.add_argument('--split', default=None, help="only 'dev' or 'test' questions")
    a = ap.parse_args()
    dev = get_device(force_cpu=a.cpu)
    model, vocab = load(a.ckpt, dev)
    rows = [r for r in load_real() if not a.split or split_of(r) == a.split][:a.limit]
    score, misses = run(lambda b: generate(model, vocab, b, dev, k=8), rows)
    print(json.dumps(score, indent=1))
    if a.out:
        json.dump({'score': score, 'misses': misses}, open(a.out, 'w'), indent=1)
