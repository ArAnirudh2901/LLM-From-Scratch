"""Leak guard: a training row whose wording matches an evaluation question (real exam questions, held-out rewrites, the
unseen synthetic set), exactly or nearly (3-gram Jaccard >= 0.8, people masked), never reaches training. train_reader
applies it to every row it loads, so no source file can leak an evaluation question, whatever its history.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re

NEAR = 0.8
HELD_OUT = ('data/reworded_dev.jsonl.gz', 'data/unseen_test.jsonl.gz')
PRINTS = 'data/eval_fingerprint.json.gz'   # hashes only (no question text): lets a machine without the private set guard against it


def mask(tokens):
    return [re.sub(r'^@\d+$', '@', re.sub(r'^#\d+$', '#', w)) for w in tokens]


def h(x: str) -> str:
    import hashlib
    return hashlib.blake2b(x.encode(), digest_size=8).hexdigest()


def grams(t):
    return {h(' '.join(t[i:i + 3])) for i in range(len(t) - 2)}


class Guard:
    def __init__(self):
        from eval.run_real import as_text, load_real
        from kinlogic.prepare import prepare
        texts = []
        try:
            texts += [prepare(as_text(r))['tokens'] for r in load_real()]
        except Exception as e:   # no real questions here: guard the held-out files only
            print('leak guard: real questions not loaded:', e, flush=True)
        for f in HELD_OUT:
            if os.path.exists(f):
                texts += [r['tokens'] for r in map(json.loads, gzip.open(f, 'rt')) if isinstance(r.get('tokens'), list)]
        items = {h(' '.join(mask(t))): grams(mask(t)) for t in texts}
        if os.path.exists(PRINTS):   # one entry per question: text and fingerprint of the same question count once
            for e, g in json.load(gzip.open(PRINTS, 'rt')):
                items.setdefault(e, set(g))
        self.exact, self.exact_order = set(items), list(items)
        self.sets = list(items.values())
        df = collections.Counter(x for g in self.sets for x in g)
        rare = max(5, len(self.sets) // 100)   # common 3-grams ("how is @") say nothing about a copy and cost the most
        self.index = collections.defaultdict(list)
        for i, g in enumerate(self.sets):
            for x in g:
                if df[x] <= rare:
                    self.index[x].append(i)
        print(f'leak guard: {len(self.sets)} evaluation questions ({len(texts)} as text here)', flush=True)

    def leaks(self, tokens) -> bool:
        t = mask(tokens)
        if h(' '.join(t)) in self.exact:
            return True
        g = grams(t)
        if len(g) < 6:
            return False
        hits = collections.Counter(i for x in g for i in self.index.get(x, ()))
        for i, n in hits.items():
            if n >= 3:
                inter = len(g & self.sets[i])
                if inter / (len(g) + len(self.sets[i]) - inter) >= NEAR:
                    return True
        return False

    def exact_only(self, tokens) -> bool:
        return h(' '.join(mask(tokens))) in self.exact

    def save_prints(self):
        with gzip.open(PRINTS, 'wt') as f:
            json.dump(sorted([e, sorted(g)] for e, g in zip(self.exact_order, self.sets)), f)

    def clean(self, rows, name=''):
        keep = [r for r in rows if not (isinstance(r, dict) and isinstance(r.get('tokens'), list) and self.leaks(r['tokens']))]
        if len(keep) != len(rows):
            print(f'leak guard: {len(rows) - len(keep)} rows dropped from {name}', flush=True)
        return keep
