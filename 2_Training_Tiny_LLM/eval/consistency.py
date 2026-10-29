"""Repeatability: one question renamed, with its sentences reordered, described by a longer exact path (LogiQAte-style
detours) or with its options shuffled must keep its answer. Groups are generated, so the exact facts and the answer
are known and accuracy is measured alongside agreement.

    python -m eval.consistency --groups 300 --reader gold      # the engine and answer mapping, given exact facts
    python -m eval.consistency --groups 300 --reader exact     # rule readers only (no model)
    python -m eval.consistency --groups 300 --ckpt runs/v10-r8p-final/best.pt --k 16 [--out consistency.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from kinlogic import exam
from kinlogic.generate import KINDS, obfuscation_set
from kinlogic.names import INDIAN_F, INDIAN_M, WORLD_F, WORLD_M, name_gender
from kinlogic.prepare import prepare

POOL = {'M': [*INDIAN_M, *WORLD_M], 'F': [*INDIAN_F, *WORLD_F]}
GENDER = {n: g for g, ns in POOL.items() for n in ns}
LABELS = [['(a)', '(b)', '(c)', '(d)', '(e)'], ['a)', 'b)', 'c)', 'd)', 'e)'], ['(1)', '(2)', '(3)', '(4)', '(5)'],
          ['1.', '2.', '3.', '4.', '5.'], ['A.', 'B.', 'C.', 'D.', 'E.']]
BASES = ('statements', 'nested', 'speech', 'identity', 'introduce', 'story', 'coded', 'puzzle', 'sufficiency')
PRONOUN = re.compile(r'\b(he|she|him|his|her|they|their|them)\b', re.I)


def rename(ex, rng):
    """The same question with every pooled name swapped for another the name table reads the same way (a name it
    holds unisex keeps a gender open, which may rightly change the answer)."""
    text, swap = ex['text'], {}
    for name in sorted(set(ex.get('names', {}).values()), key=len, reverse=True):
        g = GENDER.get(name)
        if g and re.search(rf'\b{re.escape(name)}\b', text):   # lower-cased questions keep their names
            want = name_gender([name])[0]
            pool = [n for n in rng.sample(POOL[g], 40) if n not in text and n not in swap.values()]
            same = [n for n, h in zip(pool, name_gender(pool)) if h == want]
            if not same:
                return None
            swap[name] = same[0]
    if not swap:
        return None
    text = re.sub(r'\b(' + '|'.join(map(re.escape, swap)) + r')\b', lambda m: swap[m.group(1)], text)
    return {**ex, 'text': text, 'names': {k: swap.get(v, v) for k, v in ex['names'].items()}}


def shuffle_options(ex, rng):
    t = ex['text']
    for labels in LABELS:
        i = t.find(' ' + labels[0] + ' ')
        if i < 0:
            continue
        rest = t[i:]
        at = [rest.find(' ' + l + ' ') for l in labels]
        at = [p for p in at if p >= 0]
        if len(at) < 3 or at != sorted(at):
            continue
        opts = [rest[p + len(l) + 2:q].strip() for p, l, q in zip(at, labels, at[1:] + [len(rest)])]
        order = list(range(len(opts)))
        while order == sorted(order):
            rng.shuffle(order)
        return {**ex, 'text': t[:i] + ' ' + ' '.join(f'{l} {opts[j]}' for l, j in zip(labels, order))}
    return None


def remap(lf, names, ents):
    """The facts with each named person's placeholder as the new text numbers them (None if someone is missing)."""
    new = {e['text'].lower(): e['id'] for e in ents if e['kind'] in ('name', 'letter')}
    to = {}
    for old, name in names.items():
        if name.lower() not in new:
            return None
        to[old] = new[name.lower()]
    return re.sub(r'@\d+', lambda m: to.get(m.group(0), m.group(0)), lf)


def reorder(ex, rng):
    """Statement questions with the fact sentences in another order (the question stays last)."""
    if ex.get('kind') not in RELATION_KINDS or shuffle_options(ex, random.Random(0)):   # options and code keys stay put
        return None
    sents = re.split(r'(?<=[.!?])\s+', ex['text'].strip())
    body, rest = [s for s in sents if '?' not in s], [s for s in sents if '?' in s]
    if len(body) < 2 or any(PRONOUN.search(s) for s in body) or any(e['kind'] == 'desc' for e in ex.get('ents', [])):
        return None
    order = list(range(len(body)))
    while order == sorted(order):
        rng.shuffle(order)
    text = ' '.join([body[i] for i in order] + rest)
    lf = remap(ex['lf'], ex.get('names', {}), prepare(text)['ents'])
    return {**ex, 'text': text, 'lf': lf} if lf else None


def groups(n, seed):
    """[(transform, row)] per group; the first row is the base question."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        r = random.Random(rng.getrandbits(64))
        if len(out) % 2 == 0:   # one family and pair at 0, 1 and 2 detours
            rows = obfuscation_set(r)
            if rows:
                out.append([('base', rows[0]), ('detour 1', rows[1]), ('detour 2', rows[2])])
            continue
        base = KINDS[rng.choice(BASES)](r)
        if not base or not base.get('truth'):
            continue
        g = [('base', base)]
        for name, fn in (('rename', rename), ('rename', rename), ('options', shuffle_options), ('reorder', reorder)):
            v = fn(base, r)
            if v:
                g.append((name, v))
        if len(g) > 1:
            out.append(g)
    return out


RELATION_KINDS = ('statements', 'nested', 'speech', 'identity', 'introduce', 'story', 'obfuscated')   # truth = a relation


def norm(s) -> str:
    s = re.sub(r'\([^)]*\)', '', str(s or 'none').lower()).replace('-', ' ')   # "maternal aunt (by marriage)"
    return re.sub(r'\s+', ' ', s).strip()


def agrees(answer, truth) -> bool:
    a, t = norm(answer), norm(truth)
    return a == t or t.endswith(' ' + a) or a.endswith(' ' + t)   # "grandfather" for "paternal grandfather" counts


def evaluate(gs, answer_fn):
    per = collections.defaultdict(lambda: collections.Counter())
    all_same = 0
    for g in gs:
        got = [norm(answer_fn(row).get('answer')) for _, row in g]
        all_same += len(set(got)) == 1
        for (name, row), ans in zip(g, got):
            c = per[name]
            c['n'] += 1
            c['same as base'] += ans == got[0]
            if row.get('kind') in RELATION_KINDS:   # puzzles and sufficiency answer counts, genders, options
                c['scored'] += 1
                c['correct'] += agrees(ans, row.get('truth'))
    report = {name: {'n': c['n'], 'same as base': round(c['same as base'] / c['n'], 4),
                     'correct': round(c['correct'] / c['scored'], 4) if c['scored'] else None} for name, c in per.items()}
    return {'groups': len(gs), 'all variants agree': round(all_same / max(len(gs), 1), 4), 'by transform': report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', type=int, default=300)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--reader', choices=('gold', 'exact', 'model'), default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    mode = a.reader or ('model' if a.ckpt else 'gold')
    if mode == 'model':
        from eval.evaluate import load
        from reader import generate
        from train import get_device
        dev = get_device(force_cpu=a.cpu)
        model, vocab = load(a.ckpt, dev)
        read = lambda batch: generate(model, vocab, batch, dev, k=a.k)
        answer_fn = lambda row: exam.answer(row['text'], read)
    elif mode == 'gold':
        answer_fn = lambda row: exam.answer(row['text'], lambda batch: [[(row['lf'], 0.0)] for _ in batch])
    else:
        answer_fn = lambda row: exam.answer(row['text'], lambda batch: [[('', float('-inf'))] for _ in batch])
    res = {'reader': mode, **evaluate(groups(a.groups, a.seed), answer_fn)}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
