"""The gauntlet: hard blood-relation questions of every exam type, generated from exact facts and scored per category,
each with a renamed twin for repeatability. Categories: lines of descent up to ten generations and cousins to the
fifth degree, 1-3 obfuscating detours, direct and indirect speech, introductions, identities, coded symbols, closed
family puzzles (count, who, gender, pair), data sufficiency, statements to verify, nested descriptions, shuffled
statements, 80-240 word stories and two-family stories; plus the hand-written twisted questions (eval/twisted_set.py).

It is built once with a fixed seed and is evaluation only (eval/leak.py keeps it out of training). Every expected answer
is what the answer pipeline gives from the gold facts; relation rows must also match the family the row came from.
It measures where the reader breaks on hard and twisted forms; it is not real-exam accuracy (the locked real test is).

    python -m eval.gauntlet --build [--per 60]                    # data/gauntlet_v1.jsonl.gz
    python -m eval.gauntlet --reader gold|exact                  # gold must score 100%
    python -m eval.gauntlet --ckpt runs/v10-r8p-final/best.pt --k 16 --out runs/v10-r8p-final/gauntlet.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random

from eval.consistency import RELATION_KINDS, agrees, norm, rename, shuffle_options
from kinlogic import exam
from kinlogic import generate as G

PATH = 'data/gauntlet_v1.jsonl.gz'


def _story(want_link: bool):
    def make(rng):
        from generate_balanced_question_types import (KIN_TYPES, STORY_LENGTHS, connected_long_story,
                                                      disconnected_long_story)
        spec = STORY_LENGTHS[rng.choice(list(STORY_LENGTHS))]
        if want_link:
            return connected_long_story(rng, rng.choice(KIN_TYPES), spec['words'][0], spec['clues'])
        return disconnected_long_story(rng, spec['words'][0], spec['per_component'])
    return make


def _detour(n):
    def make(rng):
        rows = G.obfuscation_set(rng, (n,))
        return rows[0] if rows else None
    return make


CATEGORIES = {
    'generations 3-5': lambda r: G.q_generations(r, *r.choice([(r.randint(3, 5), 0), (0, r.randint(3, 5))])),
    'generations 6-10': lambda r: G.q_generations(r, *r.choice([(r.randint(6, 10), 0), (0, r.randint(6, 10))])),
    'cousins deep': lambda r: G.q_generations(r, r.randint(2, 6), r.randint(2, 6)),
    'detour 1': _detour(1), 'detour 2': _detour(2), 'detour 3': _detour(3),
    'direct speech': lambda r: G.q_speech(r, 'direct'), 'indirect speech': lambda r: G.q_speech(r, 'indirect'),
    'introduction': G.KINDS['introduce'], 'identity': G.KINDS['identity'], 'nested': G.KINDS['nested'],
    'statements': G.KINDS['statements'], 'deep chain': G.KINDS['deep'], 'coded': G.KINDS['coded'],
    'puzzle': G.KINDS['puzzle'], 'sufficiency': G.KINDS['sufficiency'], 'claims': G.KINDS['claims'],
    'long story': _story(True), 'two families': _story(False),
}


def gold_answer(row) -> dict:
    return exam.answer(row['text'], lambda batch: [[(row['lf'], 0.0)] for _ in batch])


def build(per: int = 60, seed: int = 20261028):
    from build_balanced import checked
    rng, rows, seen = random.Random(seed), [], set()
    for cat, make in CATEGORIES.items():
        got, tries = 0, 0
        while got < per and tries < per * 200:
            tries += 1
            r = random.Random(rng.getrandbits(64))
            ex = make(r)
            if not ex or ' '.join(ex['tokens']) in seen or not checked(ex):
                continue
            res = gold_answer(ex)
            if ex.get('kind') in RELATION_KINDS + ('generations',) and ex.get('truth') not in (None, 'none') \
                    and not agrees(res.get('answer'), ex['truth']):
                continue   # the pipeline must give the family's answer from the gold facts
            seen.add(' '.join(ex['tokens']))
            twin = rename(ex, r) or shuffle_options(ex, r)
            rows.append({'category': cat, 'kind': ex.get('kind'), 'text': ex['text'], 'lf': ex['lf'],
                         'truth': ex.get('truth'), 'expect': norm(res.get('answer')), 'expect_option': res.get('option'),
                         'twin': twin['text'] if twin else None})
            got += 1
    rows += twisted_rows()
    return rows


def twisted_rows():
    from eval.twisted_set import T
    return [{'category': f"twisted {it['kind']}", 'kind': it['kind'], 'text': it['text'], 'lf': None, 'truth': None,
             'expect': None, 'expect_option': it['expect'], 'twin': None, 'id': it['id']} for it in T]


def correct(row, res) -> bool:
    if row.get('expect_option'):
        return res.get('option') == row['expect_option']
    return norm(res.get('answer')) == row['expect']


def score(rows, answer_fn):
    per, flips = collections.defaultdict(collections.Counter), collections.Counter()
    for row in rows:
        res = answer_fn(row['text'])
        c = per[row['category']]
        c['n'] += 1
        c['ok'] += correct(row, res)
        if row.get('twin'):
            c['twins'] += 1
            c['twin same'] += norm(answer_fn(row['twin']).get('answer')) == norm(res.get('answer'))
    out = {cat: {'n': c['n'], 'correct': round(c['ok'] / c['n'], 4),
                 'twin same answer': round(c['twin same'] / c['twins'], 4) if c['twins'] else None} for cat, c in per.items()}
    n, ok = sum(c['n'] for c in per.values()), sum(c['ok'] for c in per.values())
    return {'questions': n, 'correct': round(ok / max(n, 1), 4), 'by category': out}


def load(path=PATH):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--per', type=int, default=60)
    ap.add_argument('--reader', choices=('gold', 'exact', 'model'), default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.build:
        rows = build(a.per)
        with gzip.open(PATH, 'wt') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)
        print(f'{len(rows)} questions -> {PATH}:', dict(collections.Counter(r['category'] for r in rows)))
        return
    rows = load()
    mode = a.reader or ('model' if a.ckpt else 'gold')
    if mode == 'model':
        from eval.evaluate import load as load_model
        from reader import generate
        from train import get_device
        dev = get_device(force_cpu=a.cpu)
        model, vocab = load_model(a.ckpt, dev)
        fn = lambda text: exam.answer(text, lambda batch: generate(model, vocab, batch, dev, k=a.k))
    elif mode == 'gold':
        lf = {r['text']: r['lf'] for r in rows if r['lf']}
        lf.update({r['twin']: r['lf'] for r in rows if r['lf'] and r.get('twin')})
        fn = lambda text: exam.answer(text, lambda batch: [[(lf.get(text, ''), 0.0)] for _ in batch])
    else:
        fn = lambda text: exam.answer(text, lambda batch: [[('', float('-inf'))] for _ in batch])
    res = {'reader': mode, **score([r for r in rows if mode != 'gold' or r['lf']], fn)}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
