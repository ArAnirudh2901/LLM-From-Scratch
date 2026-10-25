"""Audit of the whole training mixture as the reader sees it (pool + every extra file with its weight) against real
exam questions: question kinds, fact types, phrase types, answers, lengths, people per question, and row defects
(missing facts, people in facts but not in the text, no question, too long for the prompt or the decoder, repeats).

    python mixture_audit.py --pool data/dataset_v6 --extra data/dataset_v8_extra.jsonl.gz*3,data/pairs_v1.jsonl.gz --n 30000
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re

QFACT = re.compile(r'(?:^|; )(?:ask|who|count|claim|part|gender\?|prof\?|pair)\b')
FACTS = ['members', 'couples', 'generations', 'singles', 'claim', 'part', 'count', 'who', 'pair', 'gen', 'has', 'only', 'not',
         'same', 'opposite', 'prof', 'male', 'female', 'stranger', 'friend', 'paternal', 'maternal', 'spouse', 'sibling', 'child', 'parent',
         'cousin', 'nephew', 'niece', 'uncle', 'aunt', 'grand', 'great', 'in-law', 'unmarried', 'fathers', 'brothers']


def rows_of(path, k=None):
    out = []
    try:
        for i, line in enumerate(gzip.open(path, 'rt')):
            out.append(json.loads(line))
            if k and i + 1 >= k:
                break
    except (EOFError, OSError):
        pass
    return out


def mixture(pool, extra, n, p_extra, p_deep, rng):
    kinds = [r for f in sorted(glob.glob(f'{pool}/kinds/*.jsonl.gz')) for r in rows_of(f, 20000)]
    deep = [r for f in sorted(glob.glob(f'{pool}/deep/*.jsonl.gz')) for r in rows_of(f, 5000)]
    ex = [(f.split('*')[0], int(f.split('*')[1]) if '*' in f else 1) for f in extra.split(',') if f]
    xs = [(r, f) for f, k in ex for r in rows_of(f) * k]
    out = []
    for _ in range(n):
        u = rng.random()
        if u < p_extra and xs:
            r, f = rng.choice(xs)
            out.append((r, f.split('/')[-1]))
        elif rng.random() < p_deep and deep:
            out.append((rng.choice(deep), 'pool-deep'))
        else:
            out.append((rng.choice(kinds), 'pool-kinds'))
    return out


def lf_words(lf):
    return set(re.findall(r'[a-z?-]+', lf))


def main():
    from eval.real_clean import broken, category
    from eval.run_real import as_text, load_real
    from kinlogic.prepare import prepare
    from phrase_audit import PAT, shares
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--extra', default='')
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--p_extra', type=float, default=0.6)
    ap.add_argument('--p_deep', type=float, default=0.25)
    ap.add_argument('--labels', type=int, default=3000)
    ap.add_argument('--out', default='data/mixture_audit.json')
    a = ap.parse_args()
    rng = random.Random(5)
    mix = mixture(a.pool, a.extra, a.n, a.p_extra, a.p_deep, rng)
    real = [r for r in load_real() if not broken(r)]
    rt = [prepare(as_text(r))['tokens'] for r in real]
    rep = {'sources': dict(collections.Counter(s for _, s in mix).most_common())}

    # defects
    d = collections.Counter()
    seen = collections.Counter()
    for r, _ in mix:
        t, lf = r.get('tokens'), r.get('lf') or ''
        if not isinstance(t, list) or not t:
            d['no tokens'] += 1
            continue
        if not lf.strip():
            d['no facts'] += 1
            continue
        if not QFACT.search(lf):
            d['no question fact'] += 1
        if set(re.findall(r'@\d+', lf)) - set(t):
            d['person in facts not in text'] += 1
        if any(w in ('', '<unk>') for w in t):
            d['empty or unk token'] += 1
        if len(t) > 416:
            d['text longer than the 416 prompt cut'] += 1
        if len(lf.split()) > 90:
            d['facts near/over the 96-token decode limit'] += 1
        seen[' '.join(t)] += 1
    d['repeated texts (extra copies)'] = sum(v - 1 for v in seen.values() if v > 1)
    rep['defects'] = {k: f'{v} ({v / len(mix):.2%})' for k, v in d.most_common()}

    # kinds
    tc = collections.Counter(category({'question': ' '.join(r.get('tokens') or [])}) for r, _ in mix)
    rc = collections.Counter(category(r) for r in real)
    rep['kinds real vs train'] = {k: [round(rc[k] / len(real), 3), round(tc[k] / len(mix), 3)] for k in set(rc) | set(tc)}

    # facts used: train vs what real questions need (gold-like readings from the exact readers where they read)
    from kinlogic.codes import with_codes
    from kinlogic.rules import read as rules_read
    from kinlogic.speech import read as speech_read
    need = collections.Counter()
    nread = 0
    for t in rt:
        lf = rules_read(t) or speech_read(t)
        if lf:
            nread += 1
            need.update(w for w in FACTS if w in lf_words(lf) or any(w in x for x in lf_words(lf)))
    have = collections.Counter()
    for r, _ in mix:
        ws = lf_words(r.get('lf') or '')
        have.update(w for w in FACTS if w in ws or any(w in x for x in ws))
    rep[f'fact words: real exact readings ({nread}) vs train'] = {w: [round(need[w] / max(1, nread), 3), round(have[w] / len(mix), 3)] for w in FACTS}

    # phrases, lengths, people
    ps_r, ps_t = shares([' '.join(t) for t in rt]), shares([' '.join(r.get('tokens') or []) for r, _ in mix])
    rep['phrases real vs train (gaps only)'] = {k: [round(ps_r[k], 3), round(ps_t[k], 3)] for k in PAT
                                                if ps_r[k] > 0.02 and (ps_r[k] > 1.5 * ps_t[k] or ps_t[k] > 4 * ps_r[k])}
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    lr, lt = [len(t) for t in rt], [len(r.get('tokens') or []) for r, _ in mix]
    rep['tokens p50/p90/p99/max real vs train'] = [[q(lr, p) for p in (.5, .9, .99, 1)], [q(lt, p) for p in (.5, .9, .99, 1)]]
    pr = [len({w for w in t if re.fullmatch(r'@\d+', w)}) for t in rt]
    pt = [len({w for w in (r.get('tokens') or []) if re.fullmatch(r'@\d+', w)}) for r, _ in mix]
    rep['people p50/p90/max real vs train'] = [[q(pr, p) for p in (.5, .9, 1)], [q(pt, p) for p in (.5, .9, 1)]]
    bins = lambda xs: dict(sorted(collections.Counter(min(x, 12) for x in xs).items()))
    rep['people per question real'], rep['people per question train'] = bins(pr), bins(pt)

    # answers
    if a.labels:
        from multiprocessing import Pool
        from build_dataset_v8 import BASE, _label, real_answer_shares
        sample = [r for r, _ in rng.sample(mix, min(a.labels, len(mix)))]
        with Pool(4) as p:
            labs = [BASE(x) for x in p.map(_label, sample, chunksize=50) if x]
        ta, ra = collections.Counter(labs), real_answer_shares()
        n = max(1, len(labs))
        rep['answers real vs train (real >= 1%)'] = {k: [round(v, 3), round(ta[k] / n, 3)] for k, v in sorted(ra.items(), key=lambda x: -x[1]) if v >= 0.01}
        rep['answers train-only top'] = {k: round(v / n, 3) for k, v in ta.most_common(40) if ra.get(k, 0) < 0.002}
    json.dump(rep, open(a.out, 'w'), indent=1)
    for k, v in rep.items():
        print(k, json.dumps(v)[:1500])


if __name__ == '__main__':
    main()
