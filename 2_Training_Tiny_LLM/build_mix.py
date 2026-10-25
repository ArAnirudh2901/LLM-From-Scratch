"""Balanced training mix: every verified row we have (pool rows given exam phrasing as training does, every extra file)
re-weighted by raking (iterative proportional fitting) so that question kinds, phrase types, answers, people per
question and length all match real exam shares at once; long questions with big families, which no source has, are
made by joining verified questions (build_norel.make, 'distract': the answer of the first question is kept).
mixture_audit.py checks the result.

    python build_mix.py --out data/mix_r8.jsonl.gz --rows 300000
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import random
import re
from multiprocessing import Pool

EXTRA = ['data/dataset_v8_extra.jsonl.gz', 'data/phrase_balance.jsonl.gz', 'data/phrase_boost.jsonl.gz', 'data/star_r2.jsonl.gz',
         'data/teach_rows.jsonl.gz', 'data/teach_sg2.jsonl.gz', 'data/pairs_v1.jsonl.gz']
CATS = {'speech': 0.30, 'statements': 0.37, 'puzzle': 0.17, 'coded': 0.12, 'sufficiency': 0.02, 'long text': 0.02}   # coded: the exact code reader reads most
PEOPLE_BINS = [2, 3, 4, 5, 6, 7, 8, 10, 13, 99]
LEN_BINS = [30, 45, 60, 80, 100, 130, 170, 999]


def _bin(x, edges):
    return next(i for i, e in enumerate(edges) if x <= e)


def _style(r, seed):
    from boost_phrases import boost, restyle
    rng = random.Random(seed)
    got = restyle(list(r['tokens']), r['lf'], rng, married=False)
    t = got[0] if got else r['tokens']
    if rng.random() < 0.4:
        t = boost(list(t), rng) or t
    return {**r, 'tokens': t}


def _joined(args):
    from build_norel import PUZZLE, make, split as split_
    split = lambda r: split_(r, PUZZLE)
    a, bs, seed = args
    rng = random.Random(seed)
    s = split(a)
    parts = [split(b) for b in bs]
    if not s or not all(parts):
        return None
    cur = s
    for p in parts:   # each joined family is a distractor; the first question and its answer stay
        r = make(cur, p, rng, kind='distract')
        if not r:
            return None
        cur = split(r)
        if not cur:
            return None
    return {**r, 'kind': 'joined-' + str(a.get('kind'))}


def main():
    from build_dataset_v8 import BASE, _label, real_answer_shares
    from eval.real_clean import broken, category
    from eval.run_real import as_text, load_real
    from kinlogic.prepare import prepare
    from phrase_audit import PAT, shares
    from mixture_audit import rows_of
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/dataset_v6')
    ap.add_argument('--out', default='data/mix_r8.jsonl.gz')
    ap.add_argument('--rows', type=int, default=300000)
    ap.add_argument('--cand', type=int, default=220000)
    ap.add_argument('--joined', type=int, default=24000)
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(9)

    ok = lambda r: isinstance(r.get('tokens'), list) and r['tokens'] and r.get('lf')
    pool = [r for f in sorted(glob.glob(f'{a.pool}/kinds/*.jsonl.gz')) for r in rows_of(f, 12000) if ok(r)]
    pool += [r for f in sorted(glob.glob(f'{a.pool}/deep/**/*.jsonl.gz', recursive=True)) for r in rows_of(f, 400) if ok(r)]
    with Pool(a.workers) as p:
        styled = p.starmap(_style, [(r, i) for i, r in enumerate(pool) if random.Random(i).random() < 0.8], chunksize=200)
    extra = [r for f in EXTRA for r in rows_of(f) if ok(r)]
    cand = pool + styled + extra
    rng.shuffle(cand)
    cand = cand[:a.cand]
    src = [r for r in extra + pool if len(r['tokens']) < 70]
    with Pool(a.workers) as p:
        joined = [r for r in p.map(_joined, [(rng.choice(src), [rng.choice(src) for _ in range(rng.choice([1, 2, 2, 3]))], i)
                                             for i in range(a.joined * 2)], chunksize=100) if r][:a.joined]
    cand += joined
    print(f'{len(pool)} pool, {len(styled)} restyled, {len(extra)} extra, {len(joined)} joined -> {len(cand)} candidates', flush=True)
    with Pool(a.workers) as p:
        labels = [BASE(x) if x else 'other' for x in p.map(_label, cand, chunksize=200)]

    real = [r for r in load_real() if not broken(r)]
    rtxt = [' '.join(prepare(as_text(r))['tokens']) for r in real]
    ra = real_answer_shares()
    seen = collections.Counter(labels)
    uni = 1.0 / max(1, len(seen))
    t_ans = {k: 0.6 * ra.get(k, 0.0) + 0.4 * uni for k in seen}
    t_ans['none'] = max(t_ans.get('none', 0), 0.02)   # no-relation rows stay
    t_ans['other'] = seen['other'] / len(cand)   # unlabelled (claims, counts, several questions): keep their share
    ps = shares(rtxt)
    key = ("X 's R is Y", "Y is X 's R", 'of-chain', 'said / says', 'pointing', 'only (only son', 'grand-/great-', 'if-conditional', 'married to',
           'who is (no comma)', ', who is/was', ', whose R is', 'negation', 'paternal / maternal', 'code symbols', 'Q: how is X related to Y',
           'coordination', 'brother-/sister-', 'pronouns')
    flags = [k for k in PAT if ps[k] > 0.02 and k.startswith(key)]
    pr = collections.Counter(_bin(len(set(re.findall(r'@\d+', t))), PEOPLE_BINS) for t in rtxt)
    lr = collections.Counter(_bin(len(t.split()), LEN_BINS) for t in rtxt)
    floor = lambda c, n: {k: max(v / n, 0.01) for k, v in c.items()}

    texts = [' '.join(r['tokens']) for r in cand]
    feats = {'answer': labels,
             'kind': [category({'question': t}) for t in texts],
             'people': [_bin(len(set(re.findall(r'@\d+', t))), PEOPLE_BINS) for t in texts],
             'length': [_bin(len(r['tokens']), LEN_BINS) for r in cand]}
    targets = {'answer': t_ans, 'kind': CATS, 'people': floor(pr, len(rtxt)), 'length': floor(lr, len(rtxt))}
    for k in flags:
        feats['ph:' + k] = [bool(PAT[k].search(t)) for t in texts]
        targets['ph:' + k] = {True: ps[k], False: 1 - ps[k]}
    w = [1.0] * len(cand)
    for it in range(12):
        for g, vals in feats.items():
            tot = collections.defaultdict(float)
            for x, wi in zip(vals, w):
                tot[x] += wi
            s = sum(tot.values())
            f = {x: (targets[g].get(x, tot[x] / s) / (tot[x] / s)) ** 0.6 if tot[x] else 1.0 for x in tot}   # damped: keeps more rows in play
            w = [min(5.0, max(0.03, wi * f[x])) for x, wi in zip(vals, w)]
    tw = sum(w)
    ess = tw * tw / sum(x * x for x in w)
    out = []
    for r, wi in zip(cand, w):   # expected copies, at most 8 of one row (no memorizing a rare row)
        c = min(8.0, a.rows * wi / tw)
        out += [r] * (int(c) + (rng.random() < c - int(c)))
    rng.shuffle(out)
    with gzip.open(a.out, 'wt') as fo:
        fo.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    rep = {}
    for g, vals in feats.items():
        tot = collections.defaultdict(float)
        for x, wi in zip(vals, w):
            tot[x] += wi
        s = sum(tot.values())
        rep[g] = {str(x): [round(targets[g].get(x, 0), 3), round(tot[x] / s, 3)] for x in sorted(tot, key=lambda x: -tot[x])[:25]}
    rep['max repeats of one row'] = max(collections.Counter(id(r) for r in out).values())
    rep['effective rows (Kish)'] = round(ess)
    json.dump(rep, open(a.out.replace('.jsonl.gz', '_manifest.json'), 'w'), indent=1)
    for g in ('kind', 'people', 'length'):
        print(g, 'target vs got', rep[g])
    print('phrase gaps left:', {g[3:]: v for g, v in rep.items() if g.startswith('ph:') and abs(v['True'][0] - v['True'][1]) > 0.03})
    print('answers (target, got):', dict(list(rep['answer'].items())[:14]))
    print(len(out), 'rows ->', a.out, '| max repeats', rep['max repeats of one row'], '| effective rows', rep['effective rows (Kish)'])


if __name__ == '__main__':
    main()
