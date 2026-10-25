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
FILL = {'married to': 'married to', 'if-conditional (if X is ... , how)': 'if-conditional', 'coordination (R of Y and the S of Z)': 'coordination (and the S of)',
        'who is (no comma)': 'relative clause, no comma (who is)', ', who is/was/has (relative clause)': 'relative clause, comma (, who is)',
        ', whose R is': 'whose clause', 'Q: how is X related to Y': 'exam question forms'}   # phrase type -> boost_phrases.CONSTRUCT rewrite


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


def _fill(job):
    how, con, row, seed = job
    if how == 'j':
        return _joined((row[0], row[1], seed))
    from build_dataset_v8 import _construct
    return _construct((con, row, seed))[1]


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
    ap.add_argument('--strict', action='store_true', help='exit 2 when a real share of 1%+ is still met at under 3/4')
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
    from build_norel import PUZZLE, split as split_
    src = [r for r in extra + pool if len(r['tokens']) < 120 and split_(r, PUZZLE)]   # joinable: each part must split cleanly
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

    targets = {'answer': t_ans, 'kind': CATS, 'people': floor(pr, len(rtxt)), 'length': floor(lr, len(rtxt))}
    targets.update({'ph:' + k: {True: ps[k], False: 1 - ps[k]} for k in flags})

    def featurize(rows, labs):
        texts = [' '.join(r['tokens']) for r in rows]
        coded = lambda t: 'means' in t and re.search(r'@\d+ [^\w\s@#] @\d+', t)   # code symbols between placeholders
        f = {'answer': labs, 'kind': ['coded' if coded(t) else category({'question': t}) for t in texts],
             'people': [_bin(len(set(re.findall(r'@\d+', t))), PEOPLE_BINS) for t in texts],
             'length': [_bin(len(r['tokens']), LEN_BINS) for r in rows]}
        f.update({'ph:' + k: [bool(PAT[k].search(t)) for t in texts] for k in flags})
        return f

    def rake(feats, n):
        w = [1.0] * n
        for _ in range(12):
            for g, vals in feats.items():
                tot = collections.defaultdict(float)
                for x, wi in zip(vals, w):
                    tot[x] += wi
                s_ = sum(tot.values())
                f = {x: (targets[g].get(x, tot[x] / s_) / (tot[x] / s_)) ** 0.6 if tot[x] else 1.0 for x in tot}   # damped: keeps more rows in play
                w = [min(5.0, max(0.03, wi * f[x])) for x, wi in zip(vals, w)]
        return w

    def shares_of(feats, w):
        out = {}
        for g, vals in feats.items():
            tot = collections.defaultdict(float)
            for x, wi in zip(vals, w):
                tot[x] += wi
            s_ = sum(tot.values())
            out[g] = {x: v / s_ for x, v in tot.items()}
        return out

    def gaps(got):
        """(group, value, target, got) wherever a real share of at least 1% is met at under 3/4."""
        return [(g, x, t, got[g].get(x, 0.0)) for g, tg in targets.items() for x, t in tg.items()
                if t >= 0.01 and got[g].get(x, 0.0) < 0.75 * t and not (g.startswith('ph:') and x is False)]

    feats = featurize(cand, labels)
    for rnd in range(3):   # fill what raking cannot reach: verified new rows, then rake again
        w = rake(feats, len(cand))
        todo = gaps(shares_of(feats, w))
        print(f'round {rnd}: {len(todo)} gaps', [(g, str(x), round(t, 3), round(v, 3)) for g, x, t, v in todo][:12], flush=True)
        if not todo:
            break
        jobs = []
        for g, x, t, v in todo:
            k = int(min(20000, (t - v) * len(cand) * 1.5))
            con = FILL.get(g[3:]) if g.startswith('ph:') else None
            if con:   # rewrite rows into the missing construction (engine re-checks changed facts)
                jobs += [('c', con, rng.choice(cand), rng.randrange(10 ** 9)) for _ in range(k)]
            elif g in ('length', 'people') or (g == 'kind' and x == 'long text'):   # long questions, big families: join verified questions
                jobs += [('j', None, (rng.choice(src), [rng.choice(src) for _ in range(rng.choice([2, 3, 3, 4]))]), rng.randrange(10 ** 9)) for _ in range(k)]
        with Pool(a.workers) as p:
            new = [r for r in p.map(_fill, jobs, chunksize=100) if r]
        if not new:
            break
        with Pool(a.workers) as p:
            labs = [BASE(x) if x else 'other' for x in p.map(_label, new, chunksize=200)]
        nf = featurize(new, labs)
        cand += new
        for g in feats:
            feats[g] += nf[g]
        print(f'  +{len(new)} verified rows from {len(jobs)} tries', flush=True)
    from eval.leak import Guard
    guard = Guard()
    w = [0.0 if guard.leaks(r['tokens']) else wi for r, wi in zip(cand, w)]   # no evaluation question, ever
    tw = sum(w)
    ess = tw * tw / sum(x * x for x in w)
    out = []
    for r, wi in zip(cand, w):   # expected copies, at most 8 of one row (no memorizing a rare row)
        c = min(8.0, a.rows * wi / tw)
        out += [r] * (int(c) + (rng.random() < c - int(c)))
    rng.shuffle(out)
    with gzip.open(a.out, 'wt') as fo:
        fo.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    got = shares_of(feats, w)
    unmet = gaps(got)
    rep = {g: {str(x): [round(targets[g].get(x, 0), 3), round(v, 3)] for x, v in sorted(got[g].items(), key=lambda z: -z[1])[:25]} for g in feats}
    rep['unmet (target, got)'] = {f'{g}={x}': [round(t, 3), round(v, 3)] for g, x, t, v in unmet}
    rep['rows'], rep['candidates'], rep['effective rows (Kish)'] = len(out), len(cand), round(ess)
    rep['max repeats of one row'] = max(collections.Counter(id(r) for r in out).values())
    json.dump(rep, open(a.out.replace('.jsonl.gz', '_manifest.json'), 'w'), indent=1)
    print(len(out), 'rows ->', a.out, '| effective rows', round(ess), '| unmet', rep['unmet (target, got)'] or 'none')
    if unmet and a.strict:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
